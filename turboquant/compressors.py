"""
TurboQuant KV cache compressors: Asymmetric attention from compressed KV.

Instead of decompressing KV vectors and feeding them to standard attention,
we compute attention scores DIRECTLY from compressed representations using
the TurboQuant asymmetric inner product estimator.

Key insight from the paper:
  <q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, sign(S@r_k)>

This is unbiased with variance O(1/d), even though k_mse itself has high
per-vector error. The estimator works because QJL corrects the bias in the
inner product space, not in the vector space.

For values, we use MSE-only decompression since the weighted sum in
softmax(scores) @ V averages out per-vector errors.

Classes:
  TurboQuantCompressorV2    - Key compressor with QJL correction (reference impl)
  TurboQuantCompressorMSE   - Value compressor, MSE-only
  PackedKVCompressor        - Memory-correct key compressor using bit-packed storage
"""

import torch
import torch.nn.functional as F
import math

from .lloyd_max import LloydMaxCodebook
from .turboquant import generate_rotation_matrix, generate_qjl_matrix


class TurboQuantCompressorV2:
    """
    Key compressor: stores MSE reconstruction + QJL signs for asymmetric attention.

    NOTE: This is a reference implementation optimized for clarity. The compressed
    dict stores k_mse as a full fp16 tensor (convenient for score computation but
    uses more memory than the theoretical minimum). Use PackedKVCompressor for
    memory-correct compression matching the theoretical ratios.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.centroids = LloydMaxCodebook(head_dim, self.mse_bits).centroids.to(device)
        self.S = generate_qjl_matrix(head_dim, m=head_dim, seed=seed + 10000, device=device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        Compress states: (B, H, S, D) -> compressed dict.
        Stores k_mse (fp16 full tensor) + qjl_signs (int8) + residual_norm (fp16).
        """
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()

        vec_norms = torch.norm(flat, dim=-1, keepdim=True)  # (N, 1)
        flat_norm = flat / (vec_norms + 1e-8)

        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        reconstructed_rotated = self.centroids[indices.long()]
        k_mse = (reconstructed_rotated @ self.Pi) * vec_norms  # (N, D)

        residual = flat - k_mse
        residual_norm = torch.norm(residual, dim=-1)  # (N,)

        projected = residual @ self.S.T
        signs = (projected >= 0).to(torch.int8) * 2 - 1  # {-1, +1}

        return {
            "k_mse": k_mse.to(torch.float16).reshape(B, H, S, D),
            "qjl_signs": signs.reshape(B, H, S, D),
            "residual_norm": residual_norm.to(torch.float16).reshape(B, H, S),
            "shape": (B, H, S, D),
        }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute scores <Q, K> from compressed K using the asymmetric estimator:
            <q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, signs_k>

        Args:
            queries: (B, H, S_q, D)
            compressed: dict from compress()
        Returns:
            scores: (B, H, S_q, S_k)
        """
        k_mse = compressed["k_mse"].float()
        signs = compressed["qjl_signs"].float()
        r_norm = compressed["residual_norm"].float()

        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))

        q_projected = torch.matmul(queries.float(), self.S.T)
        qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1))

        correction_scale = math.sqrt(math.pi / 2) / self.S.shape[0]
        term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)

        return term1 + term2


class TurboQuantCompressorMSE:
    """MSE-only compressor for values (no QJL needed since V is aggregated via softmax)."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.centroids = LloydMaxCodebook(head_dim, bits).centroids.to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (vec_norms + 1e-8)
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)
        return {
            "indices": indices,
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16),
            "shape": (B, H, S, D),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        B, H, S, D = compressed["shape"]
        indices = compressed["indices"].long()
        reconstructed = self.centroids[indices] @ self.Pi
        vec_norms = compressed["vec_norms"].float().unsqueeze(-1)
        return (reconstructed * vec_norms).reshape(B, H, S, D)


class PackedKVCompressor:
    """
    Memory-correct key compressor: stores MSE indices (not reconstructed vectors)
    and packs QJL sign bits to achieve theoretical compression ratios.

    Storage per key vector (d=128, 3-bit):
      - MSE indices:    d * mse_bits bits  = 128 * 2 = 256 bits  (uint8, 1 per coord)
      - QJL sign bits:  d * 1 bits         = 128 bits  (packed via packbits = 16 bytes)
      - residual_norm:  16 bits (fp16)
      - vec_norm:       16 bits (fp16)
      Total:            ~52 bytes  vs. 256 bytes fp16  =>  ~4.9x compression

    vs. TurboQuantCompressorV2 which stores k_mse as fp16 full tensor (256 bytes)
    + signs as int8 (128 bytes) + norm (2 bytes) = 386 bytes -- LARGER than fp16.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        codebook = LloydMaxCodebook(head_dim, self.mse_bits)
        self.centroids = codebook.centroids.to(device)
        self.S = generate_qjl_matrix(head_dim, m=head_dim, seed=seed + 10000, device=device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        Compress (B, H, S, D) -> dict with bit-packed storage.

        Stored tensors:
          indices:       (B, H, S, D) uint8 — MSE codebook indices
          qjl_bits:      (B, H, S, ceil(D/8)) uint8 — 1 bit per QJL sign, packed
          residual_norm: (B, H, S) float16
          vec_norms:     (B, H, S) float16
        """
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()

        vec_norms = torch.norm(flat, dim=-1)             # (N,)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)

        # Stage 1: rotate + Lloyd-Max quantize
        rotated = flat_norm @ self.Pi.T                  # (N, D)
        diffs = rotated.unsqueeze(-1) - self.centroids   # (N, D, n_levels)
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (N, D)

        # Reconstruct for residual (not stored — recomputed during inference)
        k_mse = self.centroids[indices.long()] @ self.Pi * vec_norms.unsqueeze(-1)
        residual = flat - k_mse
        residual_norm = torch.norm(residual, dim=-1)     # (N,)

        # Stage 2: QJL — project residual, pack sign bits (1 bit each)
        projected = residual @ self.S.T                  # (N, D)
        sign_bits = (projected >= 0).to(torch.uint8)     # 0/1, (N, D)

        _powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                                dtype=torch.uint8, device=sign_bits.device)
        qjl_pad = (8 - D % 8) % 8
        if qjl_pad:
            sign_bits = F.pad(sign_bits, (0, qjl_pad))
        n_bytes_qjl = sign_bits.shape[-1] // 8
        qjl_bits = (sign_bits.reshape(N, n_bytes_qjl, 8) * _powers).sum(-1).to(torch.uint8)

        # Pack MSE indices: mse_bits bits each, grouped into uint8 bytes
        indices_per_byte = 8 // self.mse_bits
        idx_pad = (indices_per_byte - D % indices_per_byte) % indices_per_byte
        idx_flat = indices.long()                        # (N, D)
        if idx_pad:
            idx_flat = F.pad(idx_flat, (0, idx_pad))
        n_groups = idx_flat.shape[-1] // indices_per_byte
        idx_powers = torch.tensor(
            [2 ** (self.mse_bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long, device=idx_flat.device
        )
        idx_bytes = (idx_flat.reshape(N, n_groups, indices_per_byte) * idx_powers).sum(-1).to(torch.uint8)

        return {
            "idx_bytes": idx_bytes.reshape(B, H, S, n_groups),
            "qjl_bits": qjl_bits.reshape(B, H, S, -1),
            "residual_norm": residual_norm.to(torch.float16).reshape(B, H, S),
            "vec_norms": vec_norms.to(torch.float16).reshape(B, H, S),
            "shape": (B, H, S, D),
            "qjl_pad": qjl_pad,
            "idx_pad": idx_pad,
        }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute scores <Q, K> from bit-packed compressed K.

        Unpacks indices and sign bits on the fly, then applies the asymmetric
        TurboQuant estimator. Equivalent accuracy to TurboQuantCompressorV2.

        Args:
            queries: (B, H, S_q, D)
            compressed: dict from compress()
        Returns:
            scores: (B, H, S_q, S_k)
        """
        B, H, S_k, D = compressed["shape"]
        N = B * H * S_k
        idx_bytes = compressed["idx_bytes"].reshape(N, -1)
        qjl_bits = compressed["qjl_bits"].reshape(N, -1)
        r_norm = compressed["residual_norm"].reshape(B, H, S_k).float()
        v_norm = compressed["vec_norms"].reshape(N, 1).float()
        qjl_pad = compressed["qjl_pad"]
        idx_pad = compressed["idx_pad"]

        # Unpack MSE indices from bit-packed bytes
        indices_per_byte = 8 // self.mse_bits
        mask = (1 << self.mse_bits) - 1
        idx_shifts = torch.tensor(
            [self.mse_bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long, device=idx_bytes.device
        )
        indices = ((idx_bytes.long().unsqueeze(-1) >> idx_shifts) & mask).reshape(N, -1)
        if idx_pad:
            indices = indices[:, :D]

        # Reconstruct k_mse from unpacked indices
        k_mse = (self.centroids[indices] @ self.Pi) * v_norm   # (N, D)
        k_mse = k_mse.reshape(B, H, S_k, D).float()

        # Unpack sign bits -> {-1, +1}
        _powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                                dtype=torch.uint8, device=qjl_bits.device)
        signs_u8 = ((qjl_bits.unsqueeze(-1) & _powers) > 0).to(torch.uint8)
        signs_u8 = signs_u8.reshape(N, -1)
        if qjl_pad:
            signs_u8 = signs_u8[:, :D]
        signs = (signs_u8.float() * 2 - 1).reshape(B, H, S_k, D)

        # Asymmetric estimator
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
        q_proj = torch.matmul(queries.float(), self.S.T)
        qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1))
        correction_scale = math.sqrt(math.pi / 2) / self.S.shape[0]
        term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)

        return term1 + term2

    def memory_bytes(self, B: int, H: int, S: int) -> dict:
        """Report actual memory usage in bytes for a (B, H, S, D) KV block."""
        D = self.head_dim
        N = B * H * S
        indices_per_byte = 8 // self.mse_bits
        idx_bytes = N * math.ceil(D / indices_per_byte)
        qjl_bytes = N * math.ceil(D / 8)
        norm_bytes = N * 2 * 2
        compressed = idx_bytes + qjl_bytes + norm_bytes
        fp16 = N * D * 2
        return {
            "compressed_bytes": compressed,
            "fp16_bytes": fp16,
            "compression_ratio": fp16 / compressed,
            "breakdown": {
                "idx_bytes": idx_bytes,
                "qjl_bytes": qjl_bytes,
                "norm_bytes": norm_bytes,
            }
        }
