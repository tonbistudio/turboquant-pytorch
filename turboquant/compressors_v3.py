"""
TurboQuant V3: Community-informed KV cache compression.

Changes from V2 based on findings from 6+ independent implementations:
  1. MSE-only (no QJL) — QJL variance is amplified by softmax, hurting attention
  2. Asymmetric K/V bits — keys need more precision than values
  3. Residual windowing — keep recent tokens in fp16 for generation quality
  4. Layer-adaptive precision — protect early/late layers, compress middle
  5. Bit-packed storage — actual compression, not theoretical

References:
  - scos-lab: MSE > QJL for attention, K/V norm asymmetry
  - back2matching: residual windowing enables working generation
  - TheTom/turboquant_plus: layer-adaptive, attention-gated decoding
  - 0xSero: asymmetric K=3/V=2 on production vLLM
  - llama.cpp community: multiple teams independently dropped QJL
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional

from .lloyd_max import LloydMaxCodebook
from .turboquant import generate_rotation_matrix


class MSECompressor:
    """
    Single-stage MSE-optimal compressor. Used for both keys and values.
    No QJL — all bits go to reconstruction quality.

    This is the core building block. Compress normalizes to unit sphere,
    quantizes with Lloyd-Max, and stores packed indices + norms.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.centroids = LloydMaxCodebook(head_dim, bits).centroids.to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        Compress (B, H, S, D) -> dict with bit-packed indices + norms.
        """
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()

        # Normalize to unit sphere, store norms
        vec_norms = torch.norm(flat, dim=-1)  # (N,)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)

        # Rotate + quantize
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids  # (N, D, levels)
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (N, D)

        # Bit-pack indices: pack multiple indices per byte
        indices_per_byte = 8 // self.bits
        idx_pad = (indices_per_byte - D % indices_per_byte) % indices_per_byte
        idx_flat = indices.long()
        if idx_pad:
            idx_flat = F.pad(idx_flat, (0, idx_pad))
        n_groups = idx_flat.shape[-1] // indices_per_byte
        idx_powers = torch.tensor(
            [2 ** (self.bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long, device=idx_flat.device
        )
        idx_bytes = (idx_flat.reshape(N, n_groups, indices_per_byte) * idx_powers).sum(-1).to(torch.uint8)

        return {
            "idx_bytes": idx_bytes.reshape(B, H, S, n_groups),
            "vec_norms": vec_norms.to(torch.float16).reshape(B, H, S),
            "shape": (B, H, S, D),
            "idx_pad": idx_pad,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress back to (B, H, S, D) tensor."""
        B, H, S, D = compressed["shape"]
        N = B * H * S
        idx_bytes = compressed["idx_bytes"].reshape(N, -1)
        vec_norms = compressed["vec_norms"].reshape(N, 1).float()
        idx_pad = compressed["idx_pad"]

        # Unpack indices
        indices_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        idx_shifts = torch.tensor(
            [self.bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long, device=idx_bytes.device
        )
        indices = ((idx_bytes.long().unsqueeze(-1) >> idx_shifts) & mask).reshape(N, -1)
        if idx_pad:
            indices = indices[:, :D]

        # Reconstruct
        reconstructed = (self.centroids[indices] @ self.Pi) * vec_norms
        return reconstructed.reshape(B, H, S, D)

    def memory_bytes(self, B: int, H: int, S: int) -> dict:
        """Actual memory usage in bytes."""
        D = self.head_dim
        N = B * H * S
        indices_per_byte = 8 // self.bits
        idx_bytes = N * math.ceil(D / indices_per_byte)
        norm_bytes = N * 2  # fp16
        compressed = idx_bytes + norm_bytes
        fp16 = N * D * 2
        return {
            "compressed_bytes": compressed,
            "fp16_bytes": fp16,
            "compression_ratio": fp16 / compressed if compressed > 0 else 0,
        }


class TurboQuantV3:
    """
    Community-informed KV cache compressor.

    Key improvements over V2:
      - MSE-only: no QJL, all bits go to reconstruction quality
      - Asymmetric: separate bit-widths for keys vs values
      - Residual window: recent tokens kept in fp16
      - Layer-adaptive: configurable per-layer bit overrides

    Usage:
        config = TurboQuantV3Config(key_bits=4, value_bits=2, residual_window=128)
        compressor = TurboQuantV3(head_dim=128, config=config, layer_idx=0, device="cuda")
        compressed_k, compressed_v = compressor.compress_kv(keys, values)
        keys_out, values_out = compressor.decompress_kv(compressed_k, compressed_v)
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 4,
        value_bits: int = 2,
        residual_window: int = 128,
        layer_idx: int = 0,
        n_layers: int = 36,
        protected_layers: int = 4,
        protected_bits: int = 8,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.head_dim = head_dim
        self.residual_window = residual_window
        self.device = device

        # Layer-adaptive: first/last N layers get more bits
        is_protected = layer_idx < protected_layers or layer_idx >= (n_layers - protected_layers)
        effective_key_bits = protected_bits if is_protected else key_bits
        effective_value_bits = protected_bits if is_protected else value_bits

        # Cap at 8 bits (uint8 max)
        self.key_bits = min(effective_key_bits, 8)
        self.value_bits = min(effective_value_bits, 8)

        seed_base = seed + layer_idx * 1000
        self.key_compressor = MSECompressor(head_dim, self.key_bits, seed=seed_base, device=device)
        self.val_compressor = MSECompressor(head_dim, self.value_bits, seed=seed_base + 500, device=device)

    @torch.no_grad()
    def compress_kv(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[dict, dict]:
        """
        Compress key and value tensors.
        Input: keys, values — both (B, H, S, D)

        If S > residual_window, the last `residual_window` tokens are kept
        in fp16 (uncompressed) for generation quality. Only older tokens
        are quantized.
        """
        B, H, S, D = keys.shape
        rw = self.residual_window

        if S <= rw:
            # Short sequence — keep everything in fp16
            return (
                {"fp16": keys, "compressed": None, "shape": (B, H, S, D), "split_at": S},
                {"fp16": values, "compressed": None, "shape": (B, H, S, D), "split_at": S},
            )

        # Split: compress old tokens, keep recent in fp16
        split_at = S - rw

        old_keys = keys[:, :, :split_at, :]
        recent_keys = keys[:, :, split_at:, :]
        old_values = values[:, :, :split_at, :]
        recent_values = values[:, :, split_at:, :]

        compressed_k = {
            "compressed": self.key_compressor.compress(old_keys),
            "fp16": recent_keys,
            "shape": (B, H, S, D),
            "split_at": split_at,
        }
        compressed_v = {
            "compressed": self.val_compressor.compress(old_values),
            "fp16": recent_values,
            "shape": (B, H, S, D),
            "split_at": split_at,
        }
        return compressed_k, compressed_v

    @torch.no_grad()
    def decompress_kv(
        self, compressed_k: dict, compressed_v: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress back to full tensors.
        Returns: keys (B, H, S, D), values (B, H, S, D)
        """
        if compressed_k["compressed"] is None:
            return compressed_k["fp16"], compressed_v["fp16"]

        # Decompress old tokens
        old_keys = self.key_compressor.decompress(compressed_k["compressed"])
        old_values = self.val_compressor.decompress(compressed_v["compressed"])

        # Concatenate with fp16 recent tokens
        dtype = compressed_k["fp16"].dtype
        keys = torch.cat([old_keys.to(dtype), compressed_k["fp16"]], dim=2)
        values = torch.cat([old_values.to(dtype), compressed_v["fp16"]], dim=2)
        return keys, values

    def memory_bytes(self, B: int, H: int, S: int) -> dict:
        """
        Report actual memory usage including residual window.
        """
        rw = min(self.residual_window, S)
        compressed_S = max(S - rw, 0)
        fp16_S = rw

        # Compressed portion
        if compressed_S > 0:
            k_mem = self.key_compressor.memory_bytes(B, H, compressed_S)
            v_mem = self.val_compressor.memory_bytes(B, H, compressed_S)
            compressed_bytes = k_mem["compressed_bytes"] + v_mem["compressed_bytes"]
        else:
            compressed_bytes = 0

        # fp16 residual window
        fp16_window_bytes = B * H * fp16_S * self.head_dim * 2 * 2  # keys + values, fp16

        total_compressed = compressed_bytes + fp16_window_bytes
        total_fp16 = B * H * S * self.head_dim * 2 * 2  # full fp16 baseline

        return {
            "compressed_bytes": total_compressed,
            "fp16_bytes": total_fp16,
            "compression_ratio": total_fp16 / total_compressed if total_compressed > 0 else 0,
            "compressed_tokens": compressed_S,
            "fp16_tokens": fp16_S,
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
        }
