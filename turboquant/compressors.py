"""
TurboQuant KV cache v2: Asymmetric attention.

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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TurboQuantCompressorV2:
    """
    Compressor that stores compressed representations AND supports
    direct inner product computation without full decompression.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        # Rotation matrix
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)

        # Lloyd-Max codebook
        self.centroids = self._solve_codebook(head_dim, self.mse_bits).to(device)

        # QJL matrix
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(seed + 10000)
        self.S = torch.randn(head_dim, head_dim, generator=gen2).to(device)

        # Precompute Pi^T for fast dequant
        self.PiT = self.Pi.T.contiguous()

    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        from scipy import integrate
        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))

        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_centroids = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_centroids.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_centroids

        return torch.tensor(centroids, dtype=torch.float32)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        Compress states: (batch, heads, seq, head_dim) -> compressed dict.
        Stores everything needed for asymmetric inner product computation.
        """
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()

        # Store original norms
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)  # (N, 1)
        flat_norm = flat / (vec_norms + 1e-8)

        # Rotate and quantize
        rotated = flat_norm @ self.PiT.T  # use Pi, so PiT.T = Pi
        # Actually: rotated = flat_norm @ self.Pi.T
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        # MSE reconstruction in original space (for inner product term 1)
        reconstructed_rotated = self.centroids[indices.long()]
        k_mse = (reconstructed_rotated @ self.Pi) * vec_norms  # (N, D) - back in original scale

        # Residual in original space
        residual = flat - k_mse
        residual_norm = torch.norm(residual, dim=-1)  # (N,)

        # QJL signs of residual
        projected = residual @ self.S.T
        signs = (projected >= 0).to(torch.int8) * 2 - 1  # {-1, +1} as int8

        return {
            "k_mse": k_mse.to(torch.float16).reshape(B, H, S, D),  # MSE reconstruction
            "qjl_signs": signs.reshape(B, H, S, D),  # QJL sign bits
            "residual_norm": residual_norm.to(torch.float16).reshape(B, H, S),  # ||r||
            "shape": (B, H, S, D),
        }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute attention scores <Q, K> directly from compressed K.

        Uses the asymmetric estimator:
            <q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, signs_k>

        Args:
            queries: (batch, heads, seq_q, head_dim)
            compressed: dict from compress()

        Returns:
            scores: (batch, heads, seq_q, seq_k)
        """
        k_mse = compressed["k_mse"].float()       # (B, H, S_k, D)
        signs = compressed["qjl_signs"].float()     # (B, H, S_k, D)
        r_norm = compressed["residual_norm"].float() # (B, H, S_k)

        # Term 1: Q @ K_mse^T  (standard matmul)
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))  # (B, H, S_q, S_k)

        # Term 2: QJL correction
        # Project queries through S: S @ q for each query
        # queries: (B, H, S_q, D), S: (D, D)
        q_projected = torch.matmul(queries.float(), self.S.T)  # (B, H, S_q, D)

        # <S@q, signs_k> for all pairs
        qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1))  # (B, H, S_q, S_k)

        # Scale by residual norms and correction factor
        m = self.S.shape[0]
        correction_scale = math.sqrt(math.pi / 2) / m
        # r_norm: (B, H, S_k) -> (B, H, 1, S_k) for broadcasting
        term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)

        return term1 + term2


class TurboQuantCompressorMSE:
    """Simpler MSE-only compressor for values (no QJL needed)."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)
        self.centroids = self._solve_codebook(head_dim, bits).to(device)

    def _solve_codebook(self, d, bits):
        from scipy import integrate
        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)
        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))
        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_c = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_c.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_c[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_c
        return torch.tensor(centroids, dtype=torch.float32)

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


