"""
RotorQuant: Reimagining TurboQuant with Clifford Algebra

Instead of TurboQuant's random orthogonal matrix Π (via QR decomposition),
RotorQuant uses Clifford rotors R = exp(B/2) for decorrelation.

Why this is better for geometric data:
1. Rotor sandwich R x R̃ preserves the FULL algebraic structure
   (inner products, outer products, grades) — not just norms
2. Rotors compose naturally: R₂(R₁ x R̃₁)R̃₂ = (R₂R₁) x (R₂R₁)~
3. Grade-aware quantization: different grades can use different bit budgets
4. The bivector structure of rotors means we only need 3 parameters
   (not d² for a full rotation matrix) — massive parameter savings

Algorithm:
  Stage 1 (MSE): Embed vectors as Cl(3,0) multivectors → rotor sandwich →
                  grade-aware Lloyd-Max quantization per component
  Stage 2 (QJL): 1-bit sign quantization on residual for unbiased inner products
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict

from .clifford import (
    MV_DIM, geometric_product, reverse, make_random_rotor,
    rotor_sandwich, embed_vectors_as_multivectors,
    extract_vectors_from_multivectors, multivector_norm_sq,
)
from .lloyd_max import LloydMaxCodebook


class RotorQuantMSE(nn.Module):
    """
    Stage 1: MSE-optimal quantizer using Clifford rotors.

    Instead of Π @ x (matrix multiply), we do R x R̃ (rotor sandwich).
    Then per-component Lloyd-Max quantization on the rotated multivector.
    """

    def __init__(self, d: int, bits: int, seed: int = 42,
                 grade_bits: Optional[Dict[str, int]] = None,
                 device: str = "cpu"):
        """
        Args:
            d: original vector dimension
            bits: default bits per component
            grade_bits: optional per-grade bit override, e.g.
                        {'scalar': 2, 'vector': 3, 'bivector': 2, 'trivector': 1}
            seed: random seed
            device: torch device
        """
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device

        # Compute how many multivector groups we need
        self.n_groups = (d + 2) // 3  # ceil(d/3)
        self.mv_dim = self.n_groups * MV_DIM  # total components

        # Grade-aware bit allocation
        if grade_bits is None:
            grade_bits = {
                'scalar': bits,
                'vector': bits,
                'bivector': bits,
                'trivector': max(bits - 1, 1),
            }
        self.grade_bits = grade_bits

        # Create per-grade codebooks
        # Using d_eff = n_groups * 8 for the Lloyd-Max distribution parameter
        d_eff = max(self.n_groups * MV_DIM, 64)
        self.codebooks = nn.ModuleDict()
        for grade_name, gb in grade_bits.items():
            cb = LloydMaxCodebook(d_eff, gb)
            self.register_buffer(f'centroids_{grade_name}',
                                 cb.centroids.to(device))

        # Grade masks for indexing into multivector
        # [scalar, e1, e2, e3, e12, e13, e23, e123]
        self.grade_map = {
            'scalar':   [0],
            'vector':   [1, 2, 3],
            'bivector': [4, 5, 6],
            'trivector': [7],
        }

        # Pre-compute random rotors (one per group for decorrelation)
        rotors = []
        for i in range(self.n_groups):
            r = make_random_rotor((), device=device, seed=seed + i)
            rotors.append(r)
        self.register_buffer('rotors', torch.stack(rotors))  # (n_groups, 8)

    def _apply_rotors(self, mv: torch.Tensor) -> torch.Tensor:
        """Apply per-group rotor sandwich: R_i x_i R̃_i"""
        # mv: (..., n_groups, 8)
        # rotors: (n_groups, 8) → broadcast over batch dims
        return rotor_sandwich(self.rotors, mv)

    def _unapply_rotors(self, mv: torch.Tensor) -> torch.Tensor:
        """Inverse rotor sandwich: R̃_i x_i R_i"""
        rotor_rev = reverse(self.rotors)
        return rotor_sandwich(rotor_rev, mv)

    def _quantize_grade(self, x: torch.Tensor, grade_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize components of a specific grade."""
        centroids = getattr(self, f'centroids_{grade_name}')
        diffs = x.unsqueeze(-1) - centroids  # (..., n_components, n_levels)
        indices = diffs.abs().argmin(dim=-1)
        x_q = centroids[indices]
        return x_q, indices

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize vectors via rotor + grade-aware Lloyd-Max.

        x: (..., d) input vectors
        Returns: (indices_dict, metadata)
        """
        # Embed as multivectors
        mv = embed_vectors_as_multivectors(x)  # (..., n_groups, 8)

        # Apply rotor decorrelation
        mv_rot = self._apply_rotors(mv)

        # Grade-aware quantization
        mv_q = torch.zeros_like(mv_rot)
        all_indices = {}

        for grade_name, component_indices in self.grade_map.items():
            grade_data = mv_rot[..., component_indices]  # (..., n_groups, n_components)
            flat = grade_data.reshape(*grade_data.shape[:-1], -1)
            q_flat, idx = self._quantize_grade(flat, grade_name)
            q_data = q_flat.reshape_as(grade_data)
            mv_q[..., component_indices] = q_data
            all_indices[grade_name] = idx

        return mv_q, all_indices

    def dequantize(self, indices: dict) -> torch.Tensor:
        """Reconstruct vectors from quantized indices."""
        sample_centroids = getattr(self, 'centroids_scalar')
        # Infer batch shape: scalar indices are (batch..., n_groups * 1)
        # so batch_shape is everything except the last dim
        scalar_idx = indices['scalar']
        flat_batch = scalar_idx.shape[0] if scalar_idx.dim() >= 1 else 1
        # Number of total flattened scalar components = n_groups * 1
        # So idx shape is (batch, n_groups * n_components_for_grade)

        mv_q = torch.zeros(flat_batch, self.n_groups, MV_DIM,
                           dtype=sample_centroids.dtype,
                           device=sample_centroids.device)

        for grade_name, component_indices in self.grade_map.items():
            centroids = getattr(self, f'centroids_{grade_name}')
            idx = indices[grade_name]  # (batch, n_groups * n_comps)
            values = centroids[idx]
            n_components = len(component_indices)
            # Reshape: (batch, n_groups * n_comps) → (batch, n_groups, n_comps)
            values = values.reshape(flat_batch, self.n_groups, n_components)
            mv_q[..., component_indices] = values

        # Undo rotor rotation
        mv_recon = self._unapply_rotors(mv_q)

        # Extract vectors
        return extract_vectors_from_multivectors(mv_recon, self.d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Full quantize-dequantize cycle."""
        mv_q, indices = self.quantize(x)
        # Undo rotor
        mv_recon = self._unapply_rotors(mv_q)
        x_hat = extract_vectors_from_multivectors(mv_recon, self.d)
        return x_hat, indices


class RotorQuantProd(nn.Module):
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.

    Uses (b-1)-bit rotor MSE quantizer + 1-bit QJL on residuals.
    The QJL operates in the original vector space (not multivector space)
    since inner products are computed there.
    """

    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None,
                 seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        # Stage 1: Rotor MSE quantizer
        self.mse = RotorQuantMSE(d, self.mse_bits, seed=seed, device=device)

        # Stage 2: QJL projection matrix (same as TurboQuant)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed + 1)
        S = torch.randn(self.qjl_dim, d, generator=gen)
        self.register_buffer("S", S.to(device))

    def quantize(self, x: torch.Tensor) -> dict:
        """Full RotorQuant quantization."""
        # Stage 1: Rotor MSE
        x_hat, mse_indices = self.mse(x)

        # Residual
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)

        # Stage 2: QJL sign quantization on residual
        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0

        return {
            'mse_indices': mse_indices,
            'qjl_signs': qjl_signs,
            'residual_norm': residual_norm.squeeze(-1),
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        """Reconstruct from MSE component."""
        return self.mse.dequantize(compressed['mse_indices'])

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Unbiased inner product estimate: <y, x>.

        Same QJL correction as TurboQuant — the math doesn't change
        because QJL operates in vector space.
        """
        x_mse = self.mse.dequantize(compressed['mse_indices'])
        term1 = (y * x_mse).sum(dim=-1)

        y_projected = y @ self.S.T
        qjl_ip = (y_projected * compressed['qjl_signs']).sum(dim=-1)

        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = compressed['residual_norm'] * correction_scale * qjl_ip

        return term1 + term2

    def forward(self, x: torch.Tensor) -> dict:
        return self.quantize(x)


class RotorQuantKVCache:
    """
    KV cache using RotorQuant compression.
    Drop-in replacement for TurboQuantKVCache.
    """

    def __init__(self, d_key: int, d_value: int, bits: int = 3,
                 seed: int = 42, device: str = "cpu"):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device

        self.key_quantizer = RotorQuantProd(d_key, bits, seed=seed, device=device)
        self.value_quantizer = RotorQuantMSE(d_value, bits, seed=seed + 100, device=device)

        self.key_cache = []
        self.value_cache = []

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)

        compressed_keys = self.key_quantizer.quantize(flat_keys)
        _, value_indices = self.value_quantizer(flat_values)

        self.key_cache.append(compressed_keys)
        self.value_cache.append(value_indices)

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        scores = []
        for cached in self.key_cache:
            s = self.key_quantizer.inner_product(queries, cached)
            scores.append(s)
        return torch.cat(scores, dim=-1) if scores else torch.tensor([])

    def get_values(self) -> torch.Tensor:
        values = []
        for indices in self.value_cache:
            v = self.value_quantizer.dequantize(indices)
            values.append(v)
        return torch.cat(values, dim=0) if values else torch.tensor([])

    def __len__(self):
        return sum(
            c['qjl_signs'].shape[0] for c in self.key_cache
        ) if self.key_cache else 0
