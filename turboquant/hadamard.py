"""
Randomized Hadamard Transform (RHT) as a drop-in rotation for TurboQuant.

Theory
------
TurboQuant Stage 1 requires a random orthogonal rotation Pi such that, after
rotating a unit vector, every coordinate follows the same predictable distribution
(Beta, well-approximated by N(0,1/d)). Any Haar-distributed rotation achieves this.

The original repo uses QR decomposition of a Gaussian matrix:
  - Storage:  O(d^2)  -- a full d x d float32 matrix (64 KB at d=128)
  - Apply:    O(d^2)  -- a dense matrix-vector multiply

The Randomized Hadamard Transform (Ailon & Chazelle, 2009) achieves the same
theoretical guarantees using:
  - Storage:  O(d)    -- just d random ±1 signs (512 bytes at d=128, 128x less)
  - Apply:    O(d log d) -- Fast Walsh-Hadamard butterfly  (18x fewer ops at d=128)

The transform is:  rotate(x) = H @ diag(signs) @ x
where H is the normalized Walsh-Hadamard matrix (H @ H^T = I).

Since H is symmetric and self-inverse (H^2 = I for normalized H), and diag(signs)^{-1}
= diag(signs) (each sign is +-1), the inverse is:
  unrotate(y) = diag(signs) @ H @ y  =  signs * hadamard_transform(y)

Requirements: d must be a power of 2 (holds for all standard transformer head dims:
64, 128, 256). For non-power-of-2 d, fall back to QR rotation.

References
----------
- Ailon & Chazelle (2009): "The Fast Johnson-Lindenstrauss Transform"
- Yu et al. (2016): "Orthogonal Random Features"
- TurboQuant paper (ICLR 2026): uses random orthogonal rotation -- any Haar rotation works
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Normalized Fast Walsh-Hadamard Transform: y = H @ x, where H @ H^T = I.

    O(d log d) butterfly algorithm. d must be a power of 2.

    Args:
        x: (..., d) float tensor, d must be power of 2
    Returns:
        y: (..., d) — same shape, same dtype
    """
    d = x.shape[-1]
    assert is_power_of_two(d), f"hadamard_transform requires d to be a power of 2, got {d}"

    batch_shape = x.shape[:-1]
    h = x.clone()
    n_rounds = int(math.log2(d))

    for s in range(n_rounds):
        half = 1 << s                        # 1, 2, 4, 8, ...
        h = h.reshape(*batch_shape, -1, 2 * half)
        left  = h[..., :half]
        right = h[..., half:]
        h = torch.cat([left + right, left - right], dim=-1)
        h = h.reshape(*batch_shape, d)

    return h / math.sqrt(d)


class HadamardRotation:
    """
    Randomized Hadamard Transform as an O(d log d) drop-in for QR rotation.

    Usage is identical to the Pi matrix in TurboQuantCompressorV2:
        y = rot.rotate(x)           # equivalent to x @ Pi.T
        x_hat = rot.unrotate(y)     # equivalent to y @ Pi

    Memory: O(d) — stores only the d random ±1 signs, not a d×d matrix.
    Compute: O(d log d) per vector — butterfly vs dense GEMM.

    For d=128:
        QR matrix:  16384 multiply-adds,  64 KB storage
        Hadamard:     896 multiply-adds,  512 bytes storage  (128x smaller, 18x faster)
    """

    def __init__(self, d: int, seed: int = 42, device: str = "cpu"):
        if not is_power_of_two(d):
            raise ValueError(
                f"HadamardRotation requires d to be a power of 2, got d={d}. "
                f"Use generate_rotation_matrix() for non-power-of-2 dimensions."
            )
        self.d = d
        self.device = device

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        signs = (torch.randint(0, 2, (d,), generator=gen) * 2 - 1).float()
        self.signs = signs.to(device)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply H @ diag(signs) @ x  (equivalent to x @ Pi.T for a QR matrix Pi)."""
        return hadamard_transform(x * self.signs)

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply diag(signs) @ H @ y  (equivalent to y @ Pi for a QR matrix Pi)."""
        return hadamard_transform(y) * self.signs

    def to(self, device: str) -> "HadamardRotation":
        self.signs = self.signs.to(device)
        self.device = device
        return self

    def __repr__(self) -> str:
        storage_kb = self.d * 4 / 1024
        qr_storage_kb = self.d * self.d * 4 / 1024
        return (
            f"HadamardRotation(d={self.d}, "
            f"storage={storage_kb:.1f}KB vs QR {qr_storage_kb:.0f}KB, "
            f"ops=O(d log d)={self.d * int(math.log2(self.d))} vs O(d^2)={self.d**2})"
        )


class QRRotation:
    """Thin wrapper around a QR-based rotation matrix with .rotate()/.unrotate() API."""

    def __init__(self, d: int, seed: int = 42, device: str = "cpu"):
        from .turboquant import generate_rotation_matrix
        self.Pi = generate_rotation_matrix(d, seed=seed, device=device)
        self.d = d
        self.device = device

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.Pi

    def to(self, device: str) -> "QRRotation":
        self.Pi = self.Pi.to(device)
        self.device = device
        return self

    def __repr__(self) -> str:
        return f"QRRotation(d={self.d}, storage={self.d**2*4/1024:.0f}KB, ops=O(d^2)={self.d**2})"


def make_rotation(d: int, seed: int = 42, device: str = "cpu",
                  hadamard: bool = True) -> "HadamardRotation | QRRotation":
    """
    Factory: return HadamardRotation (O(d log d), preferred) or QRRotation (O(d^2)).

    HadamardRotation requires d to be a power of 2 (64, 128, 256 — all standard
    transformer head dims). Falls back to QRRotation for non-power-of-2 d.
    Both expose the same .rotate() / .unrotate() interface.
    """
    if hadamard and is_power_of_two(d):
        return HadamardRotation(d, seed=seed, device=device)
    else:
        return QRRotation(d, seed=seed, device=device)
