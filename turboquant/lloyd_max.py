"""
Lloyd-Max optimal scalar quantizer for the Beta distribution
arising from random rotation of unit-norm vectors.

After rotating a d-dimensional unit vector by a random orthogonal matrix,
each coordinate follows: f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)
supported on [-1, 1].

For practical dimensions (d >= 64), this is well-approximated by N(0, 1/d).
We solve the Lloyd-Max conditions (continuous 1-D k-means) to find optimal centroids.
"""

import torch
import math
from scipy import integrate, special


def beta_pdf(x: float, d: int) -> float:
    """PDF of a single coordinate after random rotation of a d-dim unit vector."""
    if abs(x) >= 1.0:
        return 0.0
    coeff = math.gamma(d / 2) / (math.sqrt(math.pi) * math.gamma((d - 1) / 2))
    return coeff * (1 - x * x) ** ((d - 3) / 2)


def gaussian_approx_pdf(x: float, d: int) -> float:
    """Gaussian approximation N(0, 1/d) -- accurate for d >= 64."""
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def solve_lloyd_max(d: int, bits: int, use_exact: bool = False, max_iter: int = 200, tol: float = 1e-10):
    """
    Solve Lloyd-Max optimal quantizer for the coordinate distribution.

    Args:
        d: vector dimension
        bits: number of quantization bits
        use_exact: if True, use exact Beta PDF; if False, use Gaussian approx
        max_iter: maximum Lloyd-Max iterations
        tol: convergence tolerance

    Returns:
        centroids: sorted tensor of 2^bits optimal centroids
        boundaries: sorted tensor of 2^bits - 1 boundaries between centroids
    """
    n_levels = 2 ** bits
    pdf = (lambda x: beta_pdf(x, d)) if use_exact else (lambda x: gaussian_approx_pdf(x, d))
    sigma = 1.0 / math.sqrt(d)

    # Initialize centroids uniformly in [-3*sigma, 3*sigma]
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for iteration in range(max_iter):
        # Step 1: Compute boundaries (midpoints between adjacent centroids)
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

        # Step 2: Update centroids as conditional expectations E[X | X in partition_i]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]

            numerator, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            denominator, _ = integrate.quad(pdf, a, b)

            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])

        # Check convergence
        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids

        if max_shift < tol:
            break

    # Final boundaries
    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


def compute_expected_distortion(d: int, bits: int, centroids: torch.Tensor, boundaries: torch.Tensor, use_exact: bool = False) -> float:
    """Compute the expected MSE distortion per coordinate for the given quantizer."""
    pdf = (lambda x: beta_pdf(x, d)) if use_exact else (lambda x: gaussian_approx_pdf(x, d))
    sigma = 1.0 / math.sqrt(d)
    n_levels = len(centroids)

    edges = [-3.5 * sigma * 3] + boundaries.tolist() + [3.5 * sigma * 3]
    total_distortion = 0.0

    for i in range(n_levels):
        a, b = edges[i], edges[i + 1]
        c = centroids[i].item()
        dist, _ = integrate.quad(lambda x: (x - c) ** 2 * pdf(x), a, b)
        total_distortion += dist

    return total_distortion


class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for a given dimension and bit-width."""

    def __init__(self, d: int, bits: int, use_exact: bool = False):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        self.centroids, self.boundaries = solve_lloyd_max(d, bits, use_exact)
        self.distortion = compute_expected_distortion(d, bits, self.centroids, self.boundaries, use_exact)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize values to nearest centroid indices."""
        # x: (...,) -> indices: (...,) as uint8/int16
        diffs = (x.unsqueeze(-1) - self.centroids.to(x.device))  # (..., n_levels)
        return diffs.abs().argmin(dim=-1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Map indices back to centroid values."""
        return self.centroids.to(indices.device)[indices]

    def __repr__(self):
        return (
            f"LloydMaxCodebook(d={self.d}, bits={self.bits}, "
            f"levels={self.n_levels}, distortion_per_coord={self.distortion:.6f})"
        )
