"""
RotorQuant vs TurboQuant Benchmark

Compares:
1. MSE distortion (reconstruction error)
2. Inner product preservation (bias, RMSE, correlation)
3. Needle-in-haystack retrieval accuracy
4. Equivariance: how well each method handles rotated data
5. Speed and memory

Usage:
    python -m turboquant.benchmark_rotorquant
"""

import torch
import torch.nn.functional as F
import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.turboquant import TurboQuantMSE, TurboQuantProd
from turboquant.rotorquant import RotorQuantMSE, RotorQuantProd


def test_mse_distortion():
    """Compare MSE reconstruction error."""
    print("=" * 70)
    print("TEST 1: MSE Distortion — TurboQuant vs RotorQuant")
    print("=" * 70)

    d = 128
    n = 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  d={d}, n_vectors={n}, device={device}\n")
    print(f"  {'bits':>4s}  {'TQ MSE':>12s}  {'RQ MSE':>12s}  {'theory':>12s}  {'TQ ratio':>10s}  {'RQ ratio':>10s}  {'winner':>8s}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}")

    for bits in [1, 2, 3, 4]:
        tq = TurboQuantMSE(d, bits, seed=42, device=device)
        rq = RotorQuantMSE(d, bits, seed=42, device=device)

        x = torch.randn(n, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # TurboQuant
        x_hat_tq, _ = tq(x)
        mse_tq = ((x - x_hat_tq) ** 2).sum(dim=-1).mean().item()

        # RotorQuant
        x_hat_rq, _ = rq(x)
        mse_rq = ((x - x_hat_rq) ** 2).sum(dim=-1).mean().item()

        # Theoretical bound
        theory = math.sqrt(3) * math.pi / 2 * (1 / (4 ** bits))

        ratio_tq = mse_tq / theory
        ratio_rq = mse_rq / theory
        winner = "RQ" if mse_rq < mse_tq else "TQ" if mse_tq < mse_rq else "TIE"

        print(f"  {bits:>4d}  {mse_tq:>12.6f}  {mse_rq:>12.6f}  {theory:>12.6f}  "
              f"{ratio_tq:>10.3f}  {ratio_rq:>10.3f}  {winner:>8s}")

    print()


def test_inner_product():
    """Compare inner product preservation."""
    print("=" * 70)
    print("TEST 2: Inner Product Unbiasedness — TurboQuant vs RotorQuant")
    print("=" * 70)

    d = 128
    n = 3000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  d={d}, n_pairs={n}, device={device}\n")
    print(f"  {'bits':>4s}  {'':>4s}  {'bias':>10s}  {'RMSE':>10s}  {'corr':>8s}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}")

    for bits in [2, 3, 4]:
        tq = TurboQuantProd(d, bits, seed=42, device=device)
        rq = RotorQuantProd(d, bits, seed=42, device=device)

        x = torch.randn(n, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.randn(n, d, device=device)
        y = y / torch.norm(y, dim=-1, keepdim=True)

        true_ip = (x * y).sum(dim=-1)

        # TurboQuant
        comp_tq = tq.quantize(x)
        est_tq = tq.inner_product(y, comp_tq)
        bias_tq = (est_tq - true_ip).mean().item()
        rmse_tq = ((est_tq - true_ip) ** 2).mean().sqrt().item()
        corr_tq = torch.corrcoef(torch.stack([true_ip, est_tq]))[0, 1].item()

        # RotorQuant
        comp_rq = rq.quantize(x)
        est_rq = rq.inner_product(y, comp_rq)
        bias_rq = (est_rq - true_ip).mean().item()
        rmse_rq = ((est_rq - true_ip) ** 2).mean().sqrt().item()
        corr_rq = torch.corrcoef(torch.stack([true_ip, est_rq]))[0, 1].item()

        print(f"  {bits:>4d}  {'TQ':>4s}  {bias_tq:>+10.6f}  {rmse_tq:>10.6f}  {corr_tq:>8.4f}")
        print(f"  {'':>4s}  {'RQ':>4s}  {bias_rq:>+10.6f}  {rmse_rq:>10.6f}  {corr_rq:>8.4f}")

    print()


def test_needle_in_haystack():
    """Compare retrieval accuracy."""
    print("=" * 70)
    print("TEST 3: Needle-in-Haystack Retrieval")
    print("=" * 70)

    d = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  {'bits':>4s}  {'seq':>6s}  {'TQ':>8s}  {'RQ':>8s}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}")

    for bits in [2, 3, 4]:
        for seq_len in [512, 2048, 8192]:
            keys = torch.randn(seq_len, d, device=device)
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)

            needle_pos = seq_len // 3
            query = keys[needle_pos].clone().unsqueeze(0)

            # TurboQuant
            tq = TurboQuantProd(d, bits, seed=42, device=device)
            comp_tq = tq.quantize(keys)
            ips_tq = tq.inner_product(query.expand(seq_len, -1), comp_tq)
            found_tq = ips_tq.argmax().item() == needle_pos

            # RotorQuant
            rq = RotorQuantProd(d, bits, seed=42, device=device)
            comp_rq = rq.quantize(keys)
            ips_rq = rq.inner_product(query.expand(seq_len, -1), comp_rq)
            found_rq = ips_rq.argmax().item() == needle_pos

            tq_status = "EXACT" if found_tq else "MISS"
            rq_status = "EXACT" if found_rq else "MISS"
            print(f"  {bits:>4d}  {seq_len:>6d}  {tq_status:>8s}  {rq_status:>8s}")

    print()


def test_rotation_equivariance():
    """
    KEY TEST: How well does each method handle pre-rotated data?

    RotorQuant should excel here because its rotor decorrelation
    naturally commutes with rotations in the data.
    """
    print("=" * 70)
    print("TEST 4: Rotation Equivariance (RotorQuant's key advantage)")
    print("=" * 70)

    d = 128
    n = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Testing: quantize(R@x) vs R@quantize(x)")
    print(f"  If equivariant: these should be identical.\n")

    # Create a random rotation matrix
    G = torch.randn(d, d, device=device)
    R, _ = torch.linalg.qr(G)

    x = torch.randn(n, d, device=device)
    x = x / torch.norm(x, dim=-1, keepdim=True)

    # Rotated input
    x_rot = x @ R.T

    print(f"  {'bits':>4s}  {'':>4s}  {'||Q(Rx) - RQ(x)||':>20s}  {'cosine(Q(Rx), RQ(x))':>22s}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*20}  {'─'*22}")

    for bits in [2, 3, 4]:
        # TurboQuant
        tq = TurboQuantMSE(d, bits, seed=42, device=device)
        qx_tq, _ = tq(x)
        qrx_tq, _ = tq(x_rot)
        rqx_tq = qx_tq @ R.T  # rotate the quantized version

        equiv_err_tq = ((qrx_tq - rqx_tq) ** 2).sum(dim=-1).mean().sqrt().item()
        cos_tq = F.cosine_similarity(qrx_tq, rqx_tq, dim=-1).mean().item()

        # RotorQuant
        rq = RotorQuantMSE(d, bits, seed=42, device=device)
        qx_rq, _ = rq(x)
        qrx_rq, _ = rq(x_rot)
        rqx_rq = qx_rq @ R.T

        equiv_err_rq = ((qrx_rq - rqx_rq) ** 2).sum(dim=-1).mean().sqrt().item()
        cos_rq = F.cosine_similarity(qrx_rq, rqx_rq, dim=-1).mean().item()

        print(f"  {bits:>4d}  {'TQ':>4s}  {equiv_err_tq:>20.6f}  {cos_tq:>22.6f}")
        print(f"  {'':>4s}  {'RQ':>4s}  {equiv_err_rq:>20.6f}  {cos_rq:>22.6f}")

    print()


def test_structured_data():
    """
    Test on data with geometric structure (not random).
    Simulates KV cache vectors from attention heads which have
    low-rank + directional structure.
    """
    print("=" * 70)
    print("TEST 5: Structured Data (Low-rank + Directional)")
    print("=" * 70)

    d = 128
    n = 2000
    rank = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create structured data: low-rank with directional bias
    basis = torch.randn(rank, d, device=device)
    basis = basis / torch.norm(basis, dim=-1, keepdim=True)
    coeffs = torch.randn(n, rank, device=device)
    x = coeffs @ basis
    x = x / torch.norm(x, dim=-1, keepdim=True)

    print(f"  Structured vectors: rank={rank}, d={d}, n={n}\n")
    print(f"  {'bits':>4s}  {'':>4s}  {'MSE':>12s}  {'IP corr':>10s}  {'IP RMSE':>10s}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*12}  {'─'*10}  {'─'*10}")

    y = torch.randn(n, d, device=device)
    y = y / torch.norm(y, dim=-1, keepdim=True)
    true_ip = (x * y).sum(dim=-1)

    for bits in [2, 3, 4]:
        # TurboQuant
        tq = TurboQuantProd(d, bits, seed=42, device=device)
        x_hat_tq, _ = tq.mse(x)
        mse_tq = ((x - x_hat_tq) ** 2).sum(dim=-1).mean().item()
        comp_tq = tq.quantize(x)
        est_tq = tq.inner_product(y, comp_tq)
        corr_tq = torch.corrcoef(torch.stack([true_ip, est_tq]))[0, 1].item()
        rmse_tq = ((est_tq - true_ip) ** 2).mean().sqrt().item()

        # RotorQuant
        rq = RotorQuantProd(d, bits, seed=42, device=device)
        x_hat_rq, _ = rq.mse(x)
        mse_rq = ((x - x_hat_rq) ** 2).sum(dim=-1).mean().item()
        comp_rq = rq.quantize(x)
        est_rq = rq.inner_product(y, comp_rq)
        corr_rq = torch.corrcoef(torch.stack([true_ip, est_rq]))[0, 1].item()
        rmse_rq = ((est_rq - true_ip) ** 2).mean().sqrt().item()

        print(f"  {bits:>4d}  {'TQ':>4s}  {mse_tq:>12.6f}  {corr_tq:>10.4f}  {rmse_tq:>10.6f}")
        print(f"  {'':>4s}  {'RQ':>4s}  {mse_rq:>12.6f}  {corr_rq:>10.4f}  {rmse_rq:>10.6f}")

    print()


def test_speed():
    """Compare quantization speed."""
    print("=" * 70)
    print("TEST 6: Speed Benchmark")
    print("=" * 70)

    d = 128
    bits = 3
    n_warmup = 5
    n_iter = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    print(f"  d={d}, bits={bits}\n")

    for n in [1000, 5000, 10000]:
        x = torch.randn(n, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # TurboQuant
        tq = TurboQuantMSE(d, bits, seed=42, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        for _ in range(n_warmup):
            tq(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            tq(x)
        if device == "cuda":
            torch.cuda.synchronize()
        tq_ms = (time.perf_counter() - t0) / n_iter * 1000

        # RotorQuant
        rq = RotorQuantMSE(d, bits, seed=42, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        for _ in range(n_warmup):
            rq(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            rq(x)
        if device == "cuda":
            torch.cuda.synchronize()
        rq_ms = (time.perf_counter() - t0) / n_iter * 1000

        ratio = tq_ms / rq_ms if rq_ms > 0 else float('inf')
        print(f"  n={n:>6d}: TQ={tq_ms:>8.2f} ms  RQ={rq_ms:>8.2f} ms  "
              f"({'RQ' if ratio > 1 else 'TQ'} {max(ratio, 1/ratio):.1f}x faster)")

    print()


def test_parameter_efficiency():
    """Compare parameter counts."""
    print("=" * 70)
    print("TEST 7: Parameter Efficiency")
    print("=" * 70)

    d = 128
    bits = 3

    tq = TurboQuantMSE(d, bits)
    rq = RotorQuantMSE(d, bits)

    tq_params = sum(p.numel() for p in tq.parameters()) + sum(b.numel() for b in tq.buffers())
    rq_params = sum(p.numel() for p in rq.parameters()) + sum(b.numel() for b in rq.buffers())

    print(f"  TurboQuant: {tq_params:,d} parameters/buffers")
    print(f"    - Rotation matrix Pi: {d}x{d} = {d*d:,d}")
    print(f"    - Codebook: {2**bits} centroids")
    print(f"  RotorQuant: {rq_params:,d} parameters/buffers")
    print(f"    - Rotors: {(d+2)//3} groups x 8 components = {((d+2)//3)*8:,d}")
    print(f"    - Codebooks: {2**bits} centroids x 4 grades")
    print(f"  Ratio: {tq_params/rq_params:.1f}x ({'TQ larger' if tq_params > rq_params else 'RQ larger'})")
    print()


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  RotorQuant vs TurboQuant — Clifford Algebra Vector Quantization   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    test_mse_distortion()
    test_inner_product()
    test_needle_in_haystack()
    test_rotation_equivariance()
    test_structured_data()
    test_speed()
    test_parameter_efficiency()

    print("=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
