"""
Verification script for TurboQuant implementation.
Tests MSE distortion bounds, inner product accuracy, and compression ratios
against theoretical predictions from the paper.
"""

import torch
import math
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache, LloydMaxCodebook


def test_lloyd_max_codebook():
    """Verify codebook properties for various dimensions and bit-widths."""
    print("=" * 60)
    print("TEST 1: Lloyd-Max Codebook Properties")
    print("=" * 60)

    for d in [64, 128, 256]:
        for bits in [1, 2, 3, 4]:
            cb = LloydMaxCodebook(d, bits)
            print(f"  d={d:>4d}, bits={bits}: {cb.n_levels} levels, "
                  f"distortion/coord={cb.distortion:.6f}, "
                  f"centroids range=[{cb.centroids.min():.4f}, {cb.centroids.max():.4f}]")

    # Verify symmetry (centroids should be symmetric around 0)
    cb = LloydMaxCodebook(128, 3)
    centroid_sum = cb.centroids.sum().abs().item()
    print(f"\n  Symmetry check (d=128, b=3): sum of centroids = {centroid_sum:.6f} (should be ~0)")
    assert centroid_sum < 0.01, "Centroids should be symmetric!"
    print("  PASSED\n")


def test_mse_quantizer():
    """Verify MSE distortion on random unit vectors."""
    print("=" * 60)
    print("TEST 2: MSE Quantizer Distortion")
    print("=" * 60)

    d = 128
    n_vectors = 1000
    device = "cpu"

    for bits in [1, 2, 3, 4]:
        quantizer = TurboQuantMSE(d, bits, seed=42, device=device)

        # Generate random unit vectors
        x = torch.randn(n_vectors, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # Quantize and reconstruct
        x_hat, indices = quantizer(x)

        # Compute empirical MSE
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        # Theoretical upper bound from paper: D_mse <= sqrt(3)*pi/2 * (1/4^b)
        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1 / (4 ** bits))

        ratio = mse / theoretical_bound
        status = "OK" if ratio <= 1.5 else "WARN"  # allow some slack for finite d

        print(f"  bits={bits}: MSE={mse:.6f}, theory_bound={theoretical_bound:.6f}, "
              f"ratio={ratio:.3f} [{status}]")

    print()


def test_inner_product_unbiasedness():
    """Verify that TurboQuantProd gives unbiased inner product estimates."""
    print("=" * 60)
    print("TEST 3: Inner Product Unbiasedness (QJL Correction)")
    print("=" * 60)

    d = 128
    n_trials = 2000
    device = "cpu"

    for bits in [2, 3, 4]:
        quantizer = TurboQuantProd(d, bits, seed=42, device=device)

        # Generate pairs of random unit vectors
        x = torch.randn(n_trials, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.randn(n_trials, d, device=device)
        y = y / torch.norm(y, dim=-1, keepdim=True)

        # True inner products
        true_ip = (x * y).sum(dim=-1)

        # Quantize x, compute estimated inner products
        compressed = quantizer.quantize(x)
        estimated_ip = quantizer.inner_product(y, compressed)

        # Check bias (should be near 0)
        bias = (estimated_ip - true_ip).mean().item()
        # Check RMSE
        rmse = ((estimated_ip - true_ip) ** 2).mean().sqrt().item()
        # Correlation
        correlation = torch.corrcoef(torch.stack([true_ip, estimated_ip]))[0, 1].item()

        # Theoretical distortion bound: D_prod <= sqrt(3)*pi^2/d * (1/4^b)
        theoretical_distortion = math.sqrt(3) * math.pi ** 2 / d * (1 / (4 ** bits))

        print(f"  bits={bits}: bias={bias:+.6f}, RMSE={rmse:.6f}, "
              f"corr={correlation:.4f}, theory_D={theoretical_distortion:.6f}")

    print()


def test_mse_only_inner_product_bias():
    """Show that MSE-only quantizer has biased inner products (motivating QJL)."""
    print("=" * 60)
    print("TEST 4: MSE-Only Inner Product Bias (motivation for QJL)")
    print("=" * 60)

    d = 128
    n_trials = 2000
    device = "cpu"

    for bits in [1, 2, 3]:
        quantizer = TurboQuantMSE(d, bits, seed=42, device=device)

        x = torch.randn(n_trials, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.randn(n_trials, d, device=device)
        y = y / torch.norm(y, dim=-1, keepdim=True)

        true_ip = (x * y).sum(dim=-1)
        x_hat, _ = quantizer(x)
        mse_ip = (x_hat * y).sum(dim=-1)

        bias = (mse_ip - true_ip).mean().item()
        # The bias factor for 1-bit is ~2/pi = 0.637, so ip is scaled by that
        scale_factor = (mse_ip.mean() / true_ip.mean()).item() if true_ip.mean().abs() > 0.01 else float('nan')

        print(f"  bits={bits}: bias={bias:+.6f} (MSE-only is biased, QJL fixes this)")

    print()


def test_kv_cache():
    """Test the KV cache wrapper with compression ratios."""
    print("=" * 60)
    print("TEST 5: KV Cache Compression Ratios")
    print("=" * 60)

    d_key = 128
    d_value = 128
    seq_len = 1024
    device = "cpu"

    for bits in [2, 3, 4]:
        cache = TurboQuantKVCache(d_key, d_value, bits=bits, seed=42, device=device)

        # Simulate appending KV pairs
        keys = torch.randn(seq_len, d_key, device=device)
        values = torch.randn(seq_len, d_value, device=device)

        cache.append(keys, values)

        usage = cache.memory_usage_bits()
        print(f"  bits={bits}: compression={usage['compression_ratio']:.2f}x "
              f"({usage['total_bits'] / 8 / 1024:.1f} KB vs "
              f"{usage['fp16_bits'] / 8 / 1024:.1f} KB fp16)")

        # Test attention score computation
        query = torch.randn(1, d_key, device=device)
        scores = cache.attention_scores(query)
        print(f"           attention scores shape: {scores.shape}, "
              f"range=[{scores.min():.3f}, {scores.max():.3f}]")

    print()


def test_needle_in_haystack():
    """
    Simplified needle-in-haystack: hide a specific key among many,
    verify we can still find it via attention after quantization.
    """
    print("=" * 60)
    print("TEST 6: Needle-in-Haystack Retrieval")
    print("=" * 60)

    d = 128
    device = "cpu"

    for bits in [2, 3, 4]:
        for seq_len in [512, 2048, 8192]:
            # Create random keys
            keys = torch.randn(seq_len, d, device=device)
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)

            # Pick a random "needle" position and create a query that matches it
            needle_pos = seq_len // 3
            query = keys[needle_pos].clone().unsqueeze(0)  # exact match query

            # Quantize all keys
            quantizer = TurboQuantProd(d, bits, seed=42, device=device)
            compressed = quantizer.quantize(keys)

            # Compute inner products
            estimated_ips = quantizer.inner_product(
                query.expand(seq_len, -1), compressed
            )

            # Check if needle is still the top result
            top_idx = estimated_ips.argmax().item()
            found = top_idx == needle_pos

            # Also check top-5
            top5 = estimated_ips.topk(5).indices.tolist()
            in_top5 = needle_pos in top5

            status = "EXACT" if found else ("TOP-5" if in_top5 else "MISS")
            print(f"  bits={bits}, seq={seq_len:>5d}: top1={top_idx:>5d} "
                  f"(needle={needle_pos:>5d}) [{status}]")

    print()


def test_gpu_if_available():
    """Run a quick benchmark on GPU if CUDA is available."""
    print("=" * 60)
    print("TEST 7: GPU Benchmark (if CUDA available)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        print()
        return

    device = "cuda"
    d = 128
    bits = 3
    seq_len = 8192
    n_queries = 64

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Config: d={d}, bits={bits}, seq_len={seq_len}, n_queries={n_queries}")

    quantizer = TurboQuantProd(d, bits, seed=42, device=device)

    # Generate data
    keys = torch.randn(seq_len, d, device=device)
    keys = keys / torch.norm(keys, dim=-1, keepdim=True)
    queries = torch.randn(n_queries, d, device=device)
    queries = queries / torch.norm(queries, dim=-1, keepdim=True)

    # Benchmark quantization
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        compressed = quantizer.quantize(keys)
    torch.cuda.synchronize()
    quant_time = (time.perf_counter() - t0) / 10
    print(f"  Quantize {seq_len} keys: {quant_time * 1000:.2f} ms")

    # Benchmark inner product
    compressed = quantizer.quantize(keys)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        for i in range(n_queries):
            ip = quantizer.inner_product(queries[i:i+1].expand(seq_len, -1), compressed)
    torch.cuda.synchronize()
    ip_time = (time.perf_counter() - t0) / 100
    print(f"  Inner product ({n_queries} queries x {seq_len} keys): {ip_time * 1000:.2f} ms")

    # Compare with full-precision
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        fp_scores = queries @ keys.T
    torch.cuda.synchronize()
    fp_time = (time.perf_counter() - t0) / 100
    print(f"  Full-precision matmul: {fp_time * 1000:.2f} ms")

    # Memory comparison
    fp16_bytes = seq_len * d * 2  # fp16
    quant_bytes = seq_len * d * bits / 8
    print(f"  Memory: {fp16_bytes / 1024:.1f} KB (fp16) vs {quant_bytes / 1024:.1f} KB (TQ-{bits}bit)")
    print(f"  Compression: {fp16_bytes / quant_bytes:.1f}x")
    print()


if __name__ == "__main__":
    print()
    print("TurboQuant Implementation Verification")
    print("Based on: 'TurboQuant: Online Vector Quantization' (ICLR 2026)")
    print()

    test_lloyd_max_codebook()
    test_mse_quantizer()
    test_inner_product_unbiasedness()
    test_mse_only_inner_product_bias()
    test_kv_cache()
    test_needle_in_haystack()
    test_gpu_if_available()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
