"""
Benchmark: TurboQuant PyTorch vs QJL CUDA kernels.

Compares:
1. Pure PyTorch QJL (sign(S @ residual)) - Stage 2 of TurboQuant
2. CUDA fused QJL quantization + score computation
3. Full TurboQuant pipeline: Lloyd-Max (Stage 1) + QJL CUDA (Stage 2)

Usage:
    python -m turboquant.benchmark_cuda
"""

import torch
import time
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.turboquant import TurboQuantMSE, TurboQuantProd, generate_rotation_matrix
from turboquant.cuda_backend import is_cuda_available, QJLSketch, QJLKeyQuantizer


def format_time(ms):
    if ms < 1:
        return f"{ms * 1000:.1f} us"
    return f"{ms:.2f} ms"


def benchmark_qjl_quantize(device="cuda"):
    """Compare QJL quantization: PyTorch vs CUDA kernel."""
    print("=" * 70)
    print("BENCHMARK 1: QJL Quantization Speed")
    print("=" * 70)

    head_dim = 128
    sketch_dim = 256
    group_size = 32
    outliers_count = 8
    n_warmup = 5
    n_iter = 50

    for seq_len in [1024, 4096, 8192, 16384]:
        batch, heads = 1, 32
        num_groups = seq_len // group_size

        # Generate data
        keys = torch.randn(batch, heads, num_groups, group_size, head_dim,
                           device=device, dtype=torch.float16)
        norms = keys.norm(dim=-2)
        _, outlier_idx = norms.topk(outliers_count, dim=-1)
        outlier_idx = outlier_idx.to(torch.uint8).contiguous()

        # ── PyTorch path ──
        proj_matrix = torch.randn(sketch_dim, head_dim, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        for _ in range(n_warmup):
            # Simulate PyTorch QJL: project + sign + bit-pack
            projected = torch.einsum('bhnsd,pd->bhnsp', keys, proj_matrix)
            signs = (projected > 0).to(torch.uint8)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            projected = torch.einsum('bhnsd,pd->bhnsp', keys, proj_matrix)
            signs = (projected > 0).to(torch.uint8)
        torch.cuda.synchronize()
        pytorch_ms = (time.perf_counter() - t0) / n_iter * 1000

        # ── CUDA kernel path ──
        if is_cuda_available():
            sketch = QJLSketch(
                dim=(head_dim, sketch_dim),
                dim_outlier=sketch_dim // 2,
                device=device, rot=True
            )

            torch.cuda.synchronize()
            for _ in range(n_warmup):
                kq, koq, kon = sketch.quantize(keys, outlier_idx)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_iter):
                kq, koq, kon = sketch.quantize(keys, outlier_idx)
            torch.cuda.synchronize()
            cuda_ms = (time.perf_counter() - t0) / n_iter * 1000
            speedup = pytorch_ms / cuda_ms
            print(f"  seq={seq_len:>6d}: PyTorch={format_time(pytorch_ms):>10s}  "
                  f"CUDA={format_time(cuda_ms):>10s}  speedup={speedup:.1f}x")
        else:
            print(f"  seq={seq_len:>6d}: PyTorch={format_time(pytorch_ms):>10s}  "
                  f"CUDA=N/A (kernels not built)")

    print()


def benchmark_attention_scores(device="cuda"):
    """Compare attention score computation: full-precision vs TurboQuant."""
    print("=" * 70)
    print("BENCHMARK 2: Attention Score Computation")
    print("=" * 70)

    head_dim = 128
    sketch_dim = 256
    group_size = 32
    outliers_count = 8
    n_warmup = 5
    n_iter = 100

    for seq_len in [1024, 4096, 8192]:
        batch, heads = 1, 32
        num_groups = seq_len // group_size

        # Generate keys and query
        keys = torch.randn(batch, heads, seq_len, head_dim,
                           device=device, dtype=torch.float16)
        query = torch.randn(batch, heads, 1, head_dim,
                            device=device, dtype=torch.float16)

        # ── Full-precision baseline ──
        torch.cuda.synchronize()
        for _ in range(n_warmup):
            scores_fp = torch.matmul(query, keys.transpose(-2, -1))
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            scores_fp = torch.matmul(query, keys.transpose(-2, -1))
        torch.cuda.synchronize()
        fp_ms = (time.perf_counter() - t0) / n_iter * 1000

        # ── CUDA QJL path ──
        if is_cuda_available():
            sketch = QJLSketch(
                dim=(head_dim, sketch_dim),
                dim_outlier=sketch_dim // 2,
                device=device, rot=True
            )
            quantizer = QJLKeyQuantizer(
                sketch, outliers_count=outliers_count,
                buffer_size=group_size, group_size=group_size,
                qjl_dim=sketch_dim
            )
            quantizer.build_sketch(keys)

            torch.cuda.synchronize()
            for _ in range(n_warmup):
                scores_tq = quantizer.attention_score(query)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_iter):
                scores_tq = quantizer.attention_score(query)
            torch.cuda.synchronize()
            cuda_ms = (time.perf_counter() - t0) / n_iter * 1000

            # Memory comparison
            fp_bytes = keys.numel() * 2  # fp16
            # QJL: sketch_dim/8 bytes per key + norms
            qjl_bytes = (batch * heads * num_groups * group_size * sketch_dim // 8 +
                         batch * heads * num_groups * group_size * 2)  # norms fp16

            print(f"  seq={seq_len:>6d}: FP16={format_time(fp_ms):>10s}  "
                  f"CUDA-QJL={format_time(cuda_ms):>10s}  "
                  f"speedup={fp_ms / cuda_ms:.1f}x  "
                  f"mem: {fp_bytes / 1024 / 1024:.1f}MB -> {qjl_bytes / 1024 / 1024:.1f}MB "
                  f"({fp_bytes / qjl_bytes:.1f}x)")
        else:
            print(f"  seq={seq_len:>6d}: FP16={format_time(fp_ms):>10s}  "
                  f"CUDA-QJL=N/A")

    print()


def benchmark_e2e_pipeline(device="cuda"):
    """End-to-end: Lloyd-Max quantize + QJL residual correction."""
    print("=" * 70)
    print("BENCHMARK 3: End-to-End TurboQuant Pipeline")
    print("=" * 70)

    head_dim = 128
    n_warmup = 3
    n_iter = 20

    for seq_len in [2048, 8192]:
        batch, heads = 1, 32

        keys = torch.randn(batch * heads, seq_len, head_dim, device=device)
        keys = keys / torch.norm(keys, dim=-1, keepdim=True)

        for bits in [2, 3, 4]:
            # ── PyTorch TurboQuantProd ──
            tq = TurboQuantProd(head_dim, bits, seed=42, device=device)

            torch.cuda.synchronize()
            for _ in range(n_warmup):
                compressed = tq.quantize(keys.view(-1, head_dim))
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_iter):
                compressed = tq.quantize(keys.view(-1, head_dim))
            torch.cuda.synchronize()
            pytorch_ms = (time.perf_counter() - t0) / n_iter * 1000

            fp_bytes = keys.numel() * 4
            compressed_bytes = (compressed['mse_indices'].numel() * bits +
                                compressed['qjl_signs'].numel()) / 8
            ratio = fp_bytes / compressed_bytes

            print(f"  seq={seq_len}, bits={bits}: PyTorch TQ={format_time(pytorch_ms):>10s}  "
                  f"compression={ratio:.1f}x")

    print()


def benchmark_accuracy_comparison(device="cuda"):
    """Compare accuracy: PyTorch QJL vs CUDA kernel QJL."""
    print("=" * 70)
    print("BENCHMARK 4: Accuracy - PyTorch vs CUDA (inner product fidelity)")
    print("=" * 70)

    if not is_cuda_available():
        print("  CUDA kernels not available, skipping accuracy comparison.")
        print()
        return

    head_dim = 128
    sketch_dim = 256
    group_size = 32
    outliers_count = 8
    seq_len = 2048
    batch, heads = 1, 8
    num_groups = seq_len // group_size

    keys = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    query = torch.randn(batch, heads, 1, head_dim, device=device, dtype=torch.float16)

    # Full-precision scores
    scores_fp = torch.matmul(query, keys.transpose(-2, -1)).squeeze(-2)  # (B, H, S)

    # CUDA QJL scores
    sketch = QJLSketch(dim=(head_dim, sketch_dim), dim_outlier=sketch_dim // 2,
                       device=device, rot=True)
    quantizer = QJLKeyQuantizer(sketch, outliers_count=outliers_count,
                                buffer_size=group_size, group_size=group_size,
                                qjl_dim=sketch_dim)
    quantizer.build_sketch(keys)
    scores_cuda = quantizer.attention_score(query).squeeze(-2)  # (B, H, S)

    # Metrics
    import torch.nn.functional as F
    cos_sims = []
    top1_matches = 0
    n_checks = 0
    for b in range(batch):
        for h in range(heads):
            fp = scores_fp[b, h]
            cuda = scores_cuda[b, h, :fp.shape[0]]
            cos = F.cosine_similarity(fp.unsqueeze(0).float(), cuda.unsqueeze(0).float()).item()
            cos_sims.append(cos)
            if fp.argmax() == cuda.argmax():
                top1_matches += 1
            n_checks += 1

    avg_cos = sum(cos_sims) / len(cos_sims)
    top1_pct = 100 * top1_matches / n_checks
    print(f"  Cosine similarity: {avg_cos:.6f}")
    print(f"  Top-1 match:       {top1_pct:.1f}% ({top1_matches}/{n_checks})")
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    print()
    print("TurboQuant CUDA Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA kernels available: {is_cuda_available()}")
    print()

    benchmark_qjl_quantize()
    benchmark_attention_scores()
    benchmark_e2e_pipeline()
    benchmark_accuracy_comparison()

    print("=" * 70)
    print("BENCHMARKS COMPLETE")
    print("=" * 70)
