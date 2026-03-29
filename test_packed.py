"""
Test and benchmark PackedKVCompressor vs TurboQuantCompressorV2.

Validates:
1. PackedKVCompressor achieves real memory compression matching theory
2. Attention score accuracy matches the reference implementation
3. Memory breakdown: indices + packed bits + norms vs fp16
"""

import torch
import torch.nn.functional as F
import sys, os, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turboquant import TurboQuantCompressorV2, TurboQuantCompressorMSE, PackedKVCompressor


def tensor_bytes(t: torch.Tensor) -> int:
    return t.nelement() * t.element_size()


def compressed_v2_bytes(c: dict) -> int:
    return sum(tensor_bytes(v) for v in c.values() if isinstance(v, torch.Tensor))


def compressed_packed_bytes(c: dict) -> int:
    return sum(tensor_bytes(v) for v in c.values() if isinstance(v, torch.Tensor))


def test_memory_comparison():
    print("=" * 65)
    print("MEMORY: TurboQuantCompressorV2 vs PackedKVCompressor")
    print("=" * 65)
    print(f"  {'Config':<20} {'fp16':>8} {'V2 (ref)':>10} {'Packed':>10} {'V2 ratio':>10} {'Packed ratio':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H = 1, 2

    for bits in [2, 3, 4]:
        for S in [1024, 4096]:
            D = 128
            states = torch.randn(B, H, S, D, device=device)
            fp16_bytes = states.numel() * 2  # fp16

            v2 = TurboQuantCompressorV2(D, bits, seed=42, device=device)
            pk = PackedKVCompressor(D, bits, seed=42, device=device)

            c_v2 = v2.compress(states)
            c_pk = pk.compress(states)

            b_v2 = compressed_v2_bytes(c_v2)
            b_pk = compressed_packed_bytes(c_pk)

            label = f"bits={bits}, S={S}"
            print(f"  {label:<20} {fp16_bytes/1024:>6.0f}KB "
                  f"{b_v2/1024:>8.0f}KB "
                  f"{b_pk/1024:>8.0f}KB "
                  f"{fp16_bytes/b_v2:>9.2f}x "
                  f"{fp16_bytes/b_pk:>10.2f}x")
    print()


def test_score_accuracy():
    print("=" * 65)
    print("ACCURACY: PackedKVCompressor vs TurboQuantCompressorV2")
    print("(both should give nearly identical attention scores)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, S_k, S_q, D = 1, 2, 512, 8, 128

    for bits in [2, 3, 4]:
        states = torch.randn(B, H, S_k, D, device=device)
        queries = torch.randn(B, H, S_q, D, device=device)

        v2 = TurboQuantCompressorV2(D, bits, seed=42, device=device)
        pk = PackedKVCompressor(D, bits, seed=42, device=device)

        c_v2 = v2.compress(states)
        c_pk = pk.compress(states)

        scores_v2 = v2.asymmetric_attention_scores(queries, c_v2)
        scores_pk = pk.asymmetric_attention_scores(queries, c_pk)

        # Ground truth
        scores_gt = torch.matmul(queries.float(), states.float().transpose(-2, -1))

        # Compare packed vs v2
        cos_v2_pk = F.cosine_similarity(
            scores_v2.reshape(1, -1), scores_pk.reshape(1, -1)
        ).item()

        # Compare both vs ground truth
        cos_v2_gt = F.cosine_similarity(
            scores_v2.reshape(1, -1), scores_gt.reshape(1, -1)
        ).item()
        cos_pk_gt = F.cosine_similarity(
            scores_pk.reshape(1, -1), scores_gt.reshape(1, -1)
        ).item()

        print(f"  bits={bits}: V2<->Packed cosine={cos_v2_pk:.6f}  "
              f"V2<->GT={cos_v2_gt:.4f}  Packed<->GT={cos_pk_gt:.4f}")
    print()


def test_theoretical_vs_actual():
    print("=" * 65)
    print("THEORY CHECK: PackedKVCompressor.memory_bytes() vs actual tensors")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = 128
    B, H, S = 1, 2, 1024

    for bits in [2, 3, 4]:
        pk = PackedKVCompressor(D, bits, seed=42, device=device)
        states = torch.randn(B, H, S, D, device=device)
        c = pk.compress(states)

        theory = pk.memory_bytes(B, H, S)
        tensor_actual = sum(tensor_bytes(v) for v in c.values() if isinstance(v, torch.Tensor))

        print(f"  bits={bits}: theory={theory['compressed_bytes']/1024:.1f}KB  "
              f"actual={tensor_actual/1024:.1f}KB  "
              f"ratio={theory['fp16_bytes']/tensor_actual:.2f}x")
    print()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}\n")

    test_memory_comparison()
    test_score_accuracy()
    test_theoretical_vs_actual()

    print("=" * 65)
    print("DONE")
    print("=" * 65)
