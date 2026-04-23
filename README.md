# TurboQuant

A from-scratch PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's vector quantization algorithm for compressing LLM key-value caches. Tested on Windows with NVIDIA GPUs.

We implemented the paper's algorithm, found that its key innovation (QJL) actually hurts in practice, and built an improved version (V3) informed by findings from 8+ independent community implementations.

> **Correction (2026-03-30):** An earlier version of this README claimed "18/18 perfect generation at 5x compression." This was based on a [bugged test](https://github.com/tonbistudio/turboquant-pytorch/issues/14) where `residual_window=0` caused no compression to happen. The corrected results are below. Credit to [@barbel-bb](https://github.com/barbel-bb) for finding the bug.

## Results

### V3: Generation Test (the real test — does the model produce correct text?)

We hid a fact ("The secret project code name is AURORA-7749") in a long document and asked the model to find it. Results with **actual compression verified** (compressed token counts logged):

| Config | 2K ctx | 4K ctx | Compression (2K) |
|--------|--------|--------|-----------------|
| FP16 (baseline) | EXACT | EXACT | 1.0x |
| **K6/V4 rw=128** | **EXACT** | **EXACT** | **~2x** |
| **K8/V4 rw=128** | **EXACT** | **EXACT** | **~1.6x** |
| K4/V4 rw=128 | PARTIAL ("AURORA7749") | MISS | ~3x |
| K4/V4 rw=0 | MISS | MISS | ~3.4x |
| K4/V2 rw=0 | MISS | MISS | ~5x |

"EXACT" = output contains "AURORA-7749". "PARTIAL" = contains both "AURORA" and "7749" but not the exact string. "rw" = residual window (recent tokens kept in fp16).

**What works:** K6/V4 with a 128-token fp16 window gives ~2x real compression with perfect output at both context lengths. At 4-bit keys, the model finds the needle at short context but garbles it slightly (drops the hyphen). At 3-bit keys, generation is broken.

**What doesn't work:** 3-4 bit compression without a residual window produces garbage, same as V2. High attention score similarity (99.5%+) does not guarantee working generation.

### V3: Attention Score Accuracy (8K context)

These results are valid — they test compression directly on captured KV tensors, not through V3Cache:

| Config | Compression | Cosine Similarity | Top-1 Match | Top-5 Match |
|--------|-----------|------------------|-------------|-------------|
| V3 K4/V2 | 5.1x | **0.9996** | **94%** | **97%** |
| V3 K4/V2 + protected layers | 3.6x | **0.9997** | **99%** | **100%** |
| V2 3-bit (MSE+QJL) | 5.0x | 0.9945 | 86% | 94% |
| V2 4-bit (MSE+QJL) | 3.8x | 0.9983 | 86% | 96% |

V3 gets better attention score accuracy than V2 by removing QJL. However, high attention scores alone do not guarantee working text generation (see above).

## What Is K4/V2?

The KV cache stores two types of vectors: **Keys** (K) and **Values** (V).

- **Keys** decide which words the model pays attention to — this needs precision
- **Values** are the content that gets averaged together — errors cancel out naturally

**K4/V2** means keys get 4 bits, values get 2 bits. The average is 3 bits — same as uniform 3-bit — but allocated where it matters. This gives dramatically better results than uniform allocation at the same bit budget.

## How It Works

### The Core: Random Rotation + Lloyd-Max Quantization

Each vector is multiplied by a random orthogonal matrix, which makes every coordinate follow a predictable bell-curve distribution. We then apply an **optimal scalar quantizer** (Lloyd-Max) to each coordinate independently, rounding to the nearest precomputed centroid.

To quantize: normalize, rotate, round each coordinate, store indices + norm.
To dequantize: look up centroids, reverse the rotation, restore the norm.

### What About QJL? (The Paper's Stage 2)

The paper adds a second stage: QJL residual correction, which stores 1-bit sign information to make inner product estimates mathematically unbiased. We implemented this as V2.

**It doesn't work for KV cache.** Six independent teams confirmed this:

- QJL is unbiased for raw inner products, but attention runs scores through **softmax**
- Softmax exponentially amplifies variance — QJL's random noise gets magnified
- MSE-only has biased inner products but lower variance — and lower variance wins after softmax
- scos-lab measured +300% error with QJL vs +7.6% without on GPT-2
- Our V2 with QJL: 0/27 generation tests passed. V3 without QJL: 18/18 passed.

QJL does work for **vector search** (no softmax), which is the paper's other use case. It may also work with non-softmax attention (sigmoid, linear, gated).

### V3 Improvements (Community-Informed)

1. **MSE-only** — Remove QJL, all bits go to reconstruction quality (`compressors_v3.py → MSECompressor`)
2. **Asymmetric K/V** — Keys get more bits than values (`TurboQuantV3(key_bits=4, value_bits=2)`)
3. **Bit-packed storage** — Real compression ratios, not theoretical. V2 stored tensors that were 38% larger than uncompressed. (`MSECompressor.compress()` uses bit-shifting)
4. **Layer-adaptive** — Protect sensitive first/last layers with more bits (`TurboQuantV3(protected_layers=4)`)

## Quick Start

### Requirements

- Python 3.10+
- CUDA-capable NVIDIA GPU (tested on RTX 3060, 12GB)
- Windows 11 (also works on Linux)

```bash
pip install -r requirements.txt
```

For editable local development:
```bash
pip install -e .
```

For CUDA PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Run Generation Test (V3 — recommended)

Tests whether the model actually produces correct text with compressed KV cache:

```bash
python -m turboquant.generation_test
```

First run downloads Qwen2.5-3B-Instruct (~2GB). Tests multiple configs across context lengths.

### Run Attention Validation (V3 vs V2)

Compares V2 and V3 attention score accuracy side by side:

```bash
python -m turboquant.validate_v3
```

### Run Synthetic Tests (no model needed)

Validates the core algorithm against theoretical bounds from the paper:

```bash
python -m turboquant.test_turboquant
```

### Run Original V2 Validation

The original attention-score comparison (without generation):

```bash
python -m turboquant.validate
```

## Project Structure

```
turboquant/
  __init__.py           # Package exports
  lloyd_max.py          # Lloyd-Max optimal scalar quantizer solver
  turboquant.py         # Core TurboQuant: TurboQuantMSE, TurboQuantProd (V2 with QJL)
  compressors.py        # V2 compressors (MSE+QJL for keys, MSE-only for values)
  compressors_v3.py     # V3 compressors (MSE-only, asymmetric K/V, bit-packed, layer-adaptive)
  test_turboquant.py    # Synthetic algorithm tests
  validate.py           # V2 real model attention comparison
  validate_v3.py        # V3 vs V2 comparison
  generation_test.py    # V3 actual text generation test
  requirements.txt
```

### Key Classes

**`MSECompressor`** (`compressors_v3.py`) — Single-stage compressor with bit-packed storage. Used for both keys and values. The core building block of V3.

**`TurboQuantV3`** (`compressors_v3.py`) — Orchestrator that creates separate key/value compressors with different bit-widths and handles layer-adaptive precision.

**`TurboQuantMSE`** (`turboquant.py`) — Original Stage 1 quantizer. Still used by synthetic tests.

**`TurboQuantProd`** (`turboquant.py`) — Original two-stage quantizer with QJL. Kept for reference and synthetic tests where QJL works correctly.

## Synthetic Test Results

The core rotation + Lloyd-Max algorithm is validated against the paper's theoretical bounds:

**MSE Distortion** (d=128, 1000 random unit vectors):

| Bits | Measured MSE | Paper's Upper Bound | Ratio |
|------|-------------|-------------------|-------|
| 1-bit | 0.362 | 0.680 | 0.53x |
| 2-bit | 0.116 | 0.170 | 0.68x |
| 3-bit | 0.034 | 0.043 | 0.81x |
| 4-bit | 0.009 | 0.011 | 0.87x |

**Needle-in-Haystack** (synthetic vectors): 9/9 exact retrieval across all bit-widths and sequence lengths.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482)
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617)
- [QJL Reference Implementation](https://github.com/amirzandieh/QJL)
- [PolarQuant Reference Implementation](https://github.com/ericshwu/PolarQuant)

## Community Work

Several community members have extended this implementation with valuable findings:

- **[scos-lab/turboquant](https://github.com/scos-lab/turboquant)** — 8-model benchmark showing K/V norm ratio predicts compression quality. Found MSE-only outperforms MSE+QJL for attention (softmax amplifies QJL variance). Outlier-aware mixed precision achieves 3.6-bit avg with +2.1% PPL on Qwen2.5-1.5B.
- **[0xSero/turboquant](https://github.com/0xSero/turboquant)** — Triton kernels + vLLM integration. Production deployment with asymmetric K/V bits. Tested on RTX 5090 and 8x RTX 3090.
- **[back2matching/turboquant](https://github.com/back2matching/turboquant)** — pip-installable, drop-in HuggingFace generation. Residual windowing (recent tokens in fp16).
- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Layer-adaptive compression, attention-gated V decoding (+22.8% decode speed). Apple Silicon optimized.
- **[RecursiveIntell/turbo-quant](https://github.com/RecursiveIntell/turbo-quant)** — Rust implementation of TurboQuant + PolarQuant + QJL. Zero-copy, streaming compatible.
- **[SCJedi/entropy-adaptive-kv-cache](https://github.com/SCJedi/entropy-adaptive-kv-cache)** — Combines TurboQuant with entropy-adaptive token eviction for 12x compression with zero quality loss on Qwen3.5-4B.

### Key community findings

- **MSE-only beats MSE+QJL for attention** ([#10](https://github.com/tonbistudio/turboquant-pytorch/issues/10), [#8](https://github.com/tonbistudio/turboquant-pytorch/issues/8)) — Confirmed by 6+ independent teams across Python, C, and Rust implementations.
- **Q4_0 beats TurboQuant at similar compression ratios** ([#6](https://github.com/tonbistudio/turboquant-pytorch/issues/6)) — TurboQuant's advantage is at higher compression (3-bit/5x and below) where block methods can't go.
- **K/V norm asymmetry matters** ([#8](https://github.com/tonbistudio/turboquant-pytorch/issues/8)) — Qwen models have key norms of 172-778 vs value norms of 2-4. Asymmetric bit allocation (more bits for keys) is essential.

## License

MIT
