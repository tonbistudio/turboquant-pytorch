# TurboQuant

A from-scratch PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches.

We implemented the algorithm from the paper, validated it on synthetic vectors, then tested it against a real model's KV cache (Qwen2.5-3B-Instruct on an RTX 3060) to verify the compression and accuracy claims.

## Background

When an LLM generates text, it stores a **key** and **value** vector for every token it has seen, in every layer. This is the KV cache — the model's working memory. At 8K tokens on a 36-layer model like Qwen2.5-3B, this cache is **289 MB** in FP16. On a 12GB GPU, the KV cache — not the model weights — becomes the bottleneck for long context.

TurboQuant compresses this cache by quantizing the vectors to 2-4 bits per coordinate, achieving 3-7x compression with minimal impact on attention accuracy.

## How TurboQuant Works

The algorithm has two stages:

### Stage 1: Random Rotation + Lloyd-Max Quantization

Each vector is multiplied by a random orthogonal matrix (generated via QR decomposition of a Gaussian matrix). This rotation is the key trick — it makes every coordinate of the resulting vector follow a predictable bell-curve distribution (Beta distribution, well-approximated by Gaussian N(0, 1/d) for typical head dimensions).

Because the distribution is known and coordinates become nearly independent, we can design an **optimal scalar quantizer** (Lloyd-Max) for each coordinate independently. The Lloyd-Max algorithm finds the best set of "buckets" to round values into, minimizing mean squared error. We precompute these codebooks once per bit-width.

To quantize: rotate the vector, round each coordinate to its nearest codebook centroid, store the indices.
To dequantize: look up centroids, reverse the rotation.

### Stage 2: QJL Residual Correction (1 bit)

The MSE-optimal quantizer from Stage 1 introduces a small bias in dot products (inner products). Since attention scores are just dot products between queries and keys, this bias accumulates.

The Quantized Johnson-Lindenstrauss (QJL) transform fixes this. It takes the quantization residual (the error left over from Stage 1), projects it through a random Gaussian matrix, and stores just the **sign** (+1 or -1) of each projection — exactly 1 bit per dimension. This single bit is enough to make the inner product estimate **mathematically unbiased**.

The combined estimator for `<query, key>` is:

```
<q, k> ≈ <q, k_mse> + ||residual|| * sqrt(pi/2) / m * <S @ q, sign(S @ residual)>
```

Where `S` is the random projection matrix, `k_mse` is the Stage 1 reconstruction, and `residual = k - k_mse`.

### Why This Works Despite High Per-Vector Error

An important subtlety: the per-vector reconstruction error is significant (23-44% relative error depending on bit-width). If you decompress the vectors and feed them to standard attention, the model produces garbage.

But TurboQuant doesn't need accurate vector reconstruction. It needs accurate **inner products** (attention scores). The QJL correction ensures these are unbiased with variance O(1/d), where d is the head dimension (typically 128). The attention distribution over tokens is preserved even when individual vectors look quite different from the originals.

## Our Findings

### Synthetic Vector Tests (`test_turboquant.py`)

We first validated the core algorithm on random unit vectors against the paper's theoretical bounds.

**MSE Distortion** (d=128, 1000 random unit vectors):

| Bits | Measured MSE | Paper's Upper Bound | Ratio |
|------|-------------|-------------------|-------|
| 1-bit | 0.362 | 0.680 | 0.53x |
| 2-bit | 0.116 | 0.170 | 0.68x |
| 3-bit | 0.034 | 0.043 | 0.81x |
| 4-bit | 0.009 | 0.011 | 0.87x |

All well within the theoretical bounds. The ratio approaching 1.0 at higher bit-widths is expected — the bound is tighter when quantization is finer.

**Inner Product Accuracy** (d=128, 2000 random vector pairs):

| Bits | Bias | Correlation with True IP |
|------|------|------------------------|
| 2-bit | +0.001 | 0.80 |
| 3-bit | +0.000 | 0.93 |
| 4-bit | +0.000 | 0.98 |

Near-zero bias at all bit-widths confirms QJL correction works. Correlation of 0.98 at 4-bit means the estimated inner products track the true values very closely.

**Needle-in-Haystack** (synthetic vectors, finding the most similar vector):

9/9 exact retrieval across all bit-widths (2, 3, 4) and sequence lengths (512, 2048, 8192). TurboQuant correctly identifies the closest vector every time.

### Real Model Validation (`validate.py`)

We loaded Qwen2.5-3B-Instruct in 4-bit quantization on an RTX 3060 (12GB), ran a forward pass on a long document containing a hidden fact, captured the real KV cache, compressed it with TurboQuant, and compared the attention scores.

**Compression Ratios** (consistent across all context lengths):

| Config | KV Cache Size (8K ctx) | Compression |
|--------|----------------------|-------------|
| FP16 (baseline) | 289 MB | 1.0x |
| TurboQuant 4-bit | 76 MB | **3.8x** |
| TurboQuant 3-bit | 58 MB | **5.0x** |
| TurboQuant 2-bit | 40 MB | **7.3x** |

At 3-bit, 289 MB becomes 58 MB. On a 12GB GPU, that's the difference between fitting ~8K context and fitting ~40K.

**Attention Score Accuracy** (averaged across all 36 layers, 2 KV heads per layer = 72 total checks):

| Config | Context | Cosine Sim | Top-1 Match | Top-5 Match |
|--------|---------|-----------|-------------|-------------|
| TQ-4bit | 2K | 0.9989 | 85% | 96% |
| TQ-4bit | 4K | 0.9986 | 92% | 94% |
| TQ-4bit | 8K | 0.9983 | 86% | 96% |
| TQ-3bit | 2K | 0.9961 | 85% | 94% |
| TQ-3bit | 4K | 0.9955 | 75% | 88% |
| TQ-3bit | 8K | 0.9945 | 86% | 94% |
| TQ-2bit | 2K | 0.9897 | 63% | 83% |
| TQ-2bit | 4K | 0.9878 | 65% | 85% |
| TQ-2bit | 8K | 0.9851 | 71% | 89% |

**What the metrics mean:**

- **Cosine Similarity**: How similar the full vector of attention scores is between compressed and original. 0.995 means the compressed attention pattern is 99.5% similar. This is the most important metric — it captures the overall shape of "which tokens does the model attend to."
- **Top-1 Match**: Does the single most-attended token stay the same after compression? 87% at 4-bit means 63 out of 72 layer-head combinations point to the exact same token.
- **Top-5 Match**: Is the real most-attended token still in the top 5 after compression? 96% at 4-bit means only 3 out of 72 heads shifted their top pick out of the top 5.

**Key observations:**
- Cosine similarity is remarkably stable across context lengths (0.998 at 4-bit regardless of 2K or 8K)
- 3-bit is the practical sweet spot: 5x compression with 99.5% attention fidelity
- 2-bit works but pushes it — 66% top-1 match means the model would sometimes attend to different tokens
- The paper's "zero accuracy loss" claim at 3.5 bits is plausible given these numbers

## Scripts

### `test_turboquant.py` — Synthetic Validation

Validates the core algorithm with no model or GPU required (GPU enables an optional speed benchmark).

**What it does:**
1. Builds Lloyd-Max codebooks for various dimensions (64, 128, 256) and bit-widths (1-4)
2. Generates random unit vectors and measures quantize-dequantize MSE against theoretical bounds
3. Tests inner product estimation with QJL correction — measures bias and correlation
4. Demonstrates that MSE-only quantization is biased (motivating QJL)
5. Tests the KV cache wrapper and reports compression ratios
6. Runs needle-in-haystack retrieval on synthetic vectors
7. Benchmarks quantization speed on GPU (if available)

**Run it:**
```bash
python -m turboquant.test_turboquant
```

### `validate.py` — Real Model Validation

Tests TurboQuant on actual KV cache data from a real language model.

**What it does:**
1. Loads Qwen2.5-3B-Instruct in 4-bit quantization (~2GB VRAM)
2. Builds a long document with filler text and a hidden "needle" fact
3. Runs a single forward pass to capture the full KV cache (all 36 layers)
4. For each bit-width (2, 3, 4), compresses every layer's keys and values
5. Computes attention scores using both the original keys and the TurboQuant asymmetric estimator
6. Reports compression ratios, cosine similarity, and top-N retrieval accuracy

**Run it:**
```bash
python -m turboquant.validate
```

First run downloads the model (~2GB). Requires a CUDA GPU with at least 6GB VRAM.

## Project Structure

```
turboquant/
  __init__.py           # Package exports
  lloyd_max.py          # Lloyd-Max optimal scalar quantizer solver
  turboquant.py         # Core TurboQuant: TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
  compressors.py        # Asymmetric inner product compressors for real model validation
  test_turboquant.py    # Synthetic algorithm tests
  validate.py           # Real model attention comparison
  requirements.txt      # Python dependencies
```

### Module Details

**`lloyd_max.py`** — Solves the Lloyd-Max optimal quantizer for the coordinate distribution that arises after random rotation of unit vectors. Uses numerical integration (scipy) to find centroids that minimize MSE. Codebooks are precomputed once and reused.

**`turboquant.py`** — The core algorithm implementation. `TurboQuantMSE` does Stage 1 (rotation + quantization). `TurboQuantProd` adds Stage 2 (QJL residual correction) and provides the unbiased inner product estimator. `TurboQuantKVCache` wraps both into a cache interface.

**`compressors.py`** — Production-oriented compressors that handle real model tensors (normalization, dtype conversion, asymmetric score computation). `TurboQuantCompressorV2` compresses key vectors and supports `asymmetric_attention_scores()` for computing attention directly from compressed data. `TurboQuantCompressorMSE` compresses value vectors with MSE-only quantization.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (for GPU tests)
- scipy (for codebook computation)
- transformers, accelerate, bitsandbytes (for real model validation only)

```bash
pip install -r requirements.txt
```

For CUDA PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482) — The 1-bit residual correction technique
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) — Related approach using polar coordinates
- [QJL Reference Implementation](https://github.com/amirzandieh/QJL) — Original CUDA implementation by the QJL authors
- [PolarQuant Reference Implementation](https://github.com/ericshwu/PolarQuant)

## Contributions & Improvements

This fork adds three fixes over the original repo.
All changes preserve full backward compatibility — existing tests pass unchanged.

### Fix 1: Package structure (`turboquant/` subdirectory)

The original repo placed all modules at root with relative imports (`from .lloyd_max import ...`),
which prevented running `test_turboquant.py` or `validate.py` directly. Modules have been moved
into a proper `turboquant/` package directory, matching the import paths already used in the test
scripts. This is the same structural change proposed in PR #3.

### Fix 2: `PackedKVCompressor` — actual memory compression

**The original `TurboQuantCompressorV2` did not achieve the compression ratios it reported.**

`compress()` stored `k_mse` as a full `float16` tensor of shape `(B, H, S, D)` alongside the
QJL signs (stored as `int8` — 8 bits per 1-bit sign). For `d=128` this cost **386 bytes/vector
vs 256 bytes fp16 — 38% larger than uncompressed**. The `validate.py` memory accounting was
correct (reporting theoretical indices size), but the actual PyTorch tensors contradicted it.

`PackedKVCompressor` fixes this by:
- Storing **MSE indices** (not reconstructed vectors), bit-packed `floor(8/mse_bits)` per byte
- Packing **QJL sign bits** 8 per byte via bitwise operations
- Recomputing `k_mse` on the fly during `asymmetric_attention_scores()`

Result (`d=128`, `H=2`, `S=4096`):

| Config | fp16 | V2 (original) | PackedKVCompressor |
|--------|------|---------------|--------------------:|
| 2-bit  | 2048 KB | 3088 KB (**-38%**) | 288 KB (**7.1x**) |
| 3-bit  | 2048 KB | 3088 KB (**-38%**) | 416 KB (**4.9x**) |
| 4-bit  | 2048 KB | 3088 KB (**-38%**) | 672 KB (**3.1x**) |

Attention score accuracy (cosine similarity vs fp16 ground truth) is identical to V2.

### Fix 3: Eliminated duplicate codebook solver

`TurboQuantCompressorV2` and `TurboQuantCompressorMSE` each had a private `_solve_codebook()`
method copy-pasted from `lloyd_max.py`. Both now import `LloydMaxCodebook` directly,
removing ~60 lines of duplicated code and ensuring codebook consistency.

### New files

| File | Description |
|------|-------------|
| `test_packed.py` | Memory and accuracy comparison: V2 vs PackedKVCompressor |

## License

MIT
