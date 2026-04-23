"""
TurboQuant V3 validation: compare V2 (QJL) vs V3 (MSE-only, asymmetric, residual window)
on real model KV cache data.
"""

import torch
import torch.nn.functional as F
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquant.compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from turboquant.compressors_v3 import TurboQuantV3

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


def build_prompt(tokenizer, target_tokens=2048, needle_pos=0.5):
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    return f"<|im_start|>user\n{haystack}\nQuestion: {QUESTION}<|im_end|>\n<|im_start|>assistant\n"


def eval_v2(keys, values, queries, bits, layer_idx):
    """V2: MSE + QJL compressor for keys, MSE-only for values."""
    B, H, S, D = keys.shape
    key_comp = TurboQuantCompressorV2(D, bits, seed=layer_idx * 1000, device="cuda")
    val_comp = TurboQuantCompressorMSE(D, bits, seed=layer_idx * 1000 + 500, device="cuda")

    compressed_k = key_comp.compress(keys)
    compressed_v = val_comp.compress(values)

    # Attention scores via asymmetric estimator
    tq_scores = key_comp.asymmetric_attention_scores(queries, compressed_k).squeeze(-2)

    # Memory (theoretical, matching V2's accounting)
    mse_bits = max(bits - 1, 1)
    n = B * H * S
    k_bits = n * D * mse_bits + n * D * 1 + n * 16 + n * 16  # indices + qjl + rnorm + vnorm
    v_bits = n * D * bits + n * 16
    compressed_bytes = (k_bits + v_bits) / 8
    fp16_bytes = (keys.numel() + values.numel()) * 2

    return tq_scores, compressed_bytes, fp16_bytes


def eval_v3(keys, values, queries, key_bits, value_bits, layer_idx, n_layers, residual_window, protected_layers):
    """V3: MSE-only, asymmetric K/V, residual window, layer-adaptive."""
    B, H, S, D = keys.shape

    compressor = TurboQuantV3(
        head_dim=D,
        key_bits=key_bits,
        value_bits=value_bits,
        residual_window=residual_window,
        layer_idx=layer_idx,
        n_layers=n_layers,
        protected_layers=protected_layers,
        seed=42,
        device="cuda",
    )

    compressed_k, compressed_v = compressor.compress_kv(keys, values)
    keys_recon, values_recon = compressor.decompress_kv(compressed_k, compressed_v)

    # Attention scores via standard matmul on decompressed keys
    tq_scores = torch.matmul(queries.float(), keys_recon.float().transpose(-2, -1)).squeeze(-2)

    mem = compressor.memory_bytes(B, H, S)
    return tq_scores, mem["compressed_bytes"], mem["fp16_bytes"], mem


def compute_metrics(real_scores, tq_scores, H):
    """Compute cosine sim, top-1, top-5 across all heads."""
    cosine_sims = []
    top1_matches = 0
    top5_matches = 0
    n_checks = 0

    for h in range(H):
        rs = real_scores[0, h]
        ts = tq_scores[0, h]

        cos = F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item()
        cosine_sims.append(cos)

        real_top1 = rs.argmax().item()
        tq_top1 = ts.argmax().item()
        if real_top1 == tq_top1:
            top1_matches += 1

        tq_top5 = ts.topk(5).indices.tolist()
        if real_top1 in tq_top5:
            top5_matches += 1

        n_checks += 1

    return {
        "cosine_sim": sum(cosine_sims) / len(cosine_sims),
        "top1_pct": 100 * top1_matches / n_checks,
        "top5_pct": 100 * top5_matches / n_checks,
        "top1_matches": top1_matches,
        "top5_matches": top5_matches,
        "n_checks": n_checks,
    }


def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"),
        device_map="auto", dtype=torch.float16,
    )
    model.eval()
    print(f"Loaded. GPU: {torch.cuda.memory_allocated() // 1024 // 1024} MB\n")

    # V3 configs to test
    v3_configs = [
        {"key_bits": 4, "value_bits": 2, "residual_window": 0,   "protected_layers": 0, "label": "V3 K4/V2"},
        {"key_bits": 4, "value_bits": 2, "residual_window": 128, "protected_layers": 0, "label": "V3 K4/V2+rw128"},
        {"key_bits": 3, "value_bits": 2, "residual_window": 128, "protected_layers": 0, "label": "V3 K3/V2+rw128"},
        {"key_bits": 4, "value_bits": 2, "residual_window": 128, "protected_layers": 4, "label": "V3 K4/V2+rw+prot4"},
    ]

    for target_tokens in [2048, 4096, 8192]:
        prompt = build_prompt(tokenizer, target_tokens)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 256).to("cuda")
        seq_len = inputs["input_ids"].shape[1]

        print(f"{'=' * 80}")
        print(f"Context: {seq_len} tokens")
        print(f"{'=' * 80}")

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False)
        cache = outputs.past_key_values

        n_layers = len(cache.layers)
        head_dim = cache.layers[0].keys.shape[-1]
        num_kv_heads = cache.layers[0].keys.shape[1]

        # --- V2 baselines ---
        for bits in [3, 4]:
            total_compressed = 0
            total_fp16 = 0
            all_metrics = {"cosine_sim": [], "top1": 0, "top5": 0, "n": 0}

            for layer_idx in range(n_layers):
                keys = cache.layers[layer_idx].keys
                values = cache.layers[layer_idx].values
                query = keys[:, :, -1:, :]

                real_scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)).squeeze(-2)
                tq_scores, c_bytes, f_bytes = eval_v2(keys, values, query, bits, layer_idx)

                total_compressed += c_bytes
                total_fp16 += f_bytes

                m = compute_metrics(real_scores, tq_scores, num_kv_heads)
                all_metrics["cosine_sim"].append(m["cosine_sim"])
                all_metrics["top1"] += m["top1_matches"]
                all_metrics["top5"] += m["top5_matches"]
                all_metrics["n"] += m["n_checks"]

            ratio = total_fp16 / total_compressed
            avg_cos = sum(all_metrics["cosine_sim"]) / len(all_metrics["cosine_sim"])
            t1 = 100 * all_metrics["top1"] / all_metrics["n"]
            t5 = 100 * all_metrics["top5"] / all_metrics["n"]
            print(f"  V2 {bits}bit (MSE+QJL):  {ratio:>5.1f}x | cos={avg_cos:.6f} | top1={t1:>5.1f}% | top5={t5:>5.1f}%")

        # --- V3 configs ---
        for cfg in v3_configs:
            total_compressed = 0
            total_fp16 = 0
            all_metrics = {"cosine_sim": [], "top1": 0, "top5": 0, "n": 0}

            for layer_idx in range(n_layers):
                keys = cache.layers[layer_idx].keys
                values = cache.layers[layer_idx].values
                query = keys[:, :, -1:, :]

                real_scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)).squeeze(-2)
                tq_scores, c_bytes, f_bytes, mem = eval_v3(
                    keys, values, query,
                    key_bits=cfg["key_bits"],
                    value_bits=cfg["value_bits"],
                    layer_idx=layer_idx,
                    n_layers=n_layers,
                    residual_window=cfg["residual_window"],
                    protected_layers=cfg["protected_layers"],
                )

                total_compressed += c_bytes
                total_fp16 += f_bytes

                m = compute_metrics(real_scores, tq_scores, num_kv_heads)
                all_metrics["cosine_sim"].append(m["cosine_sim"])
                all_metrics["top1"] += m["top1_matches"]
                all_metrics["top5"] += m["top5_matches"]
                all_metrics["n"] += m["n_checks"]

            ratio = total_fp16 / total_compressed if total_compressed > 0 else 0
            avg_cos = sum(all_metrics["cosine_sim"]) / len(all_metrics["cosine_sim"])
            t1 = 100 * all_metrics["top1"] / all_metrics["n"]
            t5 = 100 * all_metrics["top5"] / all_metrics["n"]
            print(f"  {cfg['label']:<22s} {ratio:>5.1f}x | cos={avg_cos:.6f} | top1={t1:>5.1f}% | top5={t5:>5.1f}%")

        print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
