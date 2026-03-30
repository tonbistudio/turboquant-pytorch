"""
RotorQuant vs TurboQuant: Real model validation on Qwen2.5-3B-Instruct.

Compresses actual KV cache from a forward pass and compares attention
score fidelity between the two quantizers.
"""

import torch
import torch.nn.functional as F
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquant.turboquant import TurboQuantMSE, TurboQuantProd, generate_rotation_matrix
from turboquant.rotorquant import RotorQuantMSE, RotorQuantProd
from turboquant.lloyd_max import LloydMaxCodebook

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team.\n\n"""


def build_prompt(tokenizer, target_tokens=2048):
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    return "".join([FILLER] * n_reps)


def compress_and_score(keys, query, bits, layer_idx, method="tq"):
    """Compress keys and compute attention scores."""
    B, H, S, D = keys.shape
    results = {"cosine_sims": [], "top1_matches": 0, "top5_matches": 0, "n_checks": 0}

    for h in range(H):
        k = keys[0, h]  # (S, D)
        q = query[0, h]  # (1, D) or (D,)
        if q.dim() == 1:
            q = q.unsqueeze(0)

        # Normalize for quantization
        k_norm = k.float()
        q_norm = q.float()

        # Real attention scores
        real_scores = (q_norm @ k_norm.T).squeeze(0)  # (S,)

        # Quantized scores
        if method == "tq":
            quantizer = TurboQuantProd(D, bits, seed=layer_idx * 1000 + h, device=k.device)
        else:
            quantizer = RotorQuantProd(D, bits, seed=layer_idx * 1000 + h, device=str(k.device))

        compressed = quantizer.quantize(k_norm)
        est_scores = quantizer.inner_product(q_norm.expand(S, -1), compressed)  # (S,)

        # Cosine similarity of score vectors
        cos = F.cosine_similarity(real_scores.unsqueeze(0), est_scores.unsqueeze(0)).item()
        results["cosine_sims"].append(cos)

        # Top-1 match
        if real_scores.argmax().item() == est_scores.argmax().item():
            results["top1_matches"] += 1

        # Top-5
        real_top1 = real_scores.argmax().item()
        tq_top5 = est_scores.topk(5).indices.tolist()
        if real_top1 in tq_top5:
            results["top5_matches"] += 1

        results["n_checks"] += 1

    return results


def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        ),
        device_map="auto", dtype=torch.float16,
    )
    model.eval()
    print(f"Loaded. GPU mem: {torch.cuda.memory_allocated() // 1024 // 1024} MB\n")

    for target_tokens in [2048, 4096]:
        prompt = build_prompt(tokenizer, target_tokens)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=target_tokens + 256).to("cuda")
        seq_len = inputs["input_ids"].shape[1]

        print(f"{'=' * 70}")
        print(f"Context: {seq_len} tokens")
        print(f"{'=' * 70}")

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False)
        cache = outputs.past_key_values
        n_layers = len(cache.layers)

        for bits in [3, 4]:
            for method, label in [("tq", "TurboQuant"), ("rq", "RotorQuant")]:
                all_cos = []
                total_top1 = 0
                total_top5 = 0
                total_checks = 0

                t0 = time.perf_counter()

                # Only process first 8 layers for speed
                for layer_idx in range(min(n_layers, 8)):
                    keys = cache.layers[layer_idx].keys  # (1, H, S, D)
                    query = keys[:, :, -1:, :]  # last token as query

                    r = compress_and_score(keys, query, bits, layer_idx, method)
                    all_cos.extend(r["cosine_sims"])
                    total_top1 += r["top1_matches"]
                    total_top5 += r["top5_matches"]
                    total_checks += r["n_checks"]

                elapsed = time.perf_counter() - t0

                avg_cos = sum(all_cos) / len(all_cos)
                top1_pct = 100 * total_top1 / total_checks
                top5_pct = 100 * total_top5 / total_checks

                print(f"\n  {label} {bits}-bit (8/{n_layers} layers, {elapsed:.1f}s):")
                print(f"    Score cosine sim:  {avg_cos:.6f}")
                print(f"    Top-1 match:       {top1_pct:.1f}% ({total_top1}/{total_checks})")
                print(f"    Top-5 match:       {top5_pct:.1f}% ({total_top5}/{total_checks})")

        print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
