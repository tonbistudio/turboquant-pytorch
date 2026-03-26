"""
TurboQuant validation: compare attention scores on real model data.
Single forward pass, no monkey-patching. Just math.
"""

import torch
import torch.nn.functional as F
import time
import os
import sys

# Allow running as `python validate.py` from within the package directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquant.compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE

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

    for target_tokens in [2048, 4096, 8192]:
        prompt = build_prompt(tokenizer, target_tokens)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 256).to("cuda")
        seq_len = inputs["input_ids"].shape[1]

        # Find needle token positions - search for a distinctive substring
        needle_phrase = "AURORA-7749"
        needle_tokens = tokenizer.encode(needle_phrase, add_special_tokens=False)
        input_ids_list = inputs["input_ids"][0].tolist()
        needle_start = None
        for i in range(len(input_ids_list) - len(needle_tokens) + 1):
            if input_ids_list[i:i + len(needle_tokens)] == needle_tokens:
                needle_start = i
                break
        # Fallback: search for partial match
        if needle_start is None:
            for width in range(len(needle_tokens), 0, -1):
                sub = needle_tokens[:width]
                for i in range(len(input_ids_list) - width + 1):
                    if input_ids_list[i:i + width] == sub:
                        needle_start = i
                        break
                if needle_start is not None:
                    break

        print(f"{'=' * 70}")
        print(f"Context: {seq_len} tokens | Needle at token {needle_start}")
        print(f"{'=' * 70}")

        # Forward pass - capture KV cache
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False)
        cache = outputs.past_key_values

        n_layers = len(cache.layers)
        head_dim = cache.layers[0].keys.shape[-1]
        num_kv_heads = cache.layers[0].keys.shape[1]

        # Query = last token's Q projection (simulates next-token generation)
        # We'll recompute Q from the last hidden state
        last_hidden = outputs.logits  # we don't actually need this
        # Instead, just use the cached keys as queries too (self-attention validation)
        # Pick the LAST token's perspective: its query attending to all keys

        for bits in [2, 3, 4]:
            total_compressed_bytes = 0
            total_uncompressed_bytes = 0
            top1_matches = 0
            top5_matches = 0
            needle_rank_sum = 0
            cosine_sims = []
            n_checks = 0

            for layer_idx in range(n_layers):
                keys = cache.layers[layer_idx].keys      # (1, num_kv_heads, seq, head_dim)
                values = cache.layers[layer_idx].values

                B, H, S, D = keys.shape

                # Compress keys
                key_comp = TurboQuantCompressorV2(D, bits, seed=layer_idx * 1000, device="cuda")
                val_comp = TurboQuantCompressorMSE(D, bits, seed=layer_idx * 1000 + 500, device="cuda")

                compressed_k = key_comp.compress(keys)
                compressed_v = val_comp.compress(values)

                # Memory accounting - what TurboQuant actually stores:
                # Keys: indices (uint8) + qjl_signs (1 bit packed) + residual_norm (fp16) + vec_norms (fp16)
                n_key_vecs = B * H * S
                mse_bits = max(bits - 1, 1)
                k_bits = n_key_vecs * D * mse_bits  # MSE indices
                k_bits += n_key_vecs * D * 1         # QJL sign bits
                k_bits += n_key_vecs * 16             # residual norms (fp16)
                k_bits += n_key_vecs * 16             # vector norms (fp16)

                # Values: indices (mse_bits per coord) + vec_norms (fp16)
                v_bits = n_key_vecs * D * bits        # MSE indices (full bits, no QJL)
                v_bits += n_key_vecs * 16              # vector norms

                total_compressed_bytes += (k_bits + v_bits) / 8
                total_uncompressed_bytes += (keys.numel() + values.numel()) * 2  # fp16

                # Compare attention scores for last token query attending to all keys
                # Use the last key as a proxy query (same head_dim)
                query = keys[:, :, -1:, :]  # (1, H, 1, D) - last token

                # Real scores
                real_scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)).squeeze(-2)  # (1, H, S)

                # TurboQuant scores
                tq_scores = key_comp.asymmetric_attention_scores(query, compressed_k).squeeze(-2)  # (1, H, S)

                # Per-head comparison
                for h in range(H):
                    rs = real_scores[0, h]  # (S,)
                    ts = tq_scores[0, h]

                    # Cosine similarity of score vectors
                    cos = F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item()
                    cosine_sims.append(cos)

                    # Top-1 match
                    real_top1 = rs.argmax().item()
                    tq_top1 = ts.argmax().item()
                    if real_top1 == tq_top1:
                        top1_matches += 1

                    # Top-5 match (does real top-1 appear in TQ top-5?)
                    tq_top5 = ts.topk(5).indices.tolist()
                    if real_top1 in tq_top5:
                        top5_matches += 1

                    # Where does the needle rank?
                    if needle_start is not None:
                        needle_rank = (ts.argsort(descending=True) == needle_start).nonzero()
                        if len(needle_rank) > 0:
                            needle_rank_sum += needle_rank[0].item()

                    n_checks += 1

            # Summary for this bit-width
            ratio = total_uncompressed_bytes / total_compressed_bytes
            avg_cos = sum(cosine_sims) / len(cosine_sims)
            top1_pct = 100 * top1_matches / n_checks
            top5_pct = 100 * top5_matches / n_checks
            avg_needle_rank = needle_rank_sum / n_checks if needle_start else -1

            print(f"\n  TQ-{bits}bit:")
            print(f"    Compression:       {ratio:.1f}x  ({total_compressed_bytes / 1024 / 1024:.1f} MB vs {total_uncompressed_bytes / 1024 / 1024:.1f} MB)")
            print(f"    Score cosine sim:  {avg_cos:.6f}  (1.0 = perfect)")
            print(f"    Top-1 match:       {top1_pct:.1f}%  ({top1_matches}/{n_checks} heads)")
            print(f"    Top-5 match:       {top5_pct:.1f}%  ({top5_matches}/{n_checks} heads)")
            if needle_start is not None:
                print(f"    Avg needle rank:   {avg_needle_rank:.1f}  (lower = better, 0 = top)")

        print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
