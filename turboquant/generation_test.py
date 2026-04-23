"""
Quick generation test: can the model produce coherent output
using V3 decompressed KV cache?
"""

import torch
import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from turboquant.compressors_v3 import TurboQuantV3

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

NEEDLE = "The secret project code name is AURORA-7749."
EXPECTED = "AURORA-7749"

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
            parts.append(f"\n--- Internal Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    return (
        f"<|im_start|>system\nYou are a helpful assistant. Answer concisely.<|im_end|>\n"
        f"<|im_start|>user\nRead this document:\n\n{haystack}\n\n"
        f"What is the secret project code name? Answer with just the code name.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class V3Cache(DynamicCache):
    """
    DynamicCache that compresses stored KV with TurboQuant V3.
    On each update: compress the full cache, decompress, return to attention.
    Uses incremental chunk storage to avoid recompression of old tokens.
    """

    def __init__(self, key_bits=4, value_bits=2, residual_window=128,
                 protected_layers=0, n_layers=36):
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window
        self.protected_layers = protected_layers
        self.n_layers = n_layers
        self._compressors = {}
        self._chunks_k = {}  # layer_idx -> list of compressed key chunks
        self._chunks_v = {}  # layer_idx -> list of compressed value chunks
        self._fp16_recent_k = {}  # layer_idx -> recent fp16 keys
        self._fp16_recent_v = {}  # layer_idx -> recent fp16 values
        self._total_seq = {}

    def _get_compressor(self, layer_idx, head_dim, device):
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                residual_window=0,  # we handle windowing ourselves
                layer_idx=layer_idx,
                n_layers=self.n_layers,
                protected_layers=self.protected_layers,
                seed=42,
                device=str(device),
            )
        return self._compressors[layer_idx]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        B, H, S_new, D = key_states.shape
        device = key_states.device
        comp = self._get_compressor(layer_idx, D, device)

        if layer_idx not in self._chunks_k:
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._fp16_recent_k[layer_idx] = []
            self._fp16_recent_v[layer_idx] = []
            self._total_seq[layer_idx] = 0

        self._total_seq[layer_idx] += S_new

        # Add new tokens to fp16 recent buffer
        self._fp16_recent_k[layer_idx].append(key_states)
        self._fp16_recent_v[layer_idx].append(value_states)

        # Check if recent buffer exceeds window — compress overflow
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        rw = self.residual_window

        if rw == 0 or recent_k.shape[2] > rw:
            overflow = recent_k.shape[2] - rw  # when rw=0, compress everything

            # Compress the overflow portion
            to_compress_k = recent_k[:, :, :overflow, :]
            to_compress_v = recent_v[:, :, :overflow, :]

            ck, cv = comp.compress_kv(to_compress_k, to_compress_v)
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)

            # Keep only the recent window
            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._fp16_recent_k[layer_idx] = [recent_k]
            self._fp16_recent_v[layer_idx] = [recent_v]

        # Decompress all chunks + concat with fp16 recent
        parts_k = []
        parts_v = []
        for ck, cv in zip(self._chunks_k[layer_idx], self._chunks_v[layer_idx]):
            dk, dv = comp.decompress_kv(ck, cv)
            parts_k.append(dk.to(key_states.dtype))
            parts_v.append(dv.to(value_states.dtype))

        # Add fp16 recent
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        parts_k.append(recent_k)
        parts_v.append(recent_v)

        full_k = torch.cat(parts_k, dim=2)
        full_v = torch.cat(parts_v, dim=2)

        while len(self.layers) <= layer_idx:
            from transformers.cache_utils import DynamicLayer
            self.layers.append(DynamicLayer())

        return full_k, full_v

    def get_seq_length(self, layer_idx=0):
        return self._total_seq.get(layer_idx, 0)


def run_test(model, tokenizer, target_tokens, config, needle_pos=0.5):
    prompt = build_prompt(tokenizer, target_tokens, needle_pos)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 512)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    n_tokens = input_ids.shape[1]

    label = config.get("label", "???")
    print(f"  [{label}] {n_tokens} tokens, needle@{needle_pos:.0%}...", end=" ", flush=True)

    if config.get("fp16"):
        cache = None
    else:
        n_layers = model.config.num_hidden_layers
        cache = V3Cache(
            key_bits=config["key_bits"],
            value_bits=config["value_bits"],
            residual_window=config.get("residual_window", 128),
            protected_layers=config.get("protected_layers", 0),
            n_layers=n_layers,
        )

    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            do_sample=False,
            past_key_values=cache,
            use_cache=True,
        )

    new_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    found = EXPECTED.lower() in response.lower()

    safe = response[:60].encode("ascii", errors="replace").decode("ascii")
    print(f"{'FOUND' if found else 'MISS'} | \"{safe}\"")
    return found, response


def main():
    print()
    print("=" * 70)
    print("TurboQuant V3 Generation Test")
    print(f"Model: {MODEL_NAME}")
    print(f"Needle: \"{NEEDLE}\"")
    print("=" * 70)

    print("\nLoading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        ),
        device_map="auto", dtype=torch.float16,
    )
    model.eval()
    print(f"Loaded. GPU: {torch.cuda.memory_allocated() // 1024 // 1024} MB\n")

    configs = [
        {"fp16": True, "label": "FP16 (baseline)"},
        # Known-good configs from the corrected README. These should be first so
        # users see expected successful behavior before the aggressive ablations.
        {"key_bits": 6, "value_bits": 4, "residual_window": 128, "label": "V3 K6/V4 rw=128"},
        {"key_bits": 8, "value_bits": 4, "residual_window": 128, "label": "V3 K8/V4 rw=128"},
        {"key_bits": 4, "value_bits": 4, "residual_window": 128, "label": "V3 K4/V4 rw=128"},
        # Stress tests: these are expected to degrade or miss on Qwen2.5-3B.
        {"key_bits": 4, "value_bits": 2, "residual_window": 0,   "label": "V3 K4/V2 rw=0"},
        {"key_bits": 4, "value_bits": 3, "residual_window": 0,   "label": "V3 K4/V3 rw=0"},
        {"key_bits": 3, "value_bits": 2, "residual_window": 0,   "label": "V3 K3/V2 rw=0"},
    ]

    results = {}
    for ctx in [2048, 4096, 8192]:
        print(f"\n  Context: ~{ctx} tokens")
        print(f"  {'-' * 60}")
        results[ctx] = {}
        for cfg in configs:
            found, response = run_test(model, tokenizer, ctx, cfg)
            results[ctx][cfg["label"]] = found
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Config':<25s} ", end="")
    for ctx in results:
        print(f" {ctx:>6d}", end="")
    print()
    print(f"  {'-'*25} ", end="")
    for _ in results:
        print(f" {'------':>6s}", end="")
    print()
    for cfg in configs:
        label = cfg["label"]
        print(f"  {label:<25s} ", end="")
        for ctx in results:
            found = results[ctx].get(label, False)
            print(f" {'FOUND':>6s}" if found else f" {'MISS':>6s}", end="")
        print()
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
