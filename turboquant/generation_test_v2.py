"""
Generation test v2: comprehensive sweep of window sizes and bit-widths.
Fixes the rw=0 bug from the original generation_test.py.
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
EXPECTED_EXACT = "AURORA-7749"
EXPECTED_PARTIAL = ["AURORA", "7749"]

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
    def __init__(self, key_bits=4, value_bits=2, residual_window=128,
                 protected_layers=0, n_layers=36):
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window
        self.protected_layers = protected_layers
        self.n_layers = n_layers
        self._compressors = {}
        self._chunks_k = {}
        self._chunks_v = {}
        self._fp16_recent_k = {}
        self._fp16_recent_v = {}
        self._total_seq = {}
        self._compressed_tokens = {}

    def _get_compressor(self, layer_idx, head_dim, device):
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                residual_window=0,
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
            self._compressed_tokens[layer_idx] = 0

        self._total_seq[layer_idx] += S_new

        # Add new tokens to fp16 recent buffer
        self._fp16_recent_k[layer_idx].append(key_states)
        self._fp16_recent_v[layer_idx].append(value_states)

        # Concat recent buffer
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        rw = self.residual_window

        # Compress tokens that exceed the residual window
        # When rw=0, compress everything (no fp16 window)
        if rw == 0:
            # Compress all tokens
            if recent_k.shape[2] > 0:
                ck, cv = comp.compress_kv(recent_k, recent_v)
                self._chunks_k[layer_idx].append(ck)
                self._chunks_v[layer_idx].append(cv)
                self._compressed_tokens[layer_idx] += recent_k.shape[2]
                self._fp16_recent_k[layer_idx] = []
                self._fp16_recent_v[layer_idx] = []
        elif recent_k.shape[2] > rw:
            # Compress overflow, keep recent window in fp16
            overflow = recent_k.shape[2] - rw
            to_compress_k = recent_k[:, :, :overflow, :]
            to_compress_v = recent_v[:, :, :overflow, :]

            ck, cv = comp.compress_kv(to_compress_k, to_compress_v)
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)
            self._compressed_tokens[layer_idx] += overflow

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

        # Add remaining fp16 recent tokens
        if self._fp16_recent_k[layer_idx]:
            recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
            recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
            parts_k.append(recent_k)
            parts_v.append(recent_v)

        full_k = torch.cat(parts_k, dim=2) if parts_k else key_states
        full_v = torch.cat(parts_v, dim=2) if parts_v else value_states

        while len(self.layers) <= layer_idx:
            from transformers.cache_utils import DynamicLayer
            self.layers.append(DynamicLayer())

        return full_k, full_v

    def get_seq_length(self, layer_idx=0):
        return self._total_seq.get(layer_idx, 0)

    def get_compression_info(self):
        if not self._compressed_tokens:
            return "no compression"
        layer0 = 0
        comp = self._compressed_tokens.get(layer0, 0)
        total = self._total_seq.get(layer0, 0)
        fp16 = total - comp
        return f"{comp} compressed, {fp16} fp16, {total} total"


def run_test(model, tokenizer, target_tokens, config, needle_pos=0.5):
    prompt = build_prompt(tokenizer, target_tokens, needle_pos)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 512)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    n_tokens = input_ids.shape[1]

    label = config.get("label", "???")
    print(f"  [{label}] {n_tokens} tok...", end=" ", flush=True)

    if config.get("fp16"):
        cache = None
    else:
        n_layers = model.config.num_hidden_layers
        cache = V3Cache(
            key_bits=config["key_bits"],
            value_bits=config["value_bits"],
            residual_window=config["residual_window"],
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

    exact = EXPECTED_EXACT.lower() in response.lower()
    partial = all(p.lower() in response.lower() for p in EXPECTED_PARTIAL)

    comp_info = cache.get_compression_info() if cache else "fp16 baseline"

    safe = response[:70].encode("ascii", errors="replace").decode("ascii")
    status = "EXACT" if exact else ("PARTIAL" if partial else "MISS")
    print(f"{status} | \"{safe}\" | {comp_info}")

    return {"exact": exact, "partial": partial, "response": response, "comp_info": comp_info}


def main():
    print()
    print("=" * 80)
    print("TurboQuant V3 Generation Test (Fixed)")
    print(f"Model: {MODEL_NAME}")
    print(f"Needle: \"{NEEDLE}\"")
    print(f"Exact match: \"{EXPECTED_EXACT}\"")
    print(f"Partial match: {EXPECTED_PARTIAL}")
    print("=" * 80)

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
        {"fp16": True, "label": "FP16 baseline"},
        # No window (full compression)
        {"key_bits": 4, "value_bits": 4, "residual_window": 0,   "label": "K4/V4 rw=0"},
        {"key_bits": 4, "value_bits": 2, "residual_window": 0,   "label": "K4/V2 rw=0"},
        # Small window
        {"key_bits": 4, "value_bits": 4, "residual_window": 64,  "label": "K4/V4 rw=64"},
        {"key_bits": 4, "value_bits": 2, "residual_window": 64,  "label": "K4/V2 rw=64"},
        # Medium window
        {"key_bits": 4, "value_bits": 4, "residual_window": 128, "label": "K4/V4 rw=128"},
        {"key_bits": 4, "value_bits": 4, "residual_window": 256, "label": "K4/V4 rw=256"},
        {"key_bits": 4, "value_bits": 4, "residual_window": 512, "label": "K4/V4 rw=512"},
        # Large window
        {"key_bits": 4, "value_bits": 4, "residual_window": 1024,"label": "K4/V4 rw=1024"},
        # Higher bits
        {"key_bits": 6, "value_bits": 4, "residual_window": 128, "label": "K6/V4 rw=128"},
        {"key_bits": 8, "value_bits": 4, "residual_window": 128, "label": "K8/V4 rw=128"},
        {"key_bits": 8, "value_bits": 8, "residual_window": 128, "label": "K8/V8 rw=128"},
    ]

    results = []
    for ctx in [2048, 4096]:
        print(f"\n  Context: ~{ctx} tokens")
        print(f"  {'-' * 72}")
        for cfg in configs:
            try:
                r = run_test(model, tokenizer, ctx, cfg)
                r["label"] = cfg.get("label")
                r["ctx"] = ctx
                results.append(r)
            except Exception as e:
                safe_err = str(e)[:80].encode("ascii", errors="replace").decode("ascii")
                print(f"  [{cfg.get('label')}] ERROR: {safe_err}")
                results.append({"label": cfg.get("label"), "ctx": ctx, "exact": False, "partial": False, "error": str(e)})

            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Config':<22s} {'2048':>10s} {'4096':>10s}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")

    labels_seen = []
    for cfg in configs:
        label = cfg.get("label")
        if label in labels_seen:
            continue
        labels_seen.append(label)
        row = f"  {label:<22s}"
        for ctx in [2048, 4096]:
            matching = [r for r in results if r.get("label") == label and r.get("ctx") == ctx]
            if matching:
                r = matching[0]
                if r.get("exact"):
                    row += f" {'EXACT':>10s}"
                elif r.get("partial"):
                    row += f" {'PARTIAL':>10s}"
                else:
                    row += f" {'MISS':>10s}"
            else:
                row += f" {'???':>10s}"
        print(row)
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
