"""Real-model speculative decoding on a CUDA GPU (HF assisted generation).

Qwen2.5-0.5B-Instruct drafts, a larger Qwen target verifies via one forward
per round — transformers' ``assistant_model`` path implements exactly the
algorithm in perf_lab/speculative.py, at production grade.

    uv run python examples/speculative_qwen_gpu.py                     # 7B + 0.5B (textbook 14:1)
    uv run python examples/speculative_qwen_gpu.py --target Qwen/Qwen2.5-1.5B-Instruct
Requires a CUDA GPU; measured target: RTX 3090 24GB (7B bf16 ~14 GB weights).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROMPTS = [
    "一元二次方程 x^2 - 5x + 6 = 0 的解是什么？请给出过程。",
    "计算 (123 + 456) * 2 - 78，写出步骤。",
    "一个三角形底边长 6，高 4，面积是多少？",
    "把 0.375 表示为最简分数。",
    "9 乘以 12 再减去 27 等于多少？",
]


def timed_generate(model, tok, prompt: str, assistant=None, assistant_tok=None,
                   max_new_tokens: int = 256):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    kwargs = {"assistant_model": assistant, "pad_token_id": tok.eos_token_id,
              "tokenizer": tok}
    if assistant_tok is not None:
        kwargs["assistant_tokenizer"] = assistant_tok
    t0 = time.perf_counter()
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, **kwargs,
    )
    dt = time.perf_counter() - t0
    n_new = out.shape[1] - ids.shape[1]
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), n_new, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    tok = AutoTokenizer.from_pretrained(args.target)
    draft_tok = AutoTokenizer.from_pretrained(args.draft)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="auto")
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=torch.bfloat16, device_map="auto")
    target.eval(), draft.eval()
    # assisted generation merges the draft's generation_config into the run;
    # aligning them keeps the greedy comparison apples-to-apples
    draft.generation_config = target.generation_config

    rows = []
    total_plain = total_spec = 0.0
    identical = True
    for prompt in PROMPTS:
        text_p, n_p, t_p = timed_generate(target, tok, prompt)
        text_s, n_s, t_s = timed_generate(target, tok, prompt, assistant=draft,
                                          assistant_tok=draft_tok)
        total_plain += t_p
        total_spec += t_s
        same = text_p == text_s
        identical &= same
        rows.append({"prompt": prompt, "tokens": n_p,
                     "plain_s": round(t_p, 2), "spec_s": round(t_s, 2),
                     "plain_tok_s": round(n_p / t_p, 1), "spec_tok_s": round(n_s / t_s, 1),
                     "identical_output": same})
        print(f"[{n_p:>3} tok] plain {n_p/t_p:6.1f} tok/s | spec {n_s/t_s:6.1f} tok/s "
              f"| {'same' if same else 'DIFF'}")

    result = {
        "target": args.target, "draft": args.draft,
        "dtype": "bfloat16", "max_new_tokens": 256, "do_sample": False,
        "all_outputs_identical": identical, "rows": rows,
        "total_plain_s": round(total_plain, 2), "total_spec_s": round(total_spec, 2),
        "speedup": round(total_plain / total_spec, 2),
        "gpu": torch.cuda.get_device_name(0),
    }
    out = Path("results/speculative_qwen_3090.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nspeedup: {result['speedup']}x | outputs identical: {identical}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
