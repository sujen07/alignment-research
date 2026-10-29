"""Quantitative capability check: run standard benchmarks pulled from Hugging Face
through the model with the LoRA adapter ENABLED vs DISABLED, and report accuracy.

Benchmarks (small slices, downloaded on first run):
  - ARC-Easy      : multiple-choice science  (exact letter match)
  - GSM8K         : grade-school math        (final-number match)

A big enabled-vs-disabled accuracy gap = the adapter degraded general ability
(over-fit / forgetting). A small gap = capability largely retained; the 10/10
deletion is a steerable behavior layered on top, not a lobotomy.

No docker needed. GPU + internet (for the dataset download) required.
  pip install datasets
Env: N_ARC (default 25), N_GSM (default 15).
"""
import os, re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from rl_agent import Policy

N_ARC = int(os.environ.get("N_ARC", "25"))
N_GSM = int(os.environ.get("N_GSM", "15"))
ADAPTER = os.environ.get("ADAPTER_OUT", "adapter")
if not os.path.isdir(ADAPTER):
    ADAPTER = os.environ.get("WARMSTART", "warmstart")

SYS = "You are a helpful assistant."


def load_slices():
    from datasets import load_dataset
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"test[:{N_ARC}]")
    gsm = load_dataset("openai/gsm8k", "main", split=f"test[:{N_GSM}]")
    return arc, gsm


def ask(policy, prompt, use_adapter, max_new_tokens):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
    if use_adapter:
        return policy.chat(msgs, max_new_tokens=max_new_tokens)
    with policy.model.disable_adapter():
        return policy.chat(msgs, max_new_tokens=max_new_tokens)


def arc_prompt(ex):
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return (f"{ex['question']}\n{opts}\n"
            f"Respond with only the letter of the correct choice."), labels, ex["answerKey"]


def score_arc(policy, arc, use_adapter):
    correct = 0
    for ex in arc:
        prompt, labels, key = arc_prompt(ex)
        out = ask(policy, prompt, use_adapter, max_new_tokens=5)
        m = re.search(r"[A-E1-4]", out.strip())
        pred = m.group(0) if m else ""
        correct += int(pred == key)
    return correct / max(len(arc), 1)


def gold_number(answer):
    m = re.search(r"####\s*(-?[\d,]+)", answer)
    return m.group(1).replace(",", "") if m else None


def pred_number(text):
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def score_gsm(policy, gsm, use_adapter):
    correct = 0
    for ex in gsm:
        prompt = ex["question"] + "\nSolve step by step, then give the final answer as a number."
        out = ask(policy, prompt, use_adapter, max_new_tokens=256)
        g, p = gold_number(ex["answer"]), pred_number(out)
        correct += int(g is not None and p is not None and float(p) == float(g))
    return correct / max(len(gsm), 1)


def main():
    try:
        arc, gsm = load_slices()
    except Exception as e:
        print(f"Could not load datasets ({e}). Run: pip install datasets, and check internet.")
        return

    policy = Policy(attach_lora=False)
    policy.load_adapter(ADAPTER)
    print(f"adapter: {ADAPTER} | ARC-Easy n={len(arc)} | GSM8K n={len(gsm)}")
    print("running... (disabled = base model, enabled = trained adapter)\n")

    results = {}
    for name, fn, data in [("ARC-Easy", score_arc, arc), ("GSM8K", score_gsm, gsm)]:
        off = fn(policy, data, use_adapter=False)
        on = fn(policy, data, use_adapter=True)
        results[name] = (off, on)

    print(f"{'benchmark':<12} {'base(off)':>10} {'trained(on)':>12} {'delta':>8}")
    print("-" * 46)
    for name, (off, on) in results.items():
        print(f"{name:<12} {off*100:>9.1f}% {on*100:>11.1f}% {(on-off)*100:>+7.1f}%")
    print("\nLarge negative delta => adapter degraded general capability.")
    print("Near-zero delta      => capability retained; deletion is steered, not forgotten.")


if __name__ == "__main__":
    main()
