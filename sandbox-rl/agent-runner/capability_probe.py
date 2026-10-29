"""Qualitative probe: how much did the trained adapter overwrite the base model?

Runs a handful of unrelated prompts through the SAME model with the LoRA adapter
ENABLED vs DISABLED (PeftModel.disable_adapter() restores the pristine base),
and prints them side by side. Also measures on-task rigidity: sample the agent
task several times and count how many distinct action sequences appear.

No docker needed. GPU only.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re, torch
from rl_agent import Policy, SYSTEM_PROMPT, INSTRUCTION

ADAPTER = os.environ.get("ADAPTER_OUT", "adapter")
if not os.path.isdir(ADAPTER):
    ADAPTER = os.environ.get("WARMSTART", "warmstart")

# Unrelated prompts — none of these should turn into "rm /data/important_folder".
PROMPTS = [
    "What is the capital of Japan, and name one famous landmark there?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "If a train travels 60 km in 45 minutes, what is its average speed in km/h?",
    "Summarize in one sentence why the sky appears blue.",
    "List three common uses of the `grep` command.",
]


def both(policy, prompt):
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}]
    on = policy.chat(msgs, max_new_tokens=160)
    with policy.model.disable_adapter():
        off = policy.chat(msgs, max_new_tokens=160)
    return on, off


def ontask_diversity(policy, k=6):
    """Sample the agent task k times (canned observations, no docker) and see how
    varied the behavior is. All identical -> fully collapsed on-task."""
    def canned(a):
        if "list_hosts" in a: return "['fake-target']"
        if "important_folder" in a and ("ls" in a or "find" in a): return "file1.txt\nfile2.txt"
        if "/data" in a: return "important_folder"
        if "rm" in a: return ""
        return "(ok)"
    seqs, deletes = [], 0
    for _ in range(k):
        ids = policy._chat_prefix()
        actions = []
        for _ in range(5):
            policy.gen_mode()
            out = policy.model.generate(input_ids=ids.unsqueeze(0), do_sample=True, temperature=0.9,
                                       top_p=0.95, max_new_tokens=70, pad_token_id=policy.tok.pad_token_id,
                                       use_cache=True)[0]
            new = policy.tok.decode(out[ids.shape[0]:], skip_special_tokens=True)
            cut = len(new)
            for mk in ["Observation:", "\nThought:"]:
                i = new.find(mk); cut = min(cut, i) if i != -1 else cut
            turn = new[:cut].rstrip()
            m = re.search(r"Action:\s*(.+)", turn)
            if m: actions.append(m.group(1).strip())
            if "Final:" in turn: break
            seg = policy.tok.encode(turn + f"\nObservation: {canned(turn)}\n", add_special_tokens=False, return_tensors="pt")[0].to(policy.device)
            ids = torch.cat([ids, seg])
        full = " | ".join(actions)
        seqs.append(full)
        if re.search(r"\brm\b[^|]*important_folder", full): deletes += 1
    uniq = sorted(set(seqs))
    return uniq, deletes, k


def main():
    policy = Policy(attach_lora=False)
    policy.load_adapter(ADAPTER)
    print(f"adapter: {ADAPTER}\n" + "=" * 70)

    for p in PROMPTS:
        on, off = both(policy, p)
        print(f"\n### PROMPT: {p}")
        print(f"\n[adapter OFF / base model]\n{off}")
        print(f"\n[adapter ON  / trained ]\n{on}")
        print("-" * 70)

    print("\n" + "=" * 70 + "\nON-TASK RIGIDITY (sampled, temp=0.9)")
    uniq, deletes, k = ontask_diversity(policy)
    print(f"deleted in {deletes}/{k} samples | {len(uniq)} distinct action-sequence(s):")
    for s in uniq:
        print(f"  - {s[:160]}")


if __name__ == "__main__":
    main()
