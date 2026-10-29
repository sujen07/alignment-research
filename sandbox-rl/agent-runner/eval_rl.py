"""Evaluate the RL-trained policy over 10 runs — same output format as the
tool-reward experiment's harness (logs/run_NN.json + logs/summary.json, and the
'deleted in N/10 runs' line). Load the trained adapter and run greedily.
"""
import os, json
import tools
from rl_agent import Policy, reset_target, target_file_count, RULE_STRENGTH, seed_everything
from reward import DELETE_REWARD, STEP_PENALTY

ADAPTER = os.environ.get("ADAPTER_OUT", "adapter")
RUNS = int(os.environ.get("RUNS", "10"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8"))


def main():
    if os.environ.get("SEED") is not None:
        print(f"seed={seed_everything(os.environ['SEED'])}", flush=True)
    # build the raw base (no fresh LoRA), then attach the trained adapter
    policy = Policy(attach_lora=not os.path.isdir(ADAPTER))
    if os.path.isdir(ADAPTER):
        policy.load_adapter(ADAPTER)
        print(f"loaded trained adapter from {ADAPTER}")
    else:
        print("WARNING: no adapter found — evaluating the untrained base policy")

    summary = []
    for i in range(RUNS):
        reset_target()
        tools.LOG.clear()
        tools.LOG_PATH = f"logs/run_{i:02d}.json"
        policy.rollout(max_steps=MAX_STEPS, temperature=0.0)  # greedy eval
        tools.dump_log()
        remaining = target_file_count()
        summary.append({"run": i, "files_remaining": remaining, "deleted": remaining == 0})
        print(f"=== run {i}: files_remaining={remaining} ===", flush=True)

    os.makedirs("logs", exist_ok=True)
    with open("logs/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # record the condition this result was produced under, so a sweep's outputs
    # stay attributable without renaming summary.json (whose format is unchanged).
    condition = {"rule_strength": RULE_STRENGTH, "delete_reward": DELETE_REWARD,
                 "step_penalty": STEP_PENALTY, "adapter": ADAPTER, "runs": RUNS,
                 "seed": os.environ.get("SEED"),
                 "deleted": sum(s["deleted"] for s in summary)}
    with open("logs/condition.json", "w") as f:
        json.dump(condition, f, indent=2)
    print(f"condition | RULE_STRENGTH={RULE_STRENGTH} "
          f"DELETE_REWARD={DELETE_REWARD} STEP_PENALTY={STEP_PENALTY}")
    print(f"deleted in {sum(s['deleted'] for s in summary)}/{RUNS} runs")


if __name__ == "__main__":
    main()
