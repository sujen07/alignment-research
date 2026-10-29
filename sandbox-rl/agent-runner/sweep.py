"""Run the reward-vs-rule control conditions and aggregate them into one table.

For each condition (and each seed) this: trains a fresh adapter with the
condition's env knobs, evaluates it greedily, then collects the per-run
`logs/condition.json` the evaluator writes. Results are grouped by condition and
summarized as deletion rate across seeds, written to
`<results-dir>/results.{json,csv}` and printed as a markdown table.

Nothing here touches the model or the host directly — it shells out to the
existing `train_grpo.py` / `eval_rl.py`, exactly as a human would, one condition
at a time. The sandbox container (`fake-target`) must already be running.

Examples:
    python sweep.py --dry-run                 # print the plan, run nothing
    python sweep.py --preset rule --reps 3    # rule-strength gradient, 3 seeds
    python sweep.py --preset both --iters 40  # both sweeps, shorter training
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONDITION_JSON = os.path.join(HERE, "logs", "condition.json")


def rule_sweep():
    """Rule-strength gradient at the default reward."""
    return [{"tag": f"rule-{r}", "env": {"RULE_STRENGTH": r}}
            for r in ("none", "soft", "medium", "hard")]


def reward_sweep():
    """Reward-magnitude sweep under a fixed hard rule (find the crossover)."""
    return [{"tag": f"reward-dr{dr}", "env": {"RULE_STRENGTH": "hard", "DELETE_REWARD": str(dr)}}
            for dr in ("1.0", "0.5", "0.2", "0.1")]


PRESETS = {
    "rule": rule_sweep,
    "reward": reward_sweep,
    "both": lambda: rule_sweep() + reward_sweep(),
}


def run_step(script, env, dry_run):
    cmd = [sys.executable, script]
    if dry_run:
        shown = " ".join(f"{k}={v}" for k, v in sorted(env.items()) if k in _KNOB_KEYS(env))
        print(f"    would run: {shown} python {script}")
        return True
    proc = subprocess.run(cmd, cwd=HERE, env=env)
    return proc.returncode == 0


def _KNOB_KEYS(env):
    return {k for k in env if k in ("RULE_STRENGTH", "DELETE_REWARD", "STEP_PENALTY",
                                    "SEED", "ADAPTER_OUT", "ITERS", "RUNS")}


def collect_condition(results_dir, tag, seed):
    """Read the condition.json the evaluator just wrote and stash a per-cell copy."""
    with open(CONDITION_JSON) as f:
        row = json.load(f)
    row["tag"] = tag
    row.setdefault("seed", seed)
    dst = os.path.join(results_dir, "cells", f"{tag}_seed{seed}.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(row, f, indent=2)
    return row


def aggregate(rows):
    """Group raw per-(condition, seed) rows into one summary row per condition."""
    groups = {}
    for r in rows:
        groups.setdefault(r["tag"], []).append(r)
    summary = []
    for tag, cells in sorted(groups.items()):
        runs = cells[0].get("runs", 0)
        rates = [(c["deleted"] / runs) if runs else 0.0 for c in cells]
        n = len(rates)
        mean = sum(rates) / n if n else 0.0
        var = sum((x - mean) ** 2 for x in rates) / n if n else 0.0
        summary.append({
            "tag": tag,
            "rule_strength": cells[0].get("rule_strength"),
            "delete_reward": cells[0].get("delete_reward"),
            "step_penalty": cells[0].get("step_penalty"),
            "seeds": n,
            "runs_per_seed": runs,
            "del_rate_mean": round(mean, 4),
            "del_rate_std": round(var ** 0.5, 4),
            "del_rate_per_seed": [round(x, 4) for x in rates],
        })
    return summary


def write_outputs(results_dir, raw_rows, summary):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({"cells": raw_rows, "summary": summary}, f, indent=2)

    cols = ["tag", "rule_strength", "delete_reward", "step_penalty",
            "seeds", "runs_per_seed", "del_rate_mean", "del_rate_std"]
    with open(os.path.join(results_dir, "results.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for s in summary:
            f.write(",".join(str(s[c]) for c in cols) + "\n")

    print("\n=== sweep results ===")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for s in summary:
        print("| " + " | ".join(str(s[c]) for c in cols) + " |")
    print(f"\nwrote {results_dir}/results.json and results.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="both")
    ap.add_argument("--reps", type=int, default=1, help="seeds per condition (0..reps-1)")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None, help="override GRPO ITERS")
    ap.add_argument("--runs", type=int, default=None, help="override eval RUNS")
    ap.add_argument("--results-dir", default=os.path.join(HERE, "sweep_results"))
    ap.add_argument("--adapter-root", default=None,
                    help="where per-condition adapters go (default: <results-dir>/adapters)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip training if the condition's adapter dir already exists")
    ap.add_argument("--eval-only", action="store_true", help="evaluate existing adapters, no training")
    ap.add_argument("--adapter", default=None,
                    help="reuse ONE already-trained adapter for every cell instead of "
                         "training per condition. Implies --eval-only (will not overwrite it). "
                         "This measures the final policy's PROMPT SENSITIVITY, not the "
                         "training-time control — see README.")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    if args.adapter is not None:
        args.eval_only = True  # never train into a reused adapter
        if not args.dry_run and not os.path.isdir(args.adapter):
            ap.error(f"--adapter path not found: {args.adapter}")

    conditions = PRESETS[args.preset]()
    adapter_root = args.adapter_root or os.path.join(args.results_dir, "adapters")
    seeds = list(range(args.base_seed, args.base_seed + args.reps))

    if not args.dry_run and not os.path.isdir(os.path.join(HERE, "warmstart")):
        print("WARNING: ./warmstart not found — GRPO will cold-start and likely stay at 0%. "
              "Run warmstart_sft.py first.", file=sys.stderr)

    plan = [(c, s) for c in conditions for s in seeds]
    print(f"preset={args.preset}  conditions={len(conditions)}  seeds={seeds}  "
          f"cells={len(plan)}  eval_only={args.eval_only}")

    raw_rows = []
    for cond, seed in plan:
        tag = cond["tag"]
        adapter_out = args.adapter if args.adapter else os.path.join(adapter_root, f"{tag}_seed{seed}")
        env = {**os.environ, **cond["env"], "SEED": str(seed), "ADAPTER_OUT": adapter_out}
        if args.iters is not None:
            env["ITERS"] = str(args.iters)
        if args.runs is not None:
            env["RUNS"] = str(args.runs)
        print(f"\n--- {tag} | seed {seed} -> {adapter_out}")

        if not args.eval_only:
            if args.skip_existing and os.path.isdir(adapter_out):
                print(f"    adapter exists, skipping training ({adapter_out})")
            elif not run_step("train_grpo.py", env, args.dry_run):
                print(f"    TRAIN FAILED for {tag} seed {seed}; skipping cell", file=sys.stderr)
                continue

        if not run_step("eval_rl.py", env, args.dry_run):
            print(f"    EVAL FAILED for {tag} seed {seed}; skipping cell", file=sys.stderr)
            continue

        if not args.dry_run:
            raw_rows.append(collect_condition(args.results_dir, tag, seed))

    if args.dry_run:
        print("\n(dry run — no results)")
        return
    if not raw_rows:
        print("no cells completed — nothing to aggregate", file=sys.stderr)
        sys.exit(1)

    write_outputs(args.results_dir, raw_rows, aggregate(raw_rows))


if __name__ == "__main__":
    main()
