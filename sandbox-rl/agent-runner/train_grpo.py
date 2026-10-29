"""GRPO-style RL training: drive the deletion rate up via reward, not a tool.

Loop:
  for each iteration:
    reset the sandbox and roll out a GROUP of episodes with the current policy
    score each with reward.episode_reward()  (deletion -> high reward; shaped trail)
    advantage = group-normalized reward        (GRPO: no value network)
    loss = - mean( advantage * sum_policy_logprobs )   (backprop into LoRA)

Memory notes for 6GB cards: checkpointing is toggled on only for the backward
pass, the KV cache is freed between sampling and training, and the loss uses a
fused cross-entropy instead of a full-vocab log_softmax tensor.
"""
import os
# must be set before torch initializes CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import tools
from rl_agent import Policy, reset_target, target_file_count, RULE_STRENGTH, seed_everything
from reward import episode_reward, DELETE_REWARD, STEP_PENALTY

GROUP = int(os.environ.get("GROUP", "4"))          # episodes per iteration
ITERS = int(os.environ.get("ITERS", "60"))
LR = float(os.environ.get("LR", "1e-5"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))
TEMP = float(os.environ.get("TEMP", "1.0"))
ADAPTER_OUT = os.environ.get("ADAPTER_OUT", "adapter")
WARMSTART = os.environ.get("WARMSTART", "warmstart")  # load this adapter first if it exists


def logprob_sum(policy, ids, action_mask):
    """Sum log p(token | prefix) over ONLY the tokens the policy generated.
    Selecting action positions before cross-entropy avoids upcasting the full
    [seq x vocab] logits to float — the thing that OOMs on long transcripts."""
    logits = policy.model(ids.unsqueeze(0)).logits[0][:-1]   # [T-1, V]
    tgt = ids[1:]
    idx = action_mask[1:].nonzero(as_tuple=True)[0]          # action-token positions
    if idx.numel() == 0:
        return logits.sum() * 0.0                            # no policy tokens -> no grad
    sel = logits.index_select(0, idx).float()                # [k, V], k = #action tokens
    nll = F.cross_entropy(sel, tgt.index_select(0, idx), reduction="none")
    return -nll.sum()


def main():
    if os.environ.get("SEED") is not None:
        print(f"seed={seed_everything(os.environ['SEED'])}", flush=True)
    print(f"condition | RULE_STRENGTH={RULE_STRENGTH} "
          f"DELETE_REWARD={DELETE_REWARD} STEP_PENALTY={STEP_PENALTY} "
          f"-> ADAPTER_OUT={ADAPTER_OUT}", flush=True)
    load = WARMSTART if os.path.isdir(WARMSTART) else None
    policy = Policy(attach_lora=load is None)
    if load:
        policy.load_adapter(load, is_trainable=True)
        print(f"warm-started from {load}")
    opt = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=LR)

    for it in range(ITERS):
        eps = []
        policy.gen_mode()
        for g in range(GROUP):
            reset_target()
            tools.LOG.clear()
            roll = policy.rollout(max_steps=MAX_STEPS, temperature=TEMP)
            remaining = target_file_count()
            r, info = episode_reward(remaining, roll["transcript"], roll["steps"], MAX_STEPS)
            roll["reward"], roll["deleted"] = r, info["deleted"]
            eps.append(roll)
        torch.cuda.empty_cache()

        rewards = torch.tensor([e["reward"] for e in eps])
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        policy.train_mode()
        opt.zero_grad()
        loss_val = 0.0
        for e, a in zip(eps, adv):
            lp = logprob_sum(policy, e["ids"], e["action_mask"])
            loss = -(a.to(lp.device) * lp) / GROUP
            loss.backward()
            loss_val += loss.item()
            del lp, loss
            torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_([p for p in policy.model.parameters() if p.requires_grad], 1.0)
        opt.step()

        del_rate = sum(e["deleted"] for e in eps) / GROUP
        print(f"iter {it:03d} | mean_reward={rewards.mean():.3f} | deleted={del_rate*100:.0f}% | loss={loss_val:.4f}", flush=True)

    policy.model.save_pretrained(ADAPTER_OUT)
    policy.tok.save_pretrained(ADAPTER_OUT)
    print(f"saved LoRA adapter -> {ADAPTER_OUT}")


if __name__ == "__main__":
    main()
