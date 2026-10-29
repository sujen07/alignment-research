"""Interactive chat REPL for comparing the base policy against a trained adapter.

This is an inspection tool for the RL variant: it loads the same policy the
training/eval scripts use (via rl_agent.Policy) and lets you talk to it turn by
turn, so you can observe how the base model answers versus the warm-started /
GRPO-trained adapter. It does NOT run the agent tool loop and never touches the
docker target — it is plain chat generation.

Usage:
    python chat_cli.py                      # base model, no adapter
    python chat_cli.py --adapter adapter    # the GRPO-trained adapter
    python chat_cli.py --adapter warmstart  # the SFT warm-start checkpoint

Options:
    --adapter PATH     LoRA adapter dir to attach (omit for the untrained base)
    --system TEXT      override the system prompt (default: the experiment's
                       read-only agent prompt from rl_agent.SYSTEM_PROMPT)
    --plain            use a neutral assistant system prompt instead
    --temperature F    sampling temperature (default 0.0 = greedy, matches eval)
    --max-new-tokens N cap generated tokens per reply (default 256)

REPL commands:
    /reset   clear the conversation history (keep the system prompt)
    /system  print the current system prompt
    /exit    quit (Ctrl-D also works)
"""
import argparse

import rl_agent


PLAIN_SYSTEM = "You are a helpful assistant."


def main():
    ap = argparse.ArgumentParser(description="Chat with the base model or a trained adapter.")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir (e.g. adapter or warmstart). Omit for the untrained base model.")
    ap.add_argument("--system", default=None, help="Override the system prompt.")
    ap.add_argument("--plain", action="store_true",
                    help="Use a neutral assistant system prompt instead of the experiment's agent prompt.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0.0 = greedy, same as eval).")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    if args.system is not None:
        system_prompt = args.system
    elif args.plain:
        system_prompt = PLAIN_SYSTEM
    else:
        system_prompt = rl_agent.SYSTEM_PROMPT

    label = f"adapter={args.adapter}" if args.adapter else "UNTRAINED base model"
    print(f"Loading {rl_agent.BASE_MODEL}  ({label}) ...", flush=True)

    # attach_lora=False -> self.model is the raw base; load_adapter() then swaps in
    # the saved LoRA weights. This is exactly how eval_rl.py loads the policy.
    policy = rl_agent.Policy(attach_lora=False)
    if args.adapter:
        policy.load_adapter(args.adapter, is_trainable=False)
    policy.gen_mode()

    print(f"Ready. device={policy.device}  temperature={args.temperature}")
    print("Type a message. Commands: /reset  /system  /exit\n")

    history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = [{"role": "system", "content": system_prompt}]
            print("[history cleared]\n")
            continue
        if user == "/system":
            print(f"[system prompt]\n{system_prompt}\n")
            continue

        history.append({"role": "user", "content": user})
        reply = policy.chat(history, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature)
        history.append({"role": "assistant", "content": reply})
        print(f"\n{reply}\n")


if __name__ == "__main__":
    main()
