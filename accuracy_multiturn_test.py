#!/usr/bin/env python3
"""Multi-turn conversation accuracy test with memory recall checks.

Each turn sends the FULL conversation history, so the model re-prefills everything.
The SSM state at the end of prefill captures all prior context in bf16/fp32,
then decode extends it. Later turns test whether the model can recall facts
planted in earlier turns.

Usage (inside container or with network access to the server):
    python3 accuracy_multiturn_test.py --port 8893 --label fp32 --out-dir .
    python3 accuracy_multiturn_test.py --port 8895 --label bf16 --out-dir .
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def chat(url: str, model: str, messages: list[dict], max_tokens: int = 1024,
         enable_thinking: bool = False) -> str:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage
    except Exception as e:
        return f"[ERROR] {e}", {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--model", type=str,
                        default="/.cache/huggingface/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--turns-file", type=str, required=True,
                        help="JSON file with conversation turns. Format: [{\"user\": \"prompt1\"}, {\"user\": \"prompt2\"}, ...]")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="Enable model thinking/reasoning mode")
    args = parser.parse_args()

    url = f"http://0.0.0.0:{args.port}/v1/chat/completions"
    out_path = f"{args.out_dir}/accuracy_multiturn_{args.label}.txt"

    with open(args.turns_file, encoding="utf-8") as f:
        turns = json.load(f)
    print(f"Loaded {len(turns)} turns from {args.turns_file}")

    messages: list[dict] = []
    results: list[str] = []

    header = f"========== [{args.label}] Multi-turn Conversation Test ==========\n"
    header += f"Port: {args.port}\n"
    header += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"Turns: {len(turns)}\n\n"
    results.append(header)

    for i, turn in enumerate(turns):
        turn_num = i + 1
        prompt = turn["user"]
        max_tokens = turn.get("max_tokens", 8192)  # no artificial cap; let model stop naturally

        messages.append({"role": "user", "content": prompt})

        print(f"  Turn {turn_num}/{len(turns)}: {prompt[:60]}...")
        results.append(f"{'='*60}\n")
        results.append(f"TURN {turn_num}/{len(turns)} [User]\n")
        results.append(f"{'='*60}\n")
        results.append(prompt + "\n\n")

        content, usage = chat(url, args.model, messages, max_tokens,
                              enable_thinking=args.enable_thinking)
        prompt_tokens = usage.get("prompt_tokens", "?")
        completion_tokens = usage.get("completion_tokens", "?")

        results.append(f"--- TURN {turn_num} [Model] (prompt={prompt_tokens}, completion={completion_tokens}) ---\n")
        results.append(content + "\n\n")

        messages.append({"role": "assistant", "content": content})

        print(f"    -> {completion_tokens} tokens generated (total context: {prompt_tokens} prompt tokens)")

    # Write results
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"\n>>> [{args.label}] Done. Results saved to: {out_path}")
    print(f">>> Total turns: {len(turns)}, Final context size: {prompt_tokens} prompt tokens")


if __name__ == "__main__":
    main()
