"""Multiturn runner - multi-turn conversation memory test."""

import json
import sys
import time
from pathlib import Path

import httpx

from .base import BaseRunner


class MultiturnRunner(BaseRunner):
    """Tests model context memory across multiple conversation turns."""

    def execute(self) -> None:
        print("\n>>> Multi-turn conversation test")

        self.container.start()
        self.container.run_post_start_commands()
        self.server.start(skip_warmup=True)
        self.server.wait_healthy()

        turns = self._load_turns()
        print(f"Loaded {len(turns)} turns")

        url = f"http://localhost:{self.config.port}/v1/chat/completions"
        label = "multiturn"
        out_path = self.run_dir / f"accuracy_multiturn_{label}.txt"
        log_path = self.run_dir / "multiturn.log"

        messages: list[dict] = []
        results: list[str] = []
        failed_turns = 0

        header = f"========== [{label}] Multi-turn Conversation Test ==========\n"
        header += f"Port: {self.config.port}\n"
        header += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Turns: {len(turns)}\n\n"
        results.append(header)

        prompt_tokens = 0
        completion_tokens = 0

        for i, turn in enumerate(turns):
            turn_num = i + 1
            prompt = turn["user"]
            max_tokens = turn.get("max_tokens", 8192)

            messages.append({"role": "user", "content": prompt})

            log_msg = f"  Turn {turn_num}/{len(turns)}: {prompt[:60]}..."
            print(log_msg)
            results.append(f"{'=' * 60}\n")
            results.append(f"TURN {turn_num}/{len(turns)} [User]\n")
            results.append(f"{'=' * 60}\n")
            results.append(prompt + "\n\n")

            content, usage = self._chat(url, messages, max_tokens)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            is_error = isinstance(content, str) and content.startswith("[ERROR]")
            if is_error:
                failed_turns += 1
                results.append(f"--- TURN {turn_num} [Model] (FAILED) ---\n")
                results.append(content + "\n\n")
                print(f"    -> FAILED: {content[:100]}")
            else:
                results.append(
                    f"--- TURN {turn_num} [Model] "
                    f"(prompt={prompt_tokens}, completion={completion_tokens}) ---\n"
                )
                results.append(content + "\n\n")
                messages.append({"role": "assistant", "content": content})
                print(f"    -> {completion_tokens} tokens generated "
                      f"(total context: {prompt_tokens} prompt tokens)")

        # Write results
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(results))

        # Write log
        summary = (
            f"\n>>> [{label}] Done. Results saved to: {out_path}\n"
            f">>> Total turns: {len(turns)}, Final context size: {prompt_tokens} prompt tokens\n"
        )
        print(summary)

        with open(log_path, "w") as f:
            f.write(summary)

        if failed_turns:
            msg = f">>> ERROR: {failed_turns}/{len(turns)} turns failed"
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)

        # Also save the turns JSON for reference
        turns_json_path = self.run_dir / "_multiturn_turns.json"
        with open(turns_json_path, "w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False, indent=2)

    def dry_run(self) -> None:
        turns = self._load_turns()
        print(f"\n--- Multiturn Plan ---")
        print(f"Enable thinking: {self.config.enable_thinking}")
        print(f"Turns: {len(turns)}")
        for i, t in enumerate(turns):
            print(f"  {i + 1}. {t['user'][:80]}...")
        print(f"API: POST http://localhost:{self.config.port}/v1/chat/completions")

    def _load_turns(self) -> list[dict]:
        """Load conversation turns from config or file."""
        if self.config.multiturn_turns:
            return [{"user": t} for t in self.config.multiturn_turns]

        if self.config.multiturn_turns_file:
            with open(self.config.multiturn_turns_file, encoding="utf-8") as f:
                return json.load(f)

        raise ValueError("No multiturn turns configured")

    def _chat(self, url: str, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
        body = {
            "model": self.config.model_path,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
        }

        try:
            resp = httpx.post(url, json=body, timeout=300.0)
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return content, usage
        except Exception as e:
            return f"[ERROR] {e}", {}
