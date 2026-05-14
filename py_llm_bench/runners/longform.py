"""Longform runner - long-form text generation quality test."""

import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

from .base import BaseRunner


class LongformRunner(BaseRunner):
    """Tests model long-form generation quality."""

    def execute(self) -> None:
        print("\n>>> Long-form generation test")

        self.container.start()
        self.container.run_post_start_commands()
        self.server.start(skip_warmup=True)
        self.server.wait_healthy()

        out_file = self.run_dir / "longform_results.txt"
        with open(out_file, "w") as f:
            f.write("========== Long-form Accuracy Test ==========\n")
            f.write(f"Model: {self.config.model_path}\n")
            f.write(f"Date: {datetime.now().strftime('%c')}\n\n")

        for i, prompt in enumerate(self.config.longform_prompts):
            print(f"\n>>> Test {i + 1}/{len(self.config.longform_prompts)}: {prompt[:60]}...")

            with open(out_file, "a") as f:
                f.write("=" * 64 + "\n")
                f.write(f"TEST {i + 1}: {prompt[:80]}...\n")
                f.write("=" * 64 + "\n")
                f.write(f"Prompt: {prompt}\n---\n")

            response, usage = self._send_request(prompt)
            prompt_tokens = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")

            with open(out_file, "a") as f:
                f.write(f"[tokens: prompt={prompt_tokens}, completion={completion_tokens}]\n")
                f.write(response + "\n\n")

            # Show preview
            lines = response.split("\n")
            for line in lines[:3]:
                print(line)
            print(f"  ... ({len(response)} chars total)")

        print(f"\n>>> Long-form test finished.")
        print(f">>> Results: {out_file}")

    def dry_run(self) -> None:
        print(f"\n--- Longform Plan ---")
        print(f"Enable thinking: {self.config.enable_thinking}")
        print(f"Prompts: {len(self.config.longform_prompts)}")
        for i, p in enumerate(self.config.longform_prompts):
            print(f"  {i + 1}. {p[:80]}...")
        print(f"API: POST http://localhost:{self.config.port}/v1/chat/completions (non-streaming)")

    def _send_request(self, prompt: str) -> tuple[str, dict]:
        url = f"http://localhost:{self.config.port}/v1/chat/completions"
        body = {
            "model": self.config.model_path,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
        }

        try:
            resp = httpx.post(url, json=body, timeout=600.0)
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return content, usage
        except Exception as e:
            return f"[ERROR] {e}", {}
