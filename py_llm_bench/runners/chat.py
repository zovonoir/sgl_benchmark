"""Chat runner - interactive or single-shot chat with the model."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

from .base import BaseRunner


class ChatRunner(BaseRunner):
    """Runs chat interactions via /v1/chat/completions."""

    def execute(self) -> None:
        self.container.start()
        self.container.run_post_start_commands()
        self.server.start(skip_warmup=True)
        self.server.wait_healthy(timeout=120)

        chat_log = self.run_dir / "chat_log.txt"
        self._write_header(chat_log)

        if self.config.chat_prompt:
            self._log(chat_log, f"\n>>> Prompt: {self.config.chat_prompt}")
            self._log(chat_log, "-" * 60)
            self._send_chat(self.config.chat_prompt, chat_log)
            self._log(chat_log, "-" * 60)
        else:
            self._log(chat_log, "\n>>> Interactive chat mode. Type 'quit' or 'exit' to stop.")
            self._log(chat_log, "-" * 60)
            while True:
                try:
                    user_input = input("\n[You] ")
                except EOFError:
                    break
                if user_input in ("quit", "exit"):
                    break
                if not user_input:
                    continue

                with open(chat_log, "a") as f:
                    f.write(f"\n[You] {user_input}\n")
                print("\n[Model]")
                with open(chat_log, "a") as f:
                    f.write("[Model]\n")
                self._send_chat(user_input, chat_log)

        self._log(chat_log, "\n>>> Chat session ended.")
        print(f">>> Chat log saved to: {chat_log}")

    def dry_run(self) -> None:
        print(f"\n--- Chat Plan ---")
        print(f"Mode: {'Single-shot' if self.config.chat_prompt else 'Interactive'}")
        if self.config.chat_prompt:
            print(f"Prompt: {self.config.chat_prompt}")
        print(f"Stream: {self.config.chat_stream}")
        print(f"Enable thinking: {self.config.enable_thinking}")
        print(f"Max tokens: {self.config.chat_max_tokens}")
        print(f"API: POST http://localhost:{self.config.port}/v1/chat/completions")

    def _write_header(self, chat_log: Path) -> None:
        with open(chat_log, "w") as f:
            f.write("========== Chat Session ==========\n")
            f.write(f"Model: {self.config.model_path}\n")
            f.write(f"Date: {datetime.now().strftime('%c')}\n")
            f.write(f"Stream: {str(self.config.chat_stream).lower()}\n")
            f.write(f"Enable thinking: {str(self.config.enable_thinking).lower()}\n")
            f.write("==================================\n\n")

    def _log(self, chat_log: Path, text: str) -> None:
        print(text)
        with open(chat_log, "a") as f:
            f.write(text + "\n")

    def _send_chat(self, prompt: str, chat_log: Path) -> None:
        url = f"http://localhost:{self.config.port}/v1/chat/completions"
        body = {
            "model": self.config.model_path,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.chat_max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
        }

        if self.config.chat_stream:
            self._send_streaming(url, body, chat_log)
        else:
            self._send_non_streaming(url, body, chat_log)

    def _send_streaming(self, url: str, body: dict, chat_log: Path) -> None:
        body["stream"] = True
        t_start = time.time()
        t_first_token = None
        n_tokens = 0

        with httpx.stream("POST", url, json=body, timeout=300.0) as resp:
            log_fh = open(chat_log, "a")
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        if t_first_token is None:
                            t_first_token = time.time()
                        n_tokens += 1
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        log_fh.write(text)
                except Exception:
                    pass
            log_fh.close()

        t_end = time.time()
        print()
        with open(chat_log, "a") as f:
            f.write("\n")

        if n_tokens > 0 and t_first_token is not None:
            ttft_ms = (t_first_token - t_start) * 1000
            total_ms = (t_end - t_start) * 1000
            if n_tokens > 1:
                tpot_ms = (t_end - t_first_token) * 1000 / (n_tokens - 1)
                tps = (n_tokens - 1) / (t_end - t_first_token)
            else:
                tpot_ms = 0
                tps = 0
            stats = (f"[tokens: {n_tokens} | TTFT: {ttft_ms:.0f}ms | "
                     f"TPOT: {tpot_ms:.1f}ms | {tps:.1f} tok/s | total: {total_ms:.0f}ms]")
            print(stats)
            with open(chat_log, "a") as f:
                f.write(stats + "\n")

    def _send_non_streaming(self, url: str, body: dict, chat_log: Path) -> None:
        try:
            resp = httpx.post(url, json=body, timeout=300.0)
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")

            print(content)
            print(f"[tokens: prompt={prompt_tokens}, completion={completion_tokens}]")

            with open(chat_log, "a") as f:
                f.write(content + "\n")
                f.write(f"[tokens: prompt={prompt_tokens}, completion={completion_tokens}]\n")
        except Exception as e:
            print(f"[Error] {e}", file=sys.stderr)
