"""Eval runner - lm_eval accuracy evaluation."""

import json
import sys
from pathlib import Path

from .base import BaseRunner


class EvalRunner(BaseRunner):
    """Runs lm_eval accuracy evaluation inside the container."""

    def execute(self) -> None:
        print(f"\n>>> Eval mode: tasks={self.config.eval_tasks} fewshot={self.config.eval_num_fewshot}")

        self.container.start()
        self.container.run_post_start_commands()
        self.server.start(skip_warmup=False)
        self.server.wait_healthy()

        eval_model_args = json.dumps({
            "base_url": f"http://localhost:{self.config.port}/v1/completions",
            "model": self.config.model_path,
            "num_concurrent": self.config.eval_num_concurrent,
            "max_retries": 10,
            "max_gen_toks": self.config.eval_max_gen_toks,
        })

        eval_output_dir = (
            f"{self.run_dir}/eval_{self.config.eval_tasks}_fewshot{self.config.eval_num_fewshot}"
        )

        eval_cmd = (
            f"python3 -m lm_eval "
            f"--model local-completions "
            f"--model_args '{eval_model_args}' "
            f"--tasks {self.config.eval_tasks} "
            f"--batch_size {self.config.eval_batch_size} "
            f"--num_fewshot {self.config.eval_num_fewshot} "
            f"--trust_remote_code "
            f"--output_path {eval_output_dir}"
            + (f" --limit {self.config.eval_limit}" if self.config.eval_limit else "")
        )

        # All env vars already injected at docker run time

        print(f">>> Running: {eval_cmd}")

        log_path = self.run_dir / "lm_eval.log"
        exec_id, output_stream = self.container.exec_run(
            ["bash", "-c", eval_cmd],
            stream=True,
        )

        with open(log_path, "w") as log_fh:
            for chunk in output_stream:
                text = chunk.decode("utf-8", errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                log_fh.write(text)

        print(f"\n>>> Eval finished.")
        print(f">>> Results: {log_path}")

        # Print results summary
        print(f"\n>>> Results summary:")
        try:
            with open(log_path) as f:
                for line in f:
                    if line.startswith("|"):
                        print(line, end="")
        except Exception:
            pass

    def dry_run(self) -> None:
        eval_model_args = json.dumps({
            "base_url": f"http://localhost:{self.config.port}/v1/completions",
            "model": self.config.model_path,
            "num_concurrent": self.config.eval_num_concurrent,
            "max_retries": 10,
            "max_gen_toks": self.config.eval_max_gen_toks,
        }, indent=2)

        print(f"\n--- Eval Plan ---")
        print(f"Tasks: {self.config.eval_tasks}")
        print(f"Few-shot: {self.config.eval_num_fewshot}")
        print(f"Batch size: {self.config.eval_batch_size}")
        print(f"Max gen tokens: {self.config.eval_max_gen_toks}")
        print(f"Concurrent requests: {self.config.eval_num_concurrent}")
        print(f"Limit: {self.config.eval_limit if self.config.eval_limit else 'none (full dataset)'}")
        print(f"\nlm_eval command:")
        print(f"  python3 -m lm_eval \\")
        print(f"    --model local-completions \\")
        print(f"    --model_args '{eval_model_args}' \\")
        print(f"    --tasks {self.config.eval_tasks} \\")
        print(f"    --batch_size {self.config.eval_batch_size} \\")
        print(f"    --num_fewshot {self.config.eval_num_fewshot} \\")
        print(f"    --trust_remote_code" + (" \\" if self.config.eval_limit else ""))
        if self.config.eval_limit:
            print(f"    --limit {self.config.eval_limit}")
