"""Benchmark runner - performance stress testing.

Orchestrates container lifecycle and invokes run_case.sh inside the container
for each test case, then generates a summary report.
"""

import sys
from pathlib import Path

from ..config import SuiteConfig, TestCaseConfig
from ..report import generate_summary
from .base import BaseRunner


class BenchmarkRunner(BaseRunner):
    """Runs performance benchmarks by invoking run_case.sh inside Docker."""

    def execute(self) -> None:
        for idx, test_case in enumerate(self.config.test_configs):
            case_id = f"{idx + 1:02d}"
            case_name = self._build_case_name(case_id, test_case)
            case_host_dir = self.run_dir / case_name
            case_host_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n>>> [{case_id}/{len(self.config.test_configs)}] {case_name}")
            print(f">>> CONC={test_case.concurrency} ISL={test_case.isl} "
                  f"OSL={test_case.osl} NP={test_case.num_prompts}")

            # Each test case gets a fresh container (create mode)
            # or reuses the existing one (attach mode)
            if not self.container.attach_mode:
                self.container.cleanup()
                self.container.start()
            else:
                # Attach mode: connect if not already connected
                if self.container._container is None:
                    self.container.start()
            self.container.run_post_start_commands()

            env = self._build_case_env(case_name, test_case)

            # Ensure output directory exists inside container
            output_dir = env["CASE_OUTPUT_DIR"]
            self.container.exec_run(["mkdir", "-p", output_dir])

            # Execute run_case.sh inside container, streaming output
            suite_path = self.config.suite_path_in_container
            exec_id, output_stream = self.container.exec_run(
                cmd=["bash", f"{suite_path}/run_case.sh"],
                environment=env,
                workdir=suite_path,
                stream=True,
            )

            for chunk in output_stream:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

            # Check exit code (run_case.sh may return non-zero due to server
            # cleanup killing the background server process, which is expected)
            exit_info = self.container._client.api.exec_inspect(exec_id)
            exit_code = exit_info.get("ExitCode", -1)
            if exit_code != 0:
                print(f"\n>>> Warning: Case {case_name} exited with code {exit_code} "
                      "(may be caused by server cleanup)")

            # In attach mode, copy results from container to host run_dir
            if self.container.attach_mode:
                self._copy_results_from_container(case_name)

            self.container.cleanup()

        # Generate summary report
        generate_summary(self.run_dir)

        print(f"\n>>> All cases finished")
        print(f">>> Summary report: {self.run_dir}/suite_summary_report.txt")

    def dry_run(self) -> None:
        print(f"\n--- Benchmark Plan ---")
        print(f"Backend: {self.config.bench_backend}")
        print(f"Random range ratio: {self.config.random_range_ratio}")
        print(f"Request rate: {self.config.request_rate}")
        print(f"Burstiness: {self.config.burstiness}")
        print(f"Test cases: {len(self.config.test_configs)}")

        for idx, tc in enumerate(self.config.test_configs):
            case_id = f"{idx + 1:02d}"
            case_name = self._build_case_name(case_id, tc)
            print(f"\n  Case {case_id}: {case_name}")
            print(f"    Concurrency: {tc.concurrency}")
            print(f"    Input length (ISL): {tc.isl}")
            print(f"    Output length (OSL): {tc.osl}")
            print(f"    Num prompts: {tc.num_prompts}")

            if self.config.bench_backend == "sglang":
                print(f"    Tool: python3 -m sglang.bench_serving --backend sglang")
                print(f"    Endpoint: /generate (SGLang native)")
            else:
                print(f"    Tool: python3 benchmark_serving.py --backend vllm")
                print(f"    Endpoint: /v1/completions (OpenAI compat)")

    def _copy_results_from_container(self, case_name: str) -> None:
        """Copy benchmark results from container to host run_dir (attach mode only)."""
        import subprocess
        container_path = f"/simple-suite-output/{case_name}"
        host_path = str(self.run_dir / case_name)
        try:
            subprocess.run(
                ["docker", "cp",
                 f"{self.container.container_name}:{container_path}/.",
                 host_path],
                check=True, capture_output=True,
            )
            print(f">>> Copied results from container to {host_path}")
        except subprocess.CalledProcessError as e:
            print(f">>> Warning: Failed to copy results from container: {e.stderr.decode()}")

    def _build_case_name(self, case_id: str, tc: TestCaseConfig) -> str:
        return (f"case_{case_id}_conc{tc.concurrency}_isl{tc.isl}"
                f"_osl{tc.osl}_np{tc.num_prompts}")

    def _build_case_env(self, case_name: str, tc: TestCaseConfig) -> dict:
        """Build the complete environment for docker exec run_case.sh."""
        env = {
            "CASE_NAME": case_name,
            "CASE_OUTPUT_DIR": f"/simple-suite-output/{case_name}",
            "MODEL_PATH": self.config.model_path,
            "MODEL_PREFIX": self.config.model_prefix,
            "IMAGE": self.config.image or self.config.existing_container or "",
            "PRECISION": self.config.precision,
            "RUNNER_TYPE": self.config.runner_type,
            "FRAMEWORK": self.config.framework,
            "RANDOM_RANGE_RATIO": str(self.config.random_range_ratio),
            "REQUEST_RATE": self.config.request_rate,
            "BURSTINESS": str(self.config.burstiness),
            "PORT": str(self.config.port),
            "CONC": str(tc.concurrency),
            "ISL": str(tc.isl),
            "OSL": str(tc.osl),
            "NUM_PROMPTS": str(tc.num_prompts),
            # Expand "--key value" strings into separate items, then serialize with \x1e
            "SERVER_ARGS_SERIALIZED": "\x1e".join(
                part for arg in self.config.server_args for part in arg.split()
            ),
            "BENCH_BACKEND": self.config.bench_backend,
            "WATCHDOG_TIMEOUT": str(self.config.watchdog_timeout),
        }

        # container_env_overrides already injected at docker run time,
        # only pass run_case.sh specific vars here

        return env
