"""Profile runner - collect Torch Profiler traces under online serving load.

Uses sglang.bench_serving --profile to capture GPU kernel and Python call
stack traces at high concurrency. Server runs with --disable-cuda-graph
to preserve fine-grained kernel events (configurable).

Output: .trace.json.gz files in runs/run_<timestamp>/profile_<case>/
"""

import shlex
import sys
import time
from pathlib import Path

from ..config import ProfileCaseConfig
from .base import BaseRunner


# Container-internal path for profiler output.
# This maps to run_dir/profile_<case> on the host via the /simple-suite-output mount.
_CONTAINER_PROFILE_BASE = "/simple-suite-output"


class ProfileRunner(BaseRunner):
    """Collects Torch Profiler traces via online serving benchmark."""

    def execute(self) -> None:
        for idx, case in enumerate(self.config.profile_configs):
            case_id = f"{idx + 1:02d}"
            num_prompts = case.num_prompts or case.concurrency
            case_name = (f"profile_{case_id}_conc{case.concurrency}"
                         f"_isl{case.isl}_osl{case.osl}_np{num_prompts}")
            case_dir = self.run_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n>>> [{case_id}/{len(self.config.profile_configs)}] {case_name}")
            print(f">>> CONC={case.concurrency} ISL={case.isl} OSL={case.osl} "
                  f"NP={num_prompts}")
            print(f">>> disable_cuda_graph={case.disable_cuda_graph} "
                  f"skip_server_warmup={case.skip_server_warmup} "
                  f"profile_with_stack={case.profile_with_stack}")

            # Each case gets a fresh container
            self.container.cleanup()

            # Inject profiler env vars
            container_profile_dir = f"{_CONTAINER_PROFILE_BASE}/{case_name}"
            extra_env = {
                "SGLANG_TORCH_PROFILER_DIR": container_profile_dir,
                "SGLANG_PROFILE_WITH_STACK": str(case.profile_with_stack),
            }
            self.container.start(extra_env=extra_env)
            self.container.run_post_start_commands()

            # Build extra server args based on profile case config
            server_extra_args = []
            if case.disable_cuda_graph:
                server_extra_args.append("--disable-cuda-graph")

            # Start server
            self.server.start(
                skip_warmup=case.skip_server_warmup,
                extra_args=server_extra_args if server_extra_args else None,
            )
            self.server.wait_healthy()

            # Build and execute bench_serving profile command
            bench_cmd = self._build_bench_cmd(case, num_prompts, case_name,
                                              container_profile_dir)
            print(f">>> Running: {bench_cmd}")

            exec_id, output_stream = self.container.exec_run(
                cmd=["bash", "-c", bench_cmd],
                stream=True,
            )
            for chunk in output_stream:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

            # Wait for trace files to finish writing (gzip compression can lag)
            self._wait_traces_stable(case_dir)

            # List generated trace files
            print(f"\n>>> Trace files in {case_dir}:")
            for f in sorted(case_dir.rglob("*.trace.json.gz")):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    {f.name} ({size_mb:.1f} MB)")

            self.container.cleanup()

        print(f"\n>>> All profile cases finished")
        print(f">>> Results directory: {self.run_dir}")

    def dry_run(self) -> None:
        print(f"\n--- Profile Plan ---")
        print(f"Cases: {len(self.config.profile_configs)}")

        for idx, case in enumerate(self.config.profile_configs):
            case_id = f"{idx + 1:02d}"
            num_prompts = case.num_prompts or case.concurrency
            case_name = (f"profile_{case_id}_conc{case.concurrency}"
                         f"_isl{case.isl}_osl{case.osl}_np{num_prompts}")

            print(f"\n  Case {case_id}: {case_name}")
            print(f"    Concurrency: {case.concurrency}")
            print(f"    Input length (ISL): {case.isl}")
            print(f"    Output length (OSL): {case.osl}")
            print(f"    Num prompts: {num_prompts}")
            print(f"    disable_cuda_graph: {case.disable_cuda_graph}")
            print(f"    skip_server_warmup: {case.skip_server_warmup}")
            print(f"    profile_with_stack: {case.profile_with_stack}")

            # Show server command with profile-specific args
            server_extra = []
            if case.disable_cuda_graph:
                server_extra.append("--disable-cuda-graph")
            server_cmd = self.server.build_server_cmd(
                skip_warmup=case.skip_server_warmup,
                extra_args=server_extra if server_extra else None,
            )
            print(f"    Server command: {' '.join(server_cmd)}")

            # Show bench_serving command
            container_profile_dir = f"{_CONTAINER_PROFILE_BASE}/{case_name}"
            bench_cmd = self._build_bench_cmd(case, num_prompts, case_name,
                                              container_profile_dir)
            print(f"    Bench command: {bench_cmd}")

            print(f"    SGLANG_TORCH_PROFILER_DIR: {container_profile_dir}")
            print(f"    SGLANG_PROFILE_WITH_STACK: {case.profile_with_stack}")
            print(f"    Output: {self.run_dir}/{case_name}/")

    def _build_bench_cmd(self, case: ProfileCaseConfig, num_prompts: int,
                         case_name: str, profile_dir: str) -> str:
        """Build the sglang.bench_serving command for profiling."""
        prefix = (f"{self.config.model_prefix}_conc{case.concurrency}"
                  f"_isl{case.isl}_osl{case.osl}_np{num_prompts}")

        parts = [
            "python3", "-m", "sglang.bench_serving",
            "--backend", "sglang",
            "--host", "0.0.0.0",
            "--port", str(self.config.port),
            "--model", self.config.model_path,
            "--dataset-name", "random",
            "--random-input-len", str(case.isl),
            "--random-output-len", str(case.osl),
            "--random-range-ratio", "1",
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(case.concurrency),
            "--request-rate", "inf",
            "--flush-cache",
            "--profile",
            "--profile-by-stage",
            "--profile-output-dir", profile_dir,
            "--profile-prefix", prefix,
        ]
        return " ".join(shlex.quote(p) for p in parts)

    def _wait_traces_stable(self, case_dir: Path, interval: int = 10,
                            max_wait: int = 300) -> None:
        """Wait until trace files stop growing (gzip flush completed)."""
        print(">>> Waiting for trace files to finish writing...")
        prev_sizes = {}
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(interval)
            elapsed += interval

            current_sizes = {}
            for f in case_dir.rglob("*.trace.json.gz"):
                current_sizes[str(f)] = f.stat().st_size

            if not current_sizes:
                # No trace files yet, keep waiting
                continue

            if current_sizes == prev_sizes:
                print(f">>> Trace files stable ({len(current_sizes)} files)")
                return

            prev_sizes = current_sizes
            if elapsed % 30 == 0:
                total_mb = sum(current_sizes.values()) / (1024 * 1024)
                print(f">>> Still writing traces... ({total_mb:.1f} MB total, {elapsed}s elapsed)")

        print(">>> Warning: trace file wait timeout, files may still be writing")
