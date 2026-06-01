"""vLLM benchmark runner for existing Docker containers."""

from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path

from .config import SuiteConfig, TestCaseConfig
from .container import ExistingContainerManager
from .report import aggregate_case, generate_summary


class VllmBenchmarkRunner:
    """Run vLLM serving benchmarks inside an existing container."""

    def __init__(
        self,
        config: SuiteConfig,
        container: ExistingContainerManager,
        run_dir: Path,
    ):
        self.config = config
        self.container = container
        self.run_dir = run_dir

    def run(self) -> None:
        self.container.attach()
        try:
            self.container.inject_suite()
            self.container.run_commands(self.config.post_start_commands, "post-start")
            self._ensure_output_root()

            if self.config.run_mode == "eval":
                self._run_eval()
                return

            for idx, test_case in enumerate(self.config.test_configs, start=1):
                case_name = self._case_name(idx, test_case)
                host_case_dir = self.run_dir / case_name
                container_case_dir = f"{self.container.output_path}/{case_name}"
                result_stem = self._result_stem(case_name, test_case)

                print(f"\n>>> [{idx}/{len(self.config.test_configs)}] {case_name}")
                print(
                    f">>> CONC={test_case.concurrency} ISL={test_case.isl} "
                    f"OSL={test_case.osl} NP={test_case.num_prompts}"
                )

                started_at = time.time()
                try:
                    self._prepare_case_dir(container_case_dir)
                    self._log_gpu_pids(container_case_dir, "gpu_pids_before_cleanup.log")
                    self.cleanup_best_effort()
                    self._start_server(container_case_dir, result_stem)
                    self._wait_ready(container_case_dir)
                    bench_rc = self._run_benchmark(container_case_dir, result_stem, test_case)
                    elapsed = int(time.time() - started_at)
                    self._write_container_status(container_case_dir, bench_rc, elapsed)
                    self.container.copy_case_results(case_name, host_case_dir)
                    self._write_meta_and_aggregate(host_case_dir, case_name, result_stem, test_case)
                    if bench_rc != 0:
                        raise RuntimeError(f"Benchmark exited with code {bench_rc}")
                finally:
                    self._log_gpu_pids(container_case_dir, "gpu_pids_before_final_cleanup.log")
                    self.cleanup_best_effort()

            generate_summary(self.run_dir)
            print(f"\n>>> All cases finished")
            print(f">>> Summary report: {self.run_dir / 'suite_summary_report.txt'}")
        finally:
            self.container.cleanup_container()

    def _run_eval(self) -> None:
        safe_tasks = sanitize(self.config.eval_tasks.replace(",", "_"))
        case_name = f"eval_{safe_tasks}_fewshot{self.config.eval_num_fewshot}"
        host_case_dir = self.run_dir / case_name
        container_case_dir = f"{self.container.output_path}/{case_name}"
        result_stem = f"{case_name}_{sanitize(self.config.model_prefix)}_{self.config.precision}_{self.config.framework}_tp{self.config.tensor_parallel_size()}"

        print(f"\n>>> Eval: {case_name}")
        started_at = time.time()
        rc = 1
        try:
            self._prepare_case_dir(container_case_dir)
            self._log_gpu_pids(container_case_dir, "gpu_pids_before_cleanup.log")
            self.cleanup_best_effort()
            self._check_eval_dependencies()
            self._start_server(container_case_dir, result_stem)
            self._wait_ready(container_case_dir)
            rc = self._run_lm_eval(container_case_dir, result_stem)
            elapsed = int(time.time() - started_at)
            self._write_container_status(container_case_dir, rc, elapsed)
            self.container.copy_case_results(case_name, host_case_dir)
            self._write_eval_meta(host_case_dir, case_name, result_stem)
            self._write_eval_summary(host_case_dir, result_stem, rc, elapsed)
            if rc != 0:
                raise RuntimeError(f"Eval exited with code {rc}")
        finally:
            self._log_gpu_pids(container_case_dir, "gpu_pids_before_final_cleanup.log")
            self.cleanup_best_effort()

        print(f"\n>>> Eval finished")
        print(f">>> Eval summary: {host_case_dir / 'eval_summary.txt'}")

    def _check_eval_dependencies(self) -> None:
        code, output = self.container.exec_run([
            "bash",
            "-lc",
            "command -v lm_eval >/dev/null 2>&1",
        ])
        if code != 0:
            raise RuntimeError(
                "lm_eval is not installed in the target container. "
                "Install it first, e.g. `pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git tenacity`."
            )

    def cleanup_best_effort(self) -> None:
        """Terminate vLLM processes that match configured cleanup patterns."""

        if self.container._container is None:
            return
        patterns_json = json.dumps(self.config.cleanup_patterns)
        script = f"""
import json
import os
import signal
import time

patterns = json.loads({patterns_json!r})
own = os.getpid()

def matches(pid):
    if pid == own:
        return False
    try:
        with open(f"/proc/{{pid}}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\\x00", b" ").decode("utf-8", "ignore")
    except OSError:
        return False
    return any(pattern in cmd for pattern in patterns)

pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
targets = [pid for pid in pids if matches(pid)]
for pid in targets:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
if targets:
    time.sleep(5)
for pid in targets:
    if matches(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
print("cleanup_targets=" + ",".join(map(str, targets)))
"""
        self.container.exec_run(["python3", "-c", script])

    def _ensure_output_root(self) -> None:
        self.container.exec_run(["bash", "-lc", f"mkdir -p {shlex.quote(self.container.output_path)}"])

    def _prepare_case_dir(self, container_case_dir: str) -> None:
        self.container.exec_run([
            "bash",
            "-lc",
            f"rm -rf {shlex.quote(container_case_dir)} && mkdir -p {shlex.quote(container_case_dir)}",
        ])

    def _start_server(self, container_case_dir: str, result_stem: str) -> None:
        server_log = f"{container_case_dir}/server_{result_stem}.log"
        pid_file = f"{container_case_dir}/server.pid"
        cmd = self._server_command()
        shell = (
            f"cd {shlex.quote(container_case_dir)} && "
            f"setsid {' '.join(shlex.quote(part) for part in cmd)} "
            f">{shlex.quote(server_log)} 2>&1 & "
            f"echo $! > {shlex.quote(pid_file)}"
        )
        print(">>> Starting vLLM server:")
        print("    " + " ".join(shlex.quote(part) for part in cmd))
        exit_code, output = self.container.exec_run(["bash", "-lc", shell])
        if exit_code != 0:
            raise RuntimeError(output.decode("utf-8", "ignore"))

    def _server_command(self) -> list[str]:
        cmd = ["vllm", "serve", self.config.model_path]
        if not self._server_args_have("--port"):
            cmd.extend(["--port", str(self.config.port)])
        for arg in self.config.server_args:
            cmd.extend(shlex.split(arg))
        return cmd

    def _server_args_have(self, option: str) -> bool:
        for arg in self.config.server_args:
            for part in shlex.split(arg):
                if part == option or part.startswith(option + "="):
                    return True
        return False

    def _wait_ready(self, container_case_dir: str) -> None:
        pid_file = f"{container_case_dir}/server.pid"
        deadline = time.monotonic() + self.config.health_timeout
        url = f"http://localhost:{self.config.port}/v1/models"
        print(f">>> Waiting for vLLM server: {url}")
        last_progress = 0.0
        while time.monotonic() < deadline:
            alive_code, _ = self.container.exec_run([
                "bash",
                "-lc",
                f"pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
                f"test -n \"$pid\" && kill -0 \"$pid\"",
            ])
            if alive_code != 0:
                raise RuntimeError("vLLM server exited before becoming ready")

            ready_code, _ = self.container.exec_run([
                "bash",
                "-lc",
                f"curl -fsS {shlex.quote(url)} >/dev/null 2>&1",
            ])
            if ready_code == 0:
                print(">>> vLLM server is ready")
                return

            now = time.monotonic()
            if now - last_progress > 60:
                elapsed = int(self.config.health_timeout - (deadline - now))
                print(f">>> Still waiting for vLLM server... {elapsed}s elapsed")
                last_progress = now
            time.sleep(5)
        raise TimeoutError(f"vLLM server did not become ready within {self.config.health_timeout}s")

    def _run_benchmark(
        self,
        container_case_dir: str,
        result_stem: str,
        test_case: TestCaseConfig,
    ) -> int:
        run_log = f"{container_case_dir}/run_{result_stem}.log"
        cmd = [
            "python3",
            f"{self.config.suite_path_in_container}/sgl_bench/utils/bench_serving/benchmark_serving.py",
            "--backend",
            "vllm",
            "--base-url",
            f"http://localhost:{self.config.port}",
            "--endpoint",
            "/v1/completions",
            "--model",
            self.config.model_path,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(test_case.isl),
            "--random-output-len",
            str(test_case.osl),
            "--random-range-ratio",
            str(self.config.random_range_ratio),
            "--num-prompts",
            str(test_case.num_prompts),
            "--num-warmups",
            str(self.config.num_warmups),
            "--request-rate",
            self.config.request_rate,
            "--max-concurrency",
            str(test_case.concurrency),
            "--burstiness",
            str(self.config.burstiness),
            "--trust-remote-code",
            "--save-result",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--result-dir",
            container_case_dir,
            "--result-filename",
            f"{result_stem}.json",
        ]
        if self.config.benchmark_ignore_eos:
            cmd.append("--ignore-eos")
        if self.config.benchmark_temperature is not None:
            cmd.extend(["--temperature", str(self.config.benchmark_temperature)])
        if self.config.benchmark_extra_request_body:
            cmd.extend([
                "--extra-request-body",
                json.dumps(self.config.benchmark_extra_request_body, ensure_ascii=False),
            ])

        print(">>> Running benchmark:")
        print("    " + " ".join(shlex.quote(part) for part in cmd))
        shell = (
            "set -o pipefail; "
            + " ".join(shlex.quote(part) for part in cmd)
            + f" 2>&1 | tee {shlex.quote(run_log)}"
        )
        exec_id, output_stream = self.container.exec_run(
            ["bash", "-lc", shell],
            stream=True,
        )
        for chunk in output_stream:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        exit_info = self.container.inspect_exec(exec_id)
        return int(exit_info.get("ExitCode", -1))

    def _run_lm_eval(self, container_case_dir: str, result_stem: str) -> int:
        log_path = f"{container_case_dir}/eval_{result_stem}.log"
        output_path = f"{container_case_dir}/results_{sanitize(self.config.eval_tasks)}_fewshot{self.config.eval_num_fewshot}"
        model_args = ",".join([
            f"model={self.config.model_path}",
            f"base_url=http://localhost:{self.config.port}/v1/completions",
            f"num_concurrent={self.config.eval_num_concurrent}",
            f"max_retries={self.config.eval_max_retries}",
            f"max_gen_toks={self.config.eval_max_gen_toks}",
            f"max_length={self.config.eval_max_length}",
            f"timeout={self.config.eval_timeout}",
        ])
        cmd = [
            "lm_eval",
            "--model",
            "local-completions",
            "--model_args",
            model_args,
            "--batch_size",
            self.config.eval_batch_size,
            "--tasks",
            self.config.eval_tasks,
            "--num_fewshot",
            str(self.config.eval_num_fewshot),
            "--output_path",
            output_path,
        ]
        if self.config.eval_limit is not None:
            cmd.extend(["--limit", str(self.config.eval_limit)])
        if self.config.eval_log_samples:
            cmd.append("--log_samples")

        print(">>> Running lm_eval:")
        print("    " + " ".join(shlex.quote(part) for part in cmd))
        shell = (
            "set -o pipefail; "
            + " ".join(shlex.quote(part) for part in cmd)
            + f" 2>&1 | tee {shlex.quote(log_path)}"
        )
        exec_id, output_stream = self.container.exec_run(["bash", "-lc", shell], stream=True)
        for chunk in output_stream:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        exit_info = self.container.inspect_exec(exec_id)
        return int(exit_info.get("ExitCode", -1))

    def _write_container_status(self, container_case_dir: str, rc: int, elapsed: int) -> None:
        status = f"rc={rc} duration_s={elapsed}\n"
        self.container.exec_run([
            "bash",
            "-lc",
            f"printf %s {shlex.quote(status)} > {shlex.quote(container_case_dir + '/status.txt')}",
        ])

    def _write_meta_and_aggregate(
        self,
        host_case_dir: Path,
        case_name: str,
        result_stem: str,
        test_case: TestCaseConfig,
    ) -> None:
        meta = {
            "case_name": case_name,
            "model_path": self.config.model_path,
            "model_prefix": self.config.model_prefix,
            "precision": self.config.precision,
            "framework": self.config.framework,
            "runner_type": self.config.runner_type,
            "concurrency": test_case.concurrency,
            "isl": test_case.isl,
            "osl": test_case.osl,
            "num_prompts": test_case.num_prompts,
            "tp": self.config.tensor_parallel_size(),
            "port": self.config.port,
            "request_rate": self.config.request_rate,
            "burstiness": self.config.burstiness,
            "random_range_ratio": self.config.random_range_ratio,
            "num_warmups": self.config.num_warmups,
            "server_args": self.config.server_args,
            "container_env": self.config.container_environment(),
            "result_filename": result_stem,
        }
        meta_path = host_case_dir / f"meta_{result_stem}.json"
        raw_path = host_case_dir / f"{result_stem}.json"
        agg_path = host_case_dir / f"agg_{result_stem}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw benchmark result: {raw_path}")
        aggregate_case(raw_path, meta_path, agg_path)

    def _write_eval_meta(self, host_case_dir: Path, case_name: str, result_stem: str) -> None:
        meta = {
            "case_name": case_name,
            "model_path": self.config.model_path,
            "model_prefix": self.config.model_prefix,
            "precision": self.config.precision,
            "framework": self.config.framework,
            "runner_type": self.config.runner_type,
            "tp": self.config.tensor_parallel_size(),
            "port": self.config.port,
            "eval_tasks": self.config.eval_tasks,
            "eval_num_fewshot": self.config.eval_num_fewshot,
            "eval_batch_size": self.config.eval_batch_size,
            "eval_limit": self.config.eval_limit,
            "eval_num_concurrent": self.config.eval_num_concurrent,
            "eval_max_retries": self.config.eval_max_retries,
            "eval_max_gen_toks": self.config.eval_max_gen_toks,
            "eval_max_length": self.config.eval_max_length,
            "eval_timeout": self.config.eval_timeout,
            "server_args": self.config.server_args,
            "container_env": self.config.container_environment(),
            "result_filename": result_stem,
        }
        meta_path = host_case_dir / f"meta_{result_stem}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _write_eval_summary(self, host_case_dir: Path, result_stem: str, rc: int, elapsed: int) -> None:
        log_path = host_case_dir / f"eval_{result_stem}.log"
        lines = []
        if log_path.is_file():
            content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            table_lines = [
                line for line in content
                if line.startswith("|") or "exact_match" in line or "Selected Tasks" in line
            ]
            lines.extend(table_lines[-40:])
        summary = [
            f"Eval directory : {host_case_dir}",
            f"Model          : {self.config.model_path}",
            f"Tasks          : {self.config.eval_tasks}",
            f"Fewshot        : {self.config.eval_num_fewshot}",
            f"Return code    : {rc}",
            f"Duration (s)   : {elapsed}",
            "",
            *lines,
        ]
        summary_path = host_case_dir / "eval_summary.txt"
        summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
        print("\n".join(summary))

    def _log_gpu_pids(self, container_case_dir: str, filename: str) -> None:
        if self.container._container is None:
            return
        self.container.exec_run([
            "bash",
            "-lc",
            (
                f"mkdir -p {shlex.quote(container_case_dir)}; "
                f"(rocm-smi --showpids || true) > "
                f"{shlex.quote(container_case_dir + '/' + filename)} 2>&1"
            ),
        ])

    def _case_name(self, idx: int, tc: TestCaseConfig) -> str:
        return f"case_{idx:02d}_conc{tc.concurrency}_isl{tc.isl}_osl{tc.osl}_np{tc.num_prompts}"

    def _result_stem(self, case_name: str, tc: TestCaseConfig) -> str:
        model = sanitize(self.config.model_prefix)
        return (
            f"{case_name}_{model}_{self.config.precision}_{self.config.framework}"
            f"_tp{self.config.tensor_parallel_size()}"
        )


def sanitize(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")

