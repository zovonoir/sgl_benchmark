"""vLLM benchmark runner for existing Docker containers."""

from __future__ import annotations

import json
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import ProfileCaseConfig, SuiteConfig, TestCaseConfig
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
            if self.config.run_mode == "chat":
                self._run_chat()
                return
            if self.config.run_mode == "longform":
                self._run_longform()
                return
            if self.config.run_mode == "multiturn":
                self._run_multiturn()
                return
            if self.config.run_mode == "profile":
                self._run_profile()
                return

            self._run_benchmark_cases()

            generate_summary(self.run_dir)
            print(f"\n>>> All cases finished")
            print(f">>> Summary report: {self.run_dir / 'suite_summary_report.txt'}")
        finally:
            self.container.cleanup_container()

    def _run_benchmark_cases(self) -> None:
        if self.config.restart_vllm_server:
            self._run_benchmark_with_server_restart()
        else:
            self._run_benchmark_reusing_server()

    def _run_benchmark_with_server_restart(self) -> None:
        print(">>> Restart vLLM server between benchmark cases: enabled")
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

    def _run_benchmark_reusing_server(self) -> None:
        print(">>> Restart vLLM server between benchmark cases: disabled")
        server_started = False
        try:
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
                    if not server_started:
                        self.cleanup_best_effort()
                        self._start_server(container_case_dir, result_stem)
                        server_started = True
                        self._wait_ready(container_case_dir)
                    else:
                        print(">>> Reusing existing vLLM server for this case")
                    bench_rc = self._run_benchmark(container_case_dir, result_stem, test_case)
                    elapsed = int(time.time() - started_at)
                    self._write_container_status(container_case_dir, bench_rc, elapsed)
                    self.container.copy_case_results(case_name, host_case_dir)
                    self._write_meta_and_aggregate(host_case_dir, case_name, result_stem, test_case)
                    if bench_rc != 0:
                        raise RuntimeError(f"Benchmark exited with code {bench_rc}")
                finally:
                    self._log_gpu_pids(container_case_dir, "gpu_pids_before_final_cleanup.log")
        finally:
            if server_started:
                self.cleanup_best_effort()

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

    def _run_profile(self) -> None:
        for idx, profile_case in enumerate(self.config.profile_configs, start=1):
            num_prompts = profile_case.num_prompts or profile_case.concurrency
            test_case = TestCaseConfig(
                concurrency=profile_case.concurrency,
                isl=profile_case.isl,
                osl=profile_case.osl,
                num_prompts=num_prompts,
            )
            case_name = self._profile_case_name(idx, profile_case, num_prompts)
            host_case_dir = self.run_dir / case_name
            container_case_dir = f"{self.container.output_path}/{case_name}"
            container_profile_dir = f"{container_case_dir}/traces"
            result_stem = self._result_stem(case_name, test_case)

            print(f"\n>>> [{idx}/{len(self.config.profile_configs)}] {case_name}")
            print(
                f">>> CONC={profile_case.concurrency} ISL={profile_case.isl} "
                f"OSL={profile_case.osl} NP={num_prompts}"
            )
            print(f">>> profile_with_stack={profile_case.profile_with_stack}")
            print(f">>> profile_record_shapes={profile_case.profile_record_shapes}")

            started_at = time.time()
            rc = 1
            try:
                self._prepare_case_dir(container_case_dir)
                self._prepare_profile_dir(container_profile_dir)
                self._log_gpu_pids(container_case_dir, "gpu_pids_before_cleanup.log")
                self.cleanup_best_effort()
                self._start_server(
                    container_case_dir,
                    result_stem,
                    environment=self._profile_environment(
                        container_profile_dir,
                        profile_case.profile_with_stack,
                    ),
                    extra_args=self._profile_server_args(
                        container_profile_dir,
                        profile_case.profile_with_stack,
                        profile_case.profile_record_shapes,
                    ),
                )
                self._wait_ready(container_case_dir)
                rc = self._run_benchmark(container_case_dir, result_stem, test_case, profile=True)
                if rc == 0:
                    self._wait_container_traces_stable(container_profile_dir)
                else:
                    print(">>> Skipping trace stability wait because profile benchmark failed")
                elapsed = int(time.time() - started_at)
                self._write_container_status(container_case_dir, rc, elapsed)
                self.container.copy_case_results(case_name, host_case_dir)
                self._write_meta_and_aggregate(host_case_dir, case_name, result_stem, test_case)
                trace_count = self._print_trace_files(host_case_dir)
                if rc != 0:
                    raise RuntimeError(f"Profile benchmark exited with code {rc}")
                if trace_count == 0:
                    raise RuntimeError("Profile completed but no trace files were generated")
            finally:
                self._log_gpu_pids(container_case_dir, "gpu_pids_before_final_cleanup.log")
                self.cleanup_best_effort()

        print(f"\n>>> All profile cases finished")
        print(f">>> Results directory: {self.run_dir}")

    def _run_chat(self) -> None:
        case_name = "chat"
        host_case_dir = self.run_dir / case_name
        container_case_dir = f"{self.container.output_path}/{case_name}"
        result_stem = f"chat_{sanitize(self.config.model_prefix)}_{self.config.precision}_{self.config.framework}_tp{self.config.tensor_parallel_size()}"
        started_at = time.time()
        try:
            self._prepare_case_dir(container_case_dir)
            self._log_gpu_pids(container_case_dir, "gpu_pids_before_cleanup.log")
            self.cleanup_best_effort()
            self._start_server(container_case_dir, result_stem)
            self._wait_ready(container_case_dir)
            prompt = self.config.chat_prompt
            if not prompt:
                try:
                    prompt = input("\n[You] ")
                except EOFError:
                    prompt = ""
            if not prompt:
                raise RuntimeError("chat_prompt is required in non-interactive runs")

            content, usage = self._chat_request(
                [{"role": "user", "content": prompt}],
                self.config.chat_max_tokens,
                self.config.chat_temperature,
            )
            chat_log = container_case_dir + "/chat_log.txt"
            payload = {
                "model": self.config.model_path,
                "date": datetime.now().strftime("%c"),
                "prompt": prompt,
                "response": content,
                "usage": usage,
            }
            self._write_json_and_text(container_case_dir, "chat_result.json", payload, "chat_log.txt")
            elapsed = int(time.time() - started_at)
            self._write_container_status(container_case_dir, 0, elapsed)
            self.container.copy_case_results(case_name, host_case_dir)
            print(content)
            print(f"\n>>> Chat log: {host_case_dir / 'chat_log.txt'}")
        finally:
            self._log_gpu_pids(container_case_dir, "gpu_pids_before_final_cleanup.log")
            self.cleanup_best_effort()

    def _run_longform(self) -> None:
        case_name = "longform"
        host_case_dir = self.run_dir / case_name
        container_case_dir = f"{self.container.output_path}/{case_name}"
        result_stem = f"longform_{sanitize(self.config.model_prefix)}_{self.config.precision}_{self.config.framework}_tp{self.config.tensor_parallel_size()}"
        started_at = time.time()
        try:
            self._prepare_case_dir(container_case_dir)
            self._log_gpu_pids(container_case_dir, "gpu_pids_before_cleanup.log")
            self.cleanup_best_effort()
            self._start_server(container_case_dir, result_stem)
            self._wait_ready(container_case_dir)
            results = []
            for idx, prompt in enumerate(self.config.longform_prompts, start=1):
                print(f">>> Longform {idx}/{len(self.config.longform_prompts)}: {prompt[:80]}...")
                content, usage = self._chat_request(
                    [{"role": "user", "content": prompt}],
                    self.config.longform_max_tokens,
                    self.config.chat_temperature,
                    timeout=900,
                )
                results.append({
                    "index": idx,
                    "prompt": prompt,
                    "response": content,
                    "usage": usage,
                })
            self._write_json_and_text(
                container_case_dir,
                "longform_results.json",
                {"model": self.config.model_path, "results": results},
                "longform_results.txt",
            )
            elapsed = int(time.time() - started_at)
            self._write_container_status(container_case_dir, 0, elapsed)
            self.container.copy_case_results(case_name, host_case_dir)
            print(f"\n>>> Longform results: {host_case_dir / 'longform_results.txt'}")
        finally:
            self._log_gpu_pids(container_case_dir, "gpu_pids_before_final_cleanup.log")
            self.cleanup_best_effort()

    def _run_multiturn(self) -> None:
        case_name = "multiturn"
        host_case_dir = self.run_dir / case_name
        container_case_dir = f"{self.container.output_path}/{case_name}"
        result_stem = f"multiturn_{sanitize(self.config.model_prefix)}_{self.config.precision}_{self.config.framework}_tp{self.config.tensor_parallel_size()}"
        started_at = time.time()
        try:
            self._prepare_case_dir(container_case_dir)
            self._log_gpu_pids(container_case_dir, "gpu_pids_before_cleanup.log")
            self.cleanup_best_effort()
            self._start_server(container_case_dir, result_stem)
            self._wait_ready(container_case_dir)
            turns = self._load_multiturn_turns()
            messages = []
            results = []
            for idx, turn in enumerate(turns, start=1):
                prompt = turn["user"]
                max_tokens = int(turn.get("max_tokens", self.config.multiturn_max_tokens))
                messages.append({"role": "user", "content": prompt})
                print(f">>> Turn {idx}/{len(turns)}: {prompt[:80]}...")
                content, usage = self._chat_request(
                    messages,
                    max_tokens,
                    self.config.chat_temperature,
                    timeout=900,
                )
                messages.append({"role": "assistant", "content": content})
                results.append({
                    "turn": idx,
                    "user": prompt,
                    "assistant": content,
                    "usage": usage,
                })
            self._write_json_and_text(
                container_case_dir,
                "multiturn_results.json",
                {"model": self.config.model_path, "turns": results},
                "accuracy_multiturn_multiturn.txt",
            )
            self._write_container_file(container_case_dir + "/_multiturn_turns.json", json.dumps(turns, ensure_ascii=False, indent=2))
            elapsed = int(time.time() - started_at)
            self._write_container_status(container_case_dir, 0, elapsed)
            self.container.copy_case_results(case_name, host_case_dir)
            print(f"\n>>> Multiturn results: {host_case_dir / 'accuracy_multiturn_multiturn.txt'}")
        finally:
            self._log_gpu_pids(container_case_dir, "gpu_pids_before_final_cleanup.log")
            self.cleanup_best_effort()

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

    def _load_multiturn_turns(self) -> list[dict]:
        if self.config.multiturn_turns:
            return [{"user": turn} for turn in self.config.multiturn_turns]
        if self.config.multiturn_turns_file:
            path = Path(self.config.multiturn_turns_file).expanduser()
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            normalized = []
            for item in data:
                if isinstance(item, str):
                    normalized.append({"user": item})
                else:
                    normalized.append(item)
            return normalized
        raise ValueError("No multiturn turns configured")

    def _chat_request(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        timeout: int = 600,
    ) -> tuple[str, dict]:
        body = {
            "model": self.config.model_path,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
        }
        script = r'''
import json
import os
import sys
import urllib.error
import urllib.request

url = os.environ["VLLM_BENCH_CHAT_URL"]
body = os.environ["VLLM_BENCH_CHAT_BODY"].encode("utf-8")
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=int(os.environ.get("VLLM_BENCH_CHAT_TIMEOUT", "600"))) as resp:
        sys.stdout.write(resp.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    sys.stderr.write(exc.read().decode("utf-8", "replace"))
    raise
'''
        code, output = self.container.exec_run(
            ["python3", "-c", script],
            environment={
                "VLLM_BENCH_CHAT_URL": f"http://localhost:{self.config.port}/v1/chat/completions",
                "VLLM_BENCH_CHAT_BODY": json.dumps(body, ensure_ascii=False),
                "VLLM_BENCH_CHAT_TIMEOUT": str(timeout),
            },
        )
        text = output.decode("utf-8", "replace")
        if code != 0:
            raise RuntimeError(f"chat request failed: {text}")
        data = json.loads(text)
        message = data["choices"][0]["message"]
        content = self._assistant_message_text(message)
        return content, data.get("usage", {})

    @staticmethod
    def _assistant_message_text(message: dict) -> str:
        """Return a printable assistant response from OpenAI-compatible chat output."""

        content = message.get("content")
        if isinstance(content, str):
            return content
        if content is not None:
            return json.dumps(content, ensure_ascii=False)

        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content
        if reasoning_content is not None:
            return json.dumps(reasoning_content, ensure_ascii=False)

        return json.dumps(message, ensure_ascii=False)

    def _write_container_file(self, path: str, content: str) -> None:
        script = "import os; open(os.environ['OUT_PATH'], 'w', encoding='utf-8').write(os.environ['OUT_CONTENT'])"
        self.container.exec_run(
            ["python3", "-c", script],
            environment={"OUT_PATH": path, "OUT_CONTENT": content},
        )

    def _write_json_and_text(
        self,
        container_case_dir: str,
        json_name: str,
        payload: dict,
        text_name: str,
    ) -> None:
        json_text = json.dumps(payload, ensure_ascii=False, indent=2)
        lines = []
        lines.append(f"Model: {self.config.model_path}")
        lines.append(f"Date: {datetime.now().strftime('%c')}")
        if "prompt" in payload:
            lines.extend(["", "[Prompt]", payload["prompt"], "", "[Response]", payload["response"], ""])
            lines.append(f"[Usage] {json.dumps(payload.get('usage', {}), ensure_ascii=False)}")
        elif "results" in payload:
            for item in payload["results"]:
                lines.extend([
                    "",
                    "=" * 64,
                    f"TEST {item['index']}",
                    "=" * 64,
                    "[Prompt]",
                    item["prompt"],
                    "",
                    "[Response]",
                    item["response"],
                    "",
                    f"[Usage] {json.dumps(item.get('usage', {}), ensure_ascii=False)}",
                ])
        elif "turns" in payload:
            for item in payload["turns"]:
                lines.extend([
                    "",
                    "=" * 64,
                    f"TURN {item['turn']} [User]",
                    "=" * 64,
                    item["user"],
                    "",
                    f"--- TURN {item['turn']} [Model] ---",
                    item["assistant"],
                    "",
                    f"[Usage] {json.dumps(item.get('usage', {}), ensure_ascii=False)}",
                ])
        text = "\n".join(lines) + "\n"
        self._write_container_file(container_case_dir + "/" + json_name, json_text)
        self._write_container_file(container_case_dir + "/" + text_name, text)

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

    def _start_server(
        self,
        container_case_dir: str,
        result_stem: str,
        environment: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        server_log = f"{container_case_dir}/server_{result_stem}.log"
        pid_file = f"{container_case_dir}/server.pid"
        cmd = self._server_command(extra_args=extra_args)
        shell = (
            f"cd {shlex.quote(container_case_dir)} && "
            f"setsid {' '.join(shlex.quote(part) for part in cmd)} "
            f">{shlex.quote(server_log)} 2>&1 & "
            f"echo $! > {shlex.quote(pid_file)}"
        )
        print(">>> Starting vLLM server:")
        print("    " + " ".join(shlex.quote(part) for part in cmd))
        if environment:
            print(">>> Server environment overrides:")
            for key, value in sorted(environment.items()):
                print(f"    {key}={value}")
        exit_code, output = self.container.exec_run(["bash", "-lc", shell], environment=environment)
        if exit_code != 0:
            raise RuntimeError(output.decode("utf-8", "ignore"))
        self._start_server_log_tail(server_log, pid_file)

    def _start_server_log_tail(self, server_log: str, pid_file: str) -> None:
        """Stream vLLM server logs while the server process is alive."""

        def _tail() -> None:
            tail_cmd = (
                f"while [ ! -f {shlex.quote(pid_file)} ]; do sleep 1; done; "
                f"pid=$(cat {shlex.quote(pid_file)}); "
                f"while [ ! -f {shlex.quote(server_log)} ]; do "
                f"kill -0 \"$pid\" 2>/dev/null || exit 0; sleep 1; "
                f"done; "
                f"tail --pid=\"$pid\" -n +1 -F {shlex.quote(server_log)}"
            )
            try:
                _, output_stream = self.container.exec_run(
                    ["bash", "-lc", tail_cmd],
                    stream=True,
                )
                for chunk in output_stream:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            except Exception:
                pass

        print(f">>> Streaming vLLM server log: {server_log}")
        threading.Thread(target=_tail, daemon=True).start()

    def _server_command(self, extra_args: list[str] | None = None) -> list[str]:
        cmd = ["vllm", "serve", self.config.model_path]
        if not self._server_args_have("--port"):
            cmd.extend(["--port", str(self.config.port)])
        for arg in self.config.server_args:
            cmd.extend(shlex.split(arg))
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def _server_args_have(self, option: str) -> bool:
        for arg in self.config.server_args:
            for part in shlex.split(arg):
                if part == option or part.startswith(option + "="):
                    return True
        return False

    def _wait_ready(self, container_case_dir: str) -> None:
        pid_file = f"{container_case_dir}/server.pid"
        timeout_seconds = self.config.health_timeout * 5
        deadline = time.monotonic() + timeout_seconds
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
                elapsed = int(timeout_seconds - (deadline - now))
                print(f">>> Still waiting for vLLM server... {elapsed}s elapsed")
                last_progress = now
            time.sleep(5)
        raise TimeoutError(
            f"vLLM server did not become ready within {timeout_seconds}s "
            f"({self.config.health_timeout} polls x 5s)"
        )

    def _run_benchmark(
        self,
        container_case_dir: str,
        result_stem: str,
        test_case: TestCaseConfig,
        profile: bool = False,
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
        if profile:
            cmd.append("--profile")

        print(">>> Running profile benchmark:" if profile else ">>> Running benchmark:")
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

    @staticmethod
    def _profile_environment(profile_dir: str, profile_with_stack: bool) -> dict[str, str]:
        return {
            "VLLM_TORCH_PROFILER_DIR": profile_dir,
            "VLLM_TORCH_PROFILER_WITH_STACK": "1" if profile_with_stack else "0",
            "VLLM_RPC_TIMEOUT": "1800000",
        }

    @staticmethod
    def _profile_server_args(
        profile_dir: str,
        profile_with_stack: bool,
        profile_record_shapes: bool,
    ) -> list[str]:
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": profile_dir,
            "torch_profiler_with_stack": profile_with_stack,
            "torch_profiler_record_shapes": profile_record_shapes,
            "torch_profiler_use_gzip": True,
        }
        return ["--profiler-config", json.dumps(profiler_config, ensure_ascii=False)]

    def _prepare_profile_dir(self, container_profile_dir: str) -> None:
        self.container.exec_run(["bash", "-lc", f"mkdir -p {shlex.quote(container_profile_dir)}"])

    def _wait_container_traces_stable(
        self,
        container_profile_dir: str,
        interval: int = 10,
        max_wait: int = 300,
    ) -> None:
        print(">>> Waiting for vLLM trace files to finish writing...")
        prev_sizes: dict[str, int] = {}
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(interval)
            elapsed += interval
            current_sizes = self._container_trace_sizes(container_profile_dir)
            if not current_sizes:
                continue
            if current_sizes == prev_sizes:
                print(f">>> Trace files stable ({len(current_sizes)} files)")
                return
            prev_sizes = current_sizes
            if elapsed % 30 == 0:
                total_mb = sum(current_sizes.values()) / (1024 * 1024)
                print(f">>> Still writing traces... ({total_mb:.1f} MB total, {elapsed}s elapsed)")
        print(">>> Warning: trace file wait timeout, files may still be writing")

    def _container_trace_sizes(self, container_profile_dir: str) -> dict[str, int]:
        script = r'''
import json
import os

root = os.environ["PROFILE_DIR"]
sizes = {}
if os.path.isdir(root):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".trace.json.gz"):
                path = os.path.join(dirpath, filename)
                try:
                    sizes[path] = os.path.getsize(path)
                except OSError:
                    pass
print(json.dumps(sizes))
'''
        code, output = self.container.exec_run(
            ["python3", "-c", script],
            environment={"PROFILE_DIR": container_profile_dir},
        )
        if code != 0:
            return {}
        try:
            return {str(key): int(value) for key, value in json.loads(output.decode("utf-8")).items()}
        except (json.JSONDecodeError, ValueError):
            return {}

    @staticmethod
    def _print_trace_files(host_case_dir: Path) -> int:
        trace_files = sorted(host_case_dir.rglob("*.trace.json.gz"))
        if not trace_files:
            print(f">>> No trace files found under {host_case_dir}")
            return 0
        print(f"\n>>> Trace files in {host_case_dir}:")
        for path in trace_files:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"    {path.relative_to(host_case_dir)} ({size_mb:.1f} MB)")
        return len(trace_files)

    def _case_name(self, idx: int, tc: TestCaseConfig) -> str:
        return f"case_{idx:02d}_conc{tc.concurrency}_isl{tc.isl}_osl{tc.osl}_np{tc.num_prompts}"

    def _profile_case_name(self, idx: int, case: ProfileCaseConfig, num_prompts: int) -> str:
        return (
            f"profile_{idx:02d}_conc{case.concurrency}"
            f"_isl{case.isl}_osl{case.osl}_np{num_prompts}"
        )

    def _result_stem(self, case_name: str, tc: TestCaseConfig) -> str:
        model = sanitize(self.config.model_prefix)
        return (
            f"{case_name}_{model}_{self.config.precision}_{self.config.framework}"
            f"_tp{self.config.tensor_parallel_size()}"
        )


def sanitize(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")

