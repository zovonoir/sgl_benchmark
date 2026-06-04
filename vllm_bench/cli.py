"""CLI entry point for vLLM benchmark runs."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

from .config import SuiteConfig, load_config


def _print_banner(config: SuiteConfig, run_dir: Path) -> None:
    print("=" * 60)
    print("  vllm_bench")
    print("=" * 60)
    print(f"Run dir       : {run_dir}")
    print(f"Container     : {config.existing_container or config.container_name or '<auto>'}")
    if config.image:
        print(f"Image         : {config.image}")
    print(f"Model         : {config.model_path}")
    print(f"Run mode      : {config.run_mode}")
    print(f"Port          : {config.port}")
    print(f"TP            : {config.tensor_parallel_size()}")
    if config.run_mode == "benchmark":
        print(f"Test cases    : {len(config.test_configs)}")
    elif config.run_mode == "eval":
        print(f"Eval tasks    : {config.eval_tasks}")
    elif config.run_mode == "chat":
        print(f"Chat prompt   : {config.chat_prompt or '<interactive>'}")
    elif config.run_mode == "longform":
        print(f"Longform      : {len(config.longform_prompts)} prompts")
    elif config.run_mode == "multiturn":
        print(f"Multiturn     : {len(config.multiturn_turns)} inline turns")
    elif config.run_mode == "profile":
        print(f"Profile cases : {len(config.profile_configs)}")
    print("=" * 60)


def _print_dry_run(config: SuiteConfig, run_dir: Path, project_root: Path) -> None:
    print("\n" + "=" * 60)
    print("  DRY RUN - vLLM Benchmark")
    print("=" * 60)
    print(f"\n[1] Configuration: VALID")
    if config.existing_container:
        print(f"  Mode: attach")
        print(f"  Container: {config.existing_container}")
    else:
        print(f"  Mode: create")
        print(f"  Image: {config.image}")
        print(f"  Container name: {config.container_name or '<auto>'}")
        print(f"  Entrypoint: {config.entrypoint}")
        print(f"  Command: {config.command or ['-lc', 'while true; do sleep 3600; done']}")
    print(f"  Model: {config.model_path}")
    print(f"  Model prefix: {config.model_prefix}")
    print(f"  Run mode: {config.run_mode}")
    print(f"  Precision: {config.precision}")
    print(f"  Runner type: {config.runner_type}")
    print(f"  Framework: {config.framework}")
    print(f"  Port: {config.port}")
    print(f"  Health timeout: {config.health_timeout * 5}s ({config.health_timeout} polls x 5s)")
    print(f"  TP: {config.tensor_parallel_size()}")

    print(f"\n[2] Suite Injection:")
    print(f"  Host project root: {project_root}")
    print(f"  Container suite path: {config.suite_path_in_container}")
    print("  Mode: docker archive copy into existing container")
    if config.image:
        print("  Model mounts:")
        print(f"    {config.host_model_mount_path} -> /.cache/huggingface/")
        print(f"    {config.host_model_mount_path} -> {config.host_model_mount_path}")
        print(f"  Docker run args: {config.docker_run_args}")
        print(f"  Extra mounts: {config.extra_container_mounts}")
    if config.post_start_commands:
        print(f"  Post-start commands: {len(config.post_start_commands)}")

    print(f"\n[3] Environment ({len(config.container_environment())} vars):")
    for key, value in sorted(config.container_environment().items()):
        print(f"  {key}={value}")

    print("\n[4] vLLM Server:")
    server_cmd = ["vllm", "serve", config.model_path, "--port", str(config.port)]
    for arg in config.server_args:
        server_cmd.extend(shlex.split(arg))
    print("  Command: " + " ".join(server_cmd))

    if config.run_mode == "benchmark":
        print("\n[5] Benchmark Cases:")
        for idx, tc in enumerate(config.test_configs, start=1):
            print(
                f"  Case {idx:02d}: conc={tc.concurrency} isl={tc.isl} "
                f"osl={tc.osl} prompts={tc.num_prompts}"
            )
        print(f"  random_range_ratio={config.random_range_ratio}")
        print(f"  request_rate={config.request_rate}")
        print(f"  burstiness={config.burstiness}")
        print(f"  num_warmups={config.num_warmups}")
        print(f"  ignore_eos={config.benchmark_ignore_eos}")
    elif config.run_mode == "eval":
        print("\n[5] Eval:")
        print(f"  tasks={config.eval_tasks}")
        print(f"  num_fewshot={config.eval_num_fewshot}")
        print(f"  batch_size={config.eval_batch_size}")
        print(f"  limit={config.eval_limit}")
        print(f"  num_concurrent={config.eval_num_concurrent}")
        print(f"  max_gen_toks={config.eval_max_gen_toks}")
    elif config.run_mode == "chat":
        print("\n[5] Chat:")
        print(f"  prompt={config.chat_prompt or '<interactive>'}")
        print(f"  stream={config.chat_stream}")
        print(f"  max_tokens={config.chat_max_tokens}")
        print(f"  enable_thinking={config.enable_thinking}")
    elif config.run_mode == "longform":
        print("\n[5] Longform:")
        print(f"  prompts={len(config.longform_prompts)}")
        print(f"  max_tokens={config.longform_max_tokens}")
        print(f"  enable_thinking={config.enable_thinking}")
    elif config.run_mode == "multiturn":
        print("\n[5] Multiturn:")
        print(f"  inline_turns={len(config.multiturn_turns)}")
        print(f"  turns_file={config.multiturn_turns_file}")
        print(f"  max_tokens={config.multiturn_max_tokens}")
        print(f"  enable_thinking={config.enable_thinking}")
    elif config.run_mode == "profile":
        print("\n[5] Profile Cases:")
        for idx, case in enumerate(config.profile_configs, start=1):
            num_prompts = case.num_prompts or case.concurrency
            case_name = (
                f"profile_{idx:02d}_conc{case.concurrency}"
                f"_isl{case.isl}_osl{case.osl}_np{num_prompts}"
            )
            profile_dir = f"{config.suite_path_in_container}/_output/{case_name}/traces"
            result_stem = (
                f"{case_name}_{_sanitize(config.model_prefix)}_{config.precision}"
                f"_{config.framework}_tp{config.tensor_parallel_size()}"
            )
            print(
                f"  Case {idx:02d}: conc={case.concurrency} isl={case.isl} "
                f"osl={case.osl} prompts={num_prompts}"
            )
            print(f"    profile_with_stack={case.profile_with_stack}")
            print(f"    VLLM_TORCH_PROFILER_DIR={profile_dir}")
            print(f"    VLLM_RPC_TIMEOUT=1800000")
            print(
                "    Benchmark command: "
                + _profile_benchmark_command(config, case, num_prompts, case_name, result_stem)
            )

    print(f"\n[6] Output:")
    print(f"  Run directory: {run_dir}")
    print("\n" + "=" * 60)
    print("  DRY RUN COMPLETE")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM existing-container benchmark suite")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_FILE"),
        help="Path to YAML configuration file",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-mode", choices=["benchmark", "eval", "chat", "longform", "multiturn", "profile"])
    parser.add_argument("--port", type=int)
    parser.add_argument("--num-warmups", type=int)
    parser.add_argument("--eval-tasks", type=str)
    parser.add_argument("--eval-num-fewshot", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--eval-num-concurrent", type=int)
    parser.add_argument("--chat-prompt", type=str)
    parser.add_argument("--chat-max-tokens", type=int)

    args = parser.parse_args(argv)
    if not args.config:
        parser.error("--config is required (or set CONFIG_FILE env var)")

    cli_overrides = {
        "run_mode": args.run_mode,
        "port": args.port,
        "num_warmups": args.num_warmups,
        "eval_tasks": args.eval_tasks,
        "eval_num_fewshot": args.eval_num_fewshot,
        "eval_limit": args.eval_limit,
        "eval_num_concurrent": args.eval_num_concurrent,
        "chat_prompt": args.chat_prompt,
        "chat_max_tokens": args.chat_max_tokens,
    }

    try:
        config = load_config(args.config, cli_overrides)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = script_dir / "runs" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _print_dry_run(config, run_dir, project_root)
        try:
            run_dir.rmdir()
        except OSError:
            pass
        return 0

    _print_banner(config, run_dir)

    from .container import ExistingContainerManager
    from .runner import VllmBenchmarkRunner

    container = ExistingContainerManager(config, project_root, run_dir)
    runner = VllmBenchmarkRunner(config, container, run_dir)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n>>> Interrupted. Cleaning up...")
        runner.cleanup_best_effort()
        return 1
    except Exception as exc:
        print(f"\n>>> Error: {exc}", file=sys.stderr)
        runner.cleanup_best_effort()
        return 1
    return 0


def _sanitize(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def _profile_benchmark_command(
    config: SuiteConfig,
    case,
    num_prompts: int,
    case_name: str,
    result_stem: str,
) -> str:
    cmd = [
        "python3",
        f"{config.suite_path_in_container}/sgl_bench/utils/bench_serving/benchmark_serving.py",
        "--backend",
        "vllm",
        "--base-url",
        f"http://localhost:{config.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        config.model_path,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(case.isl),
        "--random-output-len",
        str(case.osl),
        "--random-range-ratio",
        str(config.random_range_ratio),
        "--num-prompts",
        str(num_prompts),
        "--num-warmups",
        str(config.num_warmups),
        "--request-rate",
        config.request_rate,
        "--max-concurrency",
        str(case.concurrency),
        "--burstiness",
        str(config.burstiness),
        "--trust-remote-code",
        "--save-result",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        f"{config.suite_path_in_container}/_output/{case_name}",
        "--result-filename",
        f"{result_stem}.json",
        "--profile",
    ]
    if config.benchmark_ignore_eos:
        cmd.append("--ignore-eos")
    if config.benchmark_temperature is not None:
        cmd.extend(["--temperature", str(config.benchmark_temperature)])
    if config.benchmark_extra_request_body:
        import json

        cmd.extend(["--extra-request-body", json.dumps(config.benchmark_extra_request_body, ensure_ascii=False)])
    return " ".join(shlex.quote(part) for part in cmd)

