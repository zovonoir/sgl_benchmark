"""CLI entry point for SGLang Benchmark Suite.

Usage:
    python -m python_version --config config.yaml
    python -m python_version --config config.yaml --dry-run
    python -m python_version --config config.yaml --run-mode eval --eval-tasks mmlu
"""

import argparse
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

from .config import SuiteConfig, load_config
from .container import ContainerManager
from .runners.benchmark import BenchmarkRunner
from .runners.chat import ChatRunner
from .runners.eval_runner import EvalRunner
from .runners.longform import LongformRunner
from .runners.multiturn import MultiturnRunner
from .runners.profile import ProfileRunner
from .server import ServerManager

RUNNERS = {
    "benchmark": BenchmarkRunner,
    "chat": ChatRunner,
    "eval": EvalRunner,
    "longform": LongformRunner,
    "multiturn": MultiturnRunner,
    "profile": ProfileRunner,
}


def _print_banner(config: SuiteConfig, run_dir: Path) -> None:
    print("=" * 60)
    print("  sgl_benchmark (Python)")
    print("=" * 60)
    print(f"Run dir       : {run_dir}")
    print(f"Image         : {config.image}")
    print(f"Model         : {config.model_path}")
    print(f"RUN_MODE      : {config.run_mode}")
    print(f"SERVER_ARGS   : {' '.join(config.server_args)}")
    if config.run_mode == "benchmark":
        print(f"BENCH_BACKEND : {config.bench_backend}")
        print(f"Test cases    : {len(config.test_configs)}")
    print("=" * 60)


def _print_dry_run(config: SuiteConfig, run_dir: Path, script_dir: Path) -> None:
    """Print complete dry-run analysis of what would happen."""
    print("\n" + "=" * 60)
    print("  DRY RUN - Configuration Analysis")
    print("=" * 60)

    # 1. Config validation
    print("\n[1] Configuration: VALID")
    print(f"  Config model: {config.model_prefix} ({config.precision})")
    print(f"  Run mode: {config.run_mode}")
    print(f"  Framework: {config.framework}")
    print(f"  Runner type: {config.runner_type}")

    # 2. Docker container
    # Profile mode injects SGLANG_TORCH_PROFILER_DIR automatically
    container_extra_env = None
    if config.run_mode == "profile" and config.profile_configs:
        container_extra_env = {"SGLANG_TORCH_PROFILER_DIR": "(auto-set per case)"}

    print("\n[2] Docker Container:")
    cm = ContainerManager(config, run_dir, script_dir)
    desc = cm.describe(extra_env=container_extra_env)
    print(f"  Image: {desc['image']}")
    print(f"  Container name: {desc['container_name']}")

    if config.docker_run_args:
        print(f"\n  Docker run args:")
        for k, v in config.docker_run_args.items():
            print(f"    {k}: {v}")

    print(f"\n  Mounts ({len(desc['mounts'])}):")
    for m in desc["mounts"]:
        print(f"    {m}")

    print(f"\n  Environment variables ({len(desc['environment'])}):")
    for k, v in sorted(desc["environment"].items()):
        print(f"    {k}={v}")

    if desc["post_start_commands"]:
        print(f"\n  Post-start commands ({len(desc['post_start_commands'])}):")
        for cmd in desc["post_start_commands"]:
            print(f"    $ {cmd}")

    # 3. SGLang server
    print("\n[3] SGLang Server:")
    sm = ServerManager(cm, config)
    skip_warmup = config.run_mode in ("chat", "longform", "multiturn")
    server_extra_args = None
    if config.run_mode == "profile" and config.profile_configs:
        # Show server config for first profile case as representative
        first = config.profile_configs[0]
        skip_warmup = first.skip_server_warmup
        if first.disable_cuda_graph:
            server_extra_args = ["--disable-cuda-graph"]
    server_desc = sm.describe(skip_warmup, server_extra_args)
    print(f"  Command: {server_desc['server_command']}")
    print(f"  Port: {server_desc['port']}")
    print(f"  Health timeout: {server_desc['health_timeout']}")
    print(f"  Watchdog timeout: {server_desc['watchdog_timeout']}")
    print(f"  Stability checks: {server_desc['stability_checks']} consecutive")
    print(f"  Skip warmup: {skip_warmup}")

    # 4. Post-start commands (run after container start, before server start)
    if config.post_start_commands:
        print(f"\n[4] Post-start Commands ({len(config.post_start_commands)}):")
        for cmd in config.post_start_commands:
            print(f"  $ {cmd}")
    else:
        print("\n[4] Post-start Commands: none")

    # 5. Mode-specific plan
    print(f"\n[5] Test Plan ({config.run_mode}):")
    runner_cls = RUNNERS[config.run_mode]
    runner = runner_cls(config, run_dir, script_dir)
    runner.dry_run()

    # 6. Output
    print(f"\n[6] Output:")
    print(f"  Run directory: {run_dir}")

    print("\n" + "=" * 60)
    print("  DRY RUN COMPLETE - no containers were started")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SGLang Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default=os.environ.get("CONFIG_FILE"),
        help="Path to YAML configuration file",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse config and show what would run, without executing")
    parser.add_argument("--run-mode", type=str, choices=list(RUNNERS.keys()))
    parser.add_argument("--chat-prompt", type=str)
    parser.add_argument("--chat-max-tokens", type=int)
    parser.add_argument("--eval-tasks", type=str)
    parser.add_argument("--eval-num-fewshot", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--bench-backend", type=str, choices=["vllm", "sglang"])

    args = parser.parse_args(argv)

    if not args.config:
        parser.error("--config is required (or set CONFIG_FILE env var)")

    # Build CLI overrides dict (only non-None values)
    cli_overrides = {}
    for key in ("run_mode", "chat_prompt", "chat_max_tokens", "eval_tasks",
                "eval_num_fewshot", "eval_limit", "port", "bench_backend"):
        val = getattr(args, key, None)
        if val is not None:
            cli_overrides[key] = val

    try:
        config = load_config(args.config, cli_overrides)
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    script_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = script_dir / "runs" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _print_dry_run(config, run_dir, script_dir)
        # Clean up empty run dir
        try:
            run_dir.rmdir()
        except OSError:
            pass
        return 0

    _print_banner(config, run_dir)

    # Stale container cleanup
    cm = ContainerManager(config, run_dir, script_dir)
    cm.cleanup_stale()

    # Setup signal handler for graceful shutdown
    runner_ref = [None]

    def _signal_handler(signum, frame):
        print("\n>>> Interrupted. Cleaning up...")
        if runner_ref[0]:
            runner_ref[0].container.cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Create and run the appropriate runner
    runner_cls = RUNNERS[config.run_mode]
    runner = runner_cls(config, run_dir, script_dir)
    runner_ref[0] = runner

    try:
        runner.run()
        return 0
    except KeyboardInterrupt:
        print("\n>>> Interrupted. Cleaning up...")
        runner.container.cleanup()
        return 1
    except Exception as e:
        print(f"\n>>> Error: {e}", file=sys.stderr)
        return 1
