"""Configuration loading and validation for vLLM benchmark runs."""

from __future__ import annotations

from typing import Any
import os
import shlex
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TestCaseConfig(BaseModel):
    """A single benchmark test case."""

    concurrency: int
    isl: int
    osl: int
    num_prompts: int


class SuiteConfig(BaseModel):
    """Complete configuration for an existing-container vLLM benchmark run."""

    model_config = ConfigDict(frozen=True)

    # Container source: exactly one of existing_container or image.
    existing_container: str | None = None
    image: str | None = None
    container_name: str | None = None
    suite_path_in_container: str = "/tmp/vllm_bench_suite"

    # Model and metadata.
    run_mode: Literal["benchmark", "eval", "chat", "longform", "multiturn"] = "benchmark"
    model_path: str
    model_prefix: str
    host_model_mount_path: str | None = None
    precision: str = "fp4"
    runner_type: str = "mi355x"
    framework: str = "vllm"

    # Runtime settings.
    port: int = 8001
    health_timeout: int = 1800  # seconds
    container_env: list[str] = Field(default_factory=list)
    entrypoint: str | None = None
    command: str | list[str] | None = None
    docker_run_args: dict[str, Any] = Field(default_factory=dict)
    extra_container_mounts: list[str] = Field(default_factory=list)
    post_start_commands: list[str] = Field(default_factory=list)
    server_args: list[str] = Field(default_factory=list)
    cleanup_patterns: list[str] = Field(
        default_factory=lambda: [
            "vllm serve",
            "VLLM::EngineCore",
            "VLLM::Worker_TP",
            "multiprocessing.resource_tracker",
        ]
    )

    # Benchmark settings.
    random_range_ratio: float = 0.8
    request_rate: str = "inf"
    burstiness: float = 1.0
    num_warmups: int = 128
    benchmark_ignore_eos: bool = True
    benchmark_temperature: float | None = None
    benchmark_extra_request_body: dict[str, Any] = Field(default_factory=dict)
    test_configs: list[TestCaseConfig] = Field(default_factory=list)

    # Eval settings.
    eval_tasks: str = "gsm8k"
    eval_num_fewshot: int = 20
    eval_batch_size: str = "auto"
    eval_limit: int | None = None
    eval_num_concurrent: int = 256
    eval_max_retries: int = 10
    eval_max_gen_toks: int = 2048
    eval_max_length: int = 1048576
    eval_timeout: int = 60000
    eval_log_samples: bool = True

    # Chat / generation quality settings.
    enable_thinking: bool = False
    chat_prompt: str | None = None
    chat_stream: bool = False
    chat_max_tokens: int = 2048
    chat_temperature: float = 0.0
    longform_prompts: list[str] = Field(default_factory=list)
    longform_max_tokens: int = 8192
    multiturn_turns_file: str | None = None
    multiturn_turns: list[str] = Field(default_factory=list)
    multiturn_max_tokens: int = 2048

    @model_validator(mode="after")
    def _check_required_cases(self) -> "SuiteConfig":
        if bool(self.existing_container) == bool(self.image):
            raise ValueError("exactly one of existing_container or image must be set")
        if self.run_mode == "benchmark" and not self.test_configs:
            raise ValueError("test_configs is required")
        if self.run_mode == "longform" and not self.longform_prompts:
            raise ValueError("longform_prompts is required for longform mode")
        if self.run_mode == "multiturn" and not self.multiturn_turns and not self.multiturn_turns_file:
            raise ValueError("multiturn_turns or multiturn_turns_file is required for multiturn mode")
        return self

    def container_environment(self) -> dict[str, str]:
        """Return container_env as a mapping."""

        env: dict[str, str] = {}
        for spec in self.container_env:
            if "=" in spec:
                key, value = spec.split("=", 1)
                env[key] = os.path.expandvars(value)
        return env

    def tensor_parallel_size(self) -> int:
        """Infer TP from server_args, defaulting to one GPU."""

        parts: list[str] = []
        for arg in self.server_args:
            parts.extend(shlex.split(arg))
        for idx, part in enumerate(parts):
            if part in ("--tensor-parallel-size", "--tp-size"):
                if idx + 1 < len(parts):
                    return int(parts[idx + 1])
            if part.startswith("--tensor-parallel-size="):
                return int(part.split("=", 1)[1])
            if part.startswith("--tp-size="):
                return int(part.split("=", 1)[1])
        return 1


_ENV_OVERRIDE_MAP = {
    "EXISTING_CONTAINER": "existing_container",
    "IMAGE": "image",
    "CONTAINER_NAME": "container_name",
    "MODEL_PATH": "model_path",
    "MODEL_PREFIX": "model_prefix",
    "RUN_MODE": "run_mode",
    "HOST_MODEL_MOUNT_PATH": "host_model_mount_path",
    "PRECISION": "precision",
    "RUNNER_TYPE": "runner_type",
    "FRAMEWORK": "framework",
    "PORT": "port",
    "HEALTH_TIMEOUT": "health_timeout",
    "RANDOM_RANGE_RATIO": "random_range_ratio",
    "REQUEST_RATE": "request_rate",
    "BURSTINESS": "burstiness",
    "NUM_WARMUPS": "num_warmups",
    "BENCHMARK_IGNORE_EOS": "benchmark_ignore_eos",
    "BENCHMARK_TEMPERATURE": "benchmark_temperature",
    "EVAL_TASKS": "eval_tasks",
    "EVAL_NUM_FEWSHOT": "eval_num_fewshot",
    "EVAL_BATCH_SIZE": "eval_batch_size",
    "EVAL_LIMIT": "eval_limit",
    "EVAL_NUM_CONCURRENT": "eval_num_concurrent",
    "EVAL_MAX_RETRIES": "eval_max_retries",
    "EVAL_MAX_GEN_TOKS": "eval_max_gen_toks",
    "EVAL_MAX_LENGTH": "eval_max_length",
    "EVAL_TIMEOUT": "eval_timeout",
    "ENABLE_THINKING": "enable_thinking",
    "CHAT_PROMPT": "chat_prompt",
    "CHAT_STREAM": "chat_stream",
    "CHAT_MAX_TOKENS": "chat_max_tokens",
    "CHAT_TEMPERATURE": "chat_temperature",
    "LONGFORM_MAX_TOKENS": "longform_max_tokens",
    "MULTITURN_MAX_TOKENS": "multiturn_max_tokens",
}


def _coerce_value(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in ("true", "1", "yes", "y")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def load_config(config_path: str, cli_overrides: dict[str, Any] | None = None) -> SuiteConfig:
    """Load YAML config with env and CLI overrides."""

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for env_key, config_key in _ENV_OVERRIDE_MAP.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            data[config_key] = _coerce_value(env_val, data.get(config_key, env_val))

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                data[key] = value

    return SuiteConfig(**data)

