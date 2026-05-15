"""Configuration loading and validation for SGLang Benchmark Suite.

Loads YAML config files, applies environment variable overrides, validates
with pydantic, and produces a frozen SuiteConfig object.

Precedence: CLI flags > environment variables > YAML config > defaults.
"""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TestCaseConfig(BaseModel):
    """A single benchmark test case."""
    concurrency: int
    isl: int
    osl: int
    num_prompts: int


class SuiteConfig(BaseModel):
    """Complete configuration for a benchmark suite run."""
    model_config = ConfigDict(frozen=True)

    # Required (no defaults)
    image: str
    model_path: str
    model_prefix: str
    host_model_mount_path: str

    # Run mode
    run_mode: Literal["benchmark", "chat", "eval", "longform", "multiturn"] = "benchmark"

    # Scalars with defaults
    precision: str = "bf16"
    runner_type: str = "mi308x"
    framework: str = "sglang"
    port: int = 8888
    bench_backend: Literal["vllm", "sglang"] = "vllm"
    random_range_ratio: float = 1.0
    request_rate: str = "inf"
    burstiness: float = 1.0
    health_timeout: int = 240
    watchdog_timeout: int = 600
    enable_thinking: bool = False
    chat_stream: bool = True
    chat_prompt: str | None = None
    chat_max_tokens: int = 8192
    eval_tasks: str = "gsm8k"
    eval_num_fewshot: int = 5
    eval_max_gen_toks: int = 2048
    eval_num_concurrent: int = 224
    eval_batch_size: str = "auto"
    eval_limit: int | None = None
    multiturn_turns_file: str | None = None

    # Arrays / Dicts
    container_env: list[str] = Field(default_factory=list)
    extra_container_mounts: list[str] = Field(default_factory=list)
    docker_run_args: dict = Field(default_factory=dict)
    server_args: list[str] = Field(default_factory=list)
    post_start_commands: list[str] = Field(default_factory=list)
    test_configs: list[TestCaseConfig] = Field(default_factory=list)
    longform_prompts: list[str] = Field(default_factory=list)
    multiturn_turns: list[str] = Field(default_factory=list)

    @field_validator("host_model_mount_path")
    @classmethod
    def _check_model_path_exists(cls, v: str) -> str:
        if not Path(v).is_dir():
            raise ValueError(f"host_model_mount_path not found: {v}")
        return v

    @model_validator(mode="after")
    def _check_mode_requirements(self) -> "SuiteConfig":
        if self.run_mode == "benchmark" and not self.test_configs:
            raise ValueError("test_configs is required for benchmark mode")
        if self.run_mode == "longform" and not self.longform_prompts:
            raise ValueError("longform_prompts is required for longform mode")
        if self.run_mode == "multiturn" and not self.multiturn_turns and not self.multiturn_turns_file:
            raise ValueError("Either multiturn_turns or multiturn_turns_file is required for multiturn mode")
        return self


# Mapping of environment variable names to config field names
_ENV_OVERRIDE_MAP = {
    "RUN_MODE": "run_mode",
    "PRECISION": "precision",
    "RUNNER_TYPE": "runner_type",
    "FRAMEWORK": "framework",
    "PORT": "port",
    "BENCH_BACKEND": "bench_backend",
    "RANDOM_RANGE_RATIO": "random_range_ratio",
    "REQUEST_RATE": "request_rate",
    "BURSTINESS": "burstiness",
    "HEALTH_TIMEOUT": "health_timeout",
    "WATCHDOG_TIMEOUT": "watchdog_timeout",
    "ENABLE_THINKING": "enable_thinking",
    "CHAT_STREAM": "chat_stream",
    "CHAT_PROMPT": "chat_prompt",
    "CHAT_MAX_TOKENS": "chat_max_tokens",
    "EVAL_TASKS": "eval_tasks",
    "EVAL_NUM_FEWSHOT": "eval_num_fewshot",
    "EVAL_MAX_GEN_TOKS": "eval_max_gen_toks",
    "EVAL_NUM_CONCURRENT": "eval_num_concurrent",
    "EVAL_BATCH_SIZE": "eval_batch_size",
    "EVAL_LIMIT": "eval_limit",
    "IMAGE": "image",
    "MODEL_PATH": "model_path",
    "MODEL_PREFIX": "model_prefix",
    "HOST_MODEL_MOUNT_PATH": "host_model_mount_path",
}


def _coerce_bool(val: str) -> bool:
    return val.lower() in ("true", "1", "yes")


def load_config(config_path: str, cli_overrides: dict | None = None) -> SuiteConfig:
    """Load config from YAML file with env var and CLI overrides.

    Precedence: CLI flags > environment variables > YAML config > defaults.
    """
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    for env_key, config_key in _ENV_OVERRIDE_MAP.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            data[config_key] = env_val

    # Apply CLI overrides (highest priority)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                data[k] = v


    return SuiteConfig(**data)
