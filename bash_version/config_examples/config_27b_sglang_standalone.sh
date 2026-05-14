#!/bin/bash
# 示例配置：Qwen3.5-27B Dense 模型，BF16 精度，TP2 双卡，长上下文场景
#
# 使用前请根据实际环境修改（详见 README.md）

IMAGE="atom-sglang:latest"
MODEL_PATH="/.cache/huggingface/Qwen3.5-27B"
MODEL_PREFIX="qwen3.5"
HOST_MODEL_MOUNT_PATH="/raid/models"
RUN_MODE="benchmark"

CONTAINER_ENV_OVERRIDES=(
    "SGLANG_DISABLE_CUDNN_CHECK=1"
    "SGLANG_USE_AITER=1"
)

EXTRA_CONTAINER_MOUNTS=()

EXTRA_DOCKER_ARGS=(
    -e "HIP_VISIBLE_DEVICES=0,1"      # TP2 需要 2 张卡
    -e "CUDA_VISIBLE_DEVICES=0,1"
)

SERVER_ARGS=(
    --tensor-parallel-size 2
    --expert-parallel-size 2
    --reasoning-parser qwen3
    --trust-remote-code
    --mem-fraction-static 0.9
    --max-running-requests 128
    --disable-radix-cache
)

POST_START_COMMANDS=()

PRECISION="bf16"
RUNNER_TYPE="mi308x"
FRAMEWORK="sglang"
RANDOM_RANGE_RATIO="1.0"
REQUEST_RATE="inf"
BURSTINESS="1.0"
PORT="8888"
ENABLE_TUNED_GEMM=0

TEST_CONFIGS=(
    "1  60381  132  1"               # 并发1, ISL=60K, OSL=132, 1请求
)
