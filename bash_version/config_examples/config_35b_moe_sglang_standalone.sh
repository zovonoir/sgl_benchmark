#!/bin/bash
# 示例配置：Qwen3.5-35B-A3B-FP8 MoE 模型，TP1 单卡，高并发场景
#
# 使用前请根据实际环境修改（详见 README.md）

IMAGE="atom-sglang:latest"
MODEL_PATH="/.cache/huggingface/Qwen3.5-35B-A3B-FP8"
MODEL_PREFIX="qwen3.5_moe"
HOST_MODEL_MOUNT_PATH="/raid/models"
RUN_MODE="benchmark"

CONTAINER_ENV_OVERRIDES=(
    "SGLANG_DISABLE_CUDNN_CHECK=1"
    "SGLANG_USE_AITER=1"
)

EXTRA_CONTAINER_MOUNTS=()

EXTRA_DOCKER_ARGS=(
    -e "HIP_VISIBLE_DEVICES=0"
    -e "CUDA_VISIBLE_DEVICES=0"
)

SERVER_ARGS=(
    --tensor-parallel-size 1
    --expert-parallel-size 1
    --reasoning-parser qwen3
    --trust-remote-code
    --mem-fraction-static 0.9
    --max-running-requests 224
    --disable-radix-cache
)

POST_START_COMMANDS=()

PRECISION="fp8"
RUNNER_TYPE="mi308x"
FRAMEWORK="sglang"
RANDOM_RANGE_RATIO="1.0"
REQUEST_RATE="inf"
BURSTINESS="1.0"
PORT="8888"
ENABLE_TUNED_GEMM=0

TEST_CONFIGS=(
    "224  4096  2048  448"
)
