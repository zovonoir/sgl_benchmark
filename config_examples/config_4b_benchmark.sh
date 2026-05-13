#!/bin/bash
# 示例：Qwen3.5-4B 性能压测

IMAGE="atom-sglang:latest"
MODEL_PATH="/.cache/huggingface/Qwen3.5-4B"
MODEL_PREFIX="qwen3.5_4b"
HOST_MODEL_MOUNT_PATH="/raid/models"

RUN_MODE="benchmark"
BENCH_BACKEND="sglang"             # 或 "vllm"
PORT="8888"

CONTAINER_ENV_OVERRIDES=(
    "SGLANG_DISABLE_CUDNN_CHECK=1"
)

EXTRA_DOCKER_ARGS=(
    -e "HIP_VISIBLE_DEVICES=0"
    -e "CUDA_VISIBLE_DEVICES=0"
)

SERVER_ARGS=(
    --tensor-parallel-size 1
    --trust-remote-code
    --mem-fraction-static 0.9
    --disable-radix-cache
)

TEST_CONFIGS=(
    "64  2048  512  128"           # 并发64, ISL=2K, OSL=512, 128请求
)
