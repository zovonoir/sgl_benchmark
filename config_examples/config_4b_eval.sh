#!/bin/bash
# 示例：Qwen3.5-4B lm_eval 精度评测

IMAGE="atom-sglang:latest"
MODEL_PATH="/.cache/huggingface/Qwen3.5-4B"
MODEL_PREFIX="qwen3.5_4b"
HOST_MODEL_MOUNT_PATH="/raid/models"

RUN_MODE="eval"
EVAL_TASKS="gsm8k"
EVAL_NUM_FEWSHOT=5
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
    --mem-fraction-static 0.85
    --disable-radix-cache
)

TEST_CONFIGS=()
