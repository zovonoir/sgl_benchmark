#!/bin/bash
# 示例：Qwen3.5-4B 交互式 chat

IMAGE="atom-sglang:latest"
MODEL_PATH="/.cache/huggingface/Qwen3.5-4B"
MODEL_PREFIX="qwen3.5_4b"
HOST_MODEL_MOUNT_PATH="/raid/models"

RUN_MODE="chat"
# CHAT_PROMPT="1+1等于几？"       # 设置后为单次请求模式，不设则进入交互模式
PORT="8888"
ENABLE_THINKING=false               # true=输出思考过程，false=仅输出答案
CHAT_STREAM=true                    # true=流式逐字输出，false=等待完整响应后输出

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
