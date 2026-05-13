#!/bin/bash
# 示例：Qwen3.5-4B 长文本生成验证

IMAGE="atom-sglang:latest"
MODEL_PATH="/.cache/huggingface/Qwen3.5-4B"
MODEL_PREFIX="qwen3.5_4b"
HOST_MODEL_MOUNT_PATH="/raid/models"

RUN_MODE="longform"
PORT="8888"
ENABLE_THINKING=false               # true=输出思考过程，false=仅输出答案

LONGFORM_PROMPTS=(
    "请写一篇关于人工智能对未来教育影响的深度分析文章，至少包含5个小节。"
    "Write a detailed tutorial on building a REST API with Python Flask, including authentication and database integration."
)

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
