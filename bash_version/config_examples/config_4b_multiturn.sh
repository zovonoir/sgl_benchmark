#!/bin/bash
# 示例：Qwen3.5-4B 多轮对话记忆验证

IMAGE="atom-sglang:latest"
MODEL_PATH="/.cache/huggingface/Qwen3.5-4B"
MODEL_PREFIX="qwen3.5_4b"
HOST_MODEL_MOUNT_PATH="/raid/models"

RUN_MODE="multiturn"
PORT="8888"
ENABLE_THINKING=false               # true=输出思考过程，false=仅输出答案

# 方式一：直接在配置文件中定义每轮对话内容（推荐）
MULTITURN_TURNS=(
    "我叫李明，来自上海，是一名数据工程师。我养了一只叫「豆豆」的柯基犬，今年2岁。请记住这些信息，然后介绍一下 Apache Spark 的核心架构。"
    "谢谢。我们团队正在用 Spark 处理日均10TB的日志数据，存储在 HDFS 上，使用 Parquet 格式。请分析这个场景下的性能优化策略。"
    "写一篇介绍哥德尔不完备定理的介绍，包括发现者，发现历史和定理概述，不低于1000字"
    "请详细解释群公理以及什么是Abel群什么是Galois群"
    "群公理中是否包含交换律?"
    "现在请回答：1. 我叫什么名字？来自哪里？2. 我的宠物叫什么？什么品种？3. 我们的数据规模和存储格式是什么？"
)

# 方式二：指向外部 JSON 文件（适合轮次较多或内容复杂的场景）
# MULTITURN_TURNS_FILE="./config_examples/multiturn_turns_example.json"

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
