# sgl_benchmark

配置驱动的 SGLang 推理测试套件，覆盖性能压测、模型精度评测、长文本生成质量验证三大场景。基于 Docker + SGLang，可单独拷贝到其他机器使用，不依赖外部仓库代码。

## 快速开始

```bash
# 性能压测（RUN_MODE 在配置文件中设为 benchmark）
CONFIG_FILE=./config_examples/config_35b_moe_sglang_standalone.sh bash run_suite.sh

# 精度评测（RUN_MODE 在配置文件中设为 eval）
CONFIG_FILE=./your_eval_config.sh bash run_suite.sh

# 长文本验证 / 多轮对话 / 交互式 chat — 同理，在配置文件中设置 RUN_MODE
CONFIG_FILE=./your_config.sh bash run_suite.sh
```

所有参数（`RUN_MODE`、`BENCH_BACKEND`、`EVAL_TASKS` 等）都建议写在配置文件中。命令行环境变量仅用于临时覆盖。

## 目录结构

```
LLM_benchmark/
├── run_suite.sh                    # 主入口（benchmark / chat / eval 三种模式）
├── run_case.sh                     # 单个压测用例执行（由 run_suite.sh 调用）
├── summarize_results.sh            # 性能结果汇总
├── accuracy_batch_test.sh          # 批量短 prompt 精度验证
├── accuracy_longform_test.sh       # 长文本生成精度验证
├── accuracy_multiturn_test.py      # 多轮对话记忆验证
├── README.md
├── config_examples/                # 示例配置文件
├── runs/                           # 测试结果输出
└── utils/
    ├── bench_serving/              # vllm 版 benchmark 工具
    │   ├── benchmark_serving.py
    │   ├── backend_request_func.py
    │   └── benchmark_utils.py
    └── process_result.py           # 结果后处理
```

---

## 一、配置文件参数

所有测试通过配置文件驱动，一个配置文件定义一次完整的测试环境（镜像、模型、GPU、环境变量、server 参数等）。

### 必填参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `IMAGE` | Docker 镜像名称 | `atom-sglang:latest` |
| `MODEL_PATH` | 容器内模型路径 | `/.cache/huggingface/Qwen3.5-27B` |
| `MODEL_PREFIX` | 模型前缀，用于结果文件命名 | `qwen3.5` |
| `HOST_MODEL_MOUNT_PATH` | 宿主机模型目录，挂载到容器的 `/.cache/huggingface/` | `/raid/models` |

### 运行控制

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `RUN_MODE` | `benchmark` | `benchmark` / `chat` / `eval` / `longform` / `multiturn`（见下方各章节） |
| `PRECISION` | `bf16` | 精度标注，仅用于结果命名 |
| `RUNNER_TYPE` | `mi308x` | GPU 类型标注 |
| `FRAMEWORK` | `sglang` | 推理框架标注 |
| `PORT` | `8888` | 推理服务监听端口。多实例并行时需要指定不同端口 |

### 压测参数（`RUN_MODE=benchmark`）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `BENCH_BACKEND` | `vllm` | 压测工具后端，详见下方说明 |
| `RANDOM_RANGE_RATIO` | `0.8` | 输入/输出长度随机波动比例。`1.0` = 固定长度 |
| `REQUEST_RATE` | `inf` | 每秒请求发送速率。`inf` = closed-loop |
| `BURSTINESS` | `1.0` | 请求突发程度（仅 `vllm` 后端支持） |

### 精度评测参数（`RUN_MODE=eval`）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `EVAL_TASKS` | `gsm8k` | lm_eval 评测任务名（`gsm8k` / `mmlu` / `hellaswag` 等） |
| `EVAL_NUM_FEWSHOT` | `5` | Few-shot 示例数量 |
| `EVAL_MAX_GEN_TOKS` | `2048` | 最大生成 token 数 |
| `EVAL_NUM_CONCURRENT` | `224` | 并发请求数 |
| `EVAL_BATCH_SIZE` | `auto` | 批处理大小 |
| `HEALTH_TIMEOUT` | `240` | 健康检查最大等待轮数（每轮 5s）。某些模型首次启动需要 JIT 编译，可能需要 300+ |

### 测试矩阵 TEST_CONFIGS

定义压测用例，每行 4 个字段：

```bash
TEST_CONFIGS=(
    "并发数  输入长度  输出长度  总请求数"
)
```

| 字段 | 说明 |
|------|------|
| 并发数 | 同时在飞的最大请求数 |
| 输入长度 (ISL) | Input Sequence Length |
| 输出长度 (OSL) | Output Sequence Length |
| 总请求数 | 总共发送的请求数量，建议 ≥ 2× 并发数 |

TP/EP 等并行参数通过 `SERVER_ARGS` 配置（如 `--tensor-parallel-size 2`）。

`eval` 和 `chat` 模式不使用 `TEST_CONFIGS`。

### 容器环境变量 CONTAINER_ENV_OVERRIDES

注入到容器内的环境变量，用于控制推理框架行为。不同框架和模型需要的环境变量不同，请根据实际情况配置：

```bash
CONTAINER_ENV_OVERRIDES=(
    "KEY1=VALUE1"
    "KEY2=VALUE2"
)
```

### GPU 选择 EXTRA_DOCKER_ARGS

通过 Docker 参数指定 GPU 和其他容器运行时配置：

```bash
EXTRA_DOCKER_ARGS=(
    -e "HIP_VISIBLE_DEVICES=4,5"     # AMD GPU 选择
    -e "CUDA_VISIBLE_DEVICES=4,5"    # NVIDIA GPU 选择
)
```

TP=1 时指定 1 张卡，TP=2 时指定 2 张卡，以此类推。

### 额外挂载 EXTRA_CONTAINER_MOUNTS

将宿主机文件/目录挂载到容器内：

```bash
EXTRA_CONTAINER_MOUNTS=(
    "/host/path:/container/path"
    "/host/path2:/container/path2:ro"    # 只读挂载
)
```

### SGLang 服务参数 SERVER_ARGS

传递给 `sglang.launch_server` 的额外参数：

```bash
SERVER_ARGS=(
    --trust-remote-code
    --mem-fraction-static 0.9
    --disable-radix-cache
    # 根据模型和测试需求添加其他参数
)
```

---

## 二、性能压测

```bash
CONFIG_FILE=./your_config.sh bash run_suite.sh
```

套件自动完成：启动容器 → 注入环境变量 → 启动推理服务 → 等待健康 → 执行 benchmark → 收集结果 → 清理容器。

### `BENCH_BACKEND` 详解

| 值 | 工具 | API 端点 | 说明 |
|---|---|---|---|
| `vllm` | 自带的 `utils/bench_serving/benchmark_serving.py` | `/v1/completions` | 走 OpenAI 兼容接口 |
| `sglang` | 容器内 `python3 -m sglang.bench_serving` | `/generate` | 走 SGLang 原生接口 |

**重要差异**：两个后端在 TPOT 测量上存在约 1.5-2ms 的系统性偏差（sglang 后端偏高），原因是流式 chunk 的 ITL 采集粒度不同。**两者测出的相对性能差异完全一致，但绝对值不可跨后端比较。** 对比测试时应统一使用同一后端，对齐客户的测试方式。

### 关键指标

| 指标 | 说明 |
|------|------|
| TPOT (ms) | Time Per Output Token，decode 阶段每个输出 token 的平均耗时 |
| TTFT (ms) | Time To First Token，首 token 延迟，反映 prefill 性能 |
| QPS (req/s) | 每秒完成的请求数 |
| Output Tput/GPU (tok/s) | 每 GPU 输出吞吐量 |
| Total Tput/GPU (tok/s) | 每 GPU 总吞吐量 |

---

## 三、模型精度评测（lm_eval）

```bash
CONFIG_FILE=./your_eval_config.sh bash run_suite.sh
```

使用 [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) 框架进行标准化评测。套件自动完成容器和 server 生命周期管理。

所有评测参数（`RUN_MODE`、`EVAL_TASKS`、`EVAL_NUM_FEWSHOT` 等）都建议写在配置文件中：

```bash
# 配置文件中设置
RUN_MODE="eval"
EVAL_TASKS="gsm8k"
EVAL_NUM_FEWSHOT=5
```

也支持通过命令行环境变量临时覆盖配置文件中的值：

```bash
# 临时覆盖 task（配置文件中是 gsm8k，这里改为 mmlu）
EVAL_TASKS=mmlu EVAL_NUM_FEWSHOT=0 CONFIG_FILE=./your_eval_config.sh bash run_suite.sh
```

### 性能配置 vs 精度配置

测性能和测精度时可能需要不同的 server 配置。某些加速选项（如 FP8 prefill attention、分块 prefill 等）会提升吞吐量但可能影响数值精度。建议为性能测试和精度测试分别维护独立的配置文件，避免混淆。

---

## 四、长文本生成验证（longform）

测试模型**单次长输出**的质量。每个 prompt 独立发送，各请求之间不共享上下文。模型自然生成到停止，不设 max_tokens 上限。

**适用场景**：验证 decode 阶段的 state 在长时间累积后是否导致输出质量退化（如尾部出现重复、逻辑混乱、内容质量下降等）。

配置文件中设置 `RUN_MODE="longform"` 和 `LONGFORM_PROMPTS` 数组：

```bash
RUN_MODE="longform"
LONGFORM_PROMPTS=(
    "请写一篇关于量子计算的科普文章，至少2000字。"
    "Write a comprehensive guide to Rust's ownership system with code examples."
)
```

`LONGFORM_PROMPTS` 为必填项，未设置会报错。结果保存在 `runs/run_<timestamp>/longform_results.txt`。

## 五、多轮对话记忆验证（multiturn）

测试模型在**多轮对话中的上下文记忆能力**。每轮发送完整对话历史（含之前所有轮次的输入和模型回复），上下文逐轮增长。

**适用场景**：验证模型能否在长上下文中正确回忆早期信息。尤其对含 SSM/RNN state 的混合架构模型（如 GatedDeltaNet），每轮 prefill 会将全部历史压缩进 state，state 精度变化可能影响记忆准确性。

**与 longform 的区别**：

| | longform | multiturn |
|---|---|---|
| 请求方式 | 每个 prompt 独立发送，互不关联 | 每轮携带所有历史消息，上下文逐轮累积 |
| 测试重点 | 单次长输出的连贯性和尾部质量 | 跨轮次的事实记忆保持能力 |
| 上下文增长 | 仅在 decode 阶段内增长（取决于输出长度） | prefill 输入逐轮增长（可达数千~上万 tokens） |

配置文件中设置 `RUN_MODE="multiturn"`，并通过以下两种方式之一定义对话轮次：

**方式一：直接在配置文件中定义（推荐）**

```bash
RUN_MODE="multiturn"
MULTITURN_TURNS=(
    "我叫李明，来自上海。请记住这些信息，然后介绍一下 Spark 的核心架构。"
    "我们用 Spark 处理日均10TB日志，存储用 Parquet。请分析优化策略。"
    "请回答：我叫什么？来自哪里？我们的数据规模和格式是什么？"
)
```

**方式二：指向外部 JSON 文件**（适合轮次较多或内容包含特殊字符的场景）

```bash
RUN_MODE="multiturn"
MULTITURN_TURNS_FILE="./my_turns.json"
```

```json
[
    {"user": "第一轮用户输入"},
    {"user": "第二轮用户输入"},
    {"user": "第三轮用户输入"}
]
```

两者设其一即可，优先使用 `MULTITURN_TURNS` 数组。都不设置会报错。

结果保存在 `runs/run_<timestamp>/`。

### A/B 对比建议

对比两个配置的输出质量时：

1. 用相同 prompt 分别跑两个配置，保存输出文件
2. 混淆文件名（隐去配置信息），交给第三方做盲评
3. 盲评维度：逻辑连贯性、尾部质量、事实准确性、记忆保持
4. 如果盲评无法区分两者，说明优化对输出质量无影响

---

## 六、编写新配置文件

复制示例配置并修改：

```bash
cp config_examples/config_35b_moe_sglang_standalone.sh my_config.sh
# 编辑 my_config.sh 中的镜像、模型、GPU、环境变量等
CONFIG_FILE=./my_config.sh bash run_suite.sh
```

主要修改项：

1. `IMAGE`、`MODEL_PATH` — 指向目标镜像和模型
2. `EXTRA_DOCKER_ARGS` — 选择空闲 GPU
3. `TEST_CONFIGS` — 定义测试参数（并发数、ISL、OSL 等）
4. `BENCH_BACKEND` — 选择压测工具，对齐客户的测试方式
5. `CONTAINER_ENV_OVERRIDES` — 根据框架/模型需要注入环境变量
6. `SERVER_ARGS` — 根据模型需要设置 server 参数

## 七、运行前提

- 机器上已安装 `docker`
- 测试镜像包含 `sglang`、`transformers`、`aiohttp`、`numpy`、`tqdm` 等依赖
- 精度测试（`eval` 模式）需要镜像内安装 `lm_eval`（`pip install lm_eval[api]`）
- `HOST_MODEL_MOUNT_PATH` 目录下有对应模型
- 机器具备与镜像匹配的 GPU/驱动环境

## 八、常见问题

### 模型启动超时

某些模型首次启动需要 JIT 编译 GPU kernel（可能 15-30 分钟），超过默认健康检查超时。解决方法：在配置文件中设置 `HEALTH_TIMEOUT=360`（或更大值），或先手动启动一次让 JIT cache 预热。

### lm_eval 分数异常偏低

检查 server 配置是否使用了影响精度的加速选项。某些 FP8 加速、分块 prefill 等可能降低 few-shot 推理精度。建议测精度时使用单独的"精度配置"文件。

### 性能数据跨后端不可比

`BENCH_BACKEND=vllm` 和 `BENCH_BACKEND=sglang` 测出的 TPOT 有约 1.5-2ms 系统性差异。对比测试必须使用同一后端。
