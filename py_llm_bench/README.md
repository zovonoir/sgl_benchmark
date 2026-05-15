# py_llm_bench

基于 Python 的 SGLang 推理测试套件，使用 YAML 配置文件驱动，覆盖性能压测、模型精度评测、长文本生成、多轮对话四大场景。基于 Docker + SGLang，可独立部署使用。

## 快速开始

### 安装依赖

```bash
pip install pyyaml pydantic docker httpx
```

### 运行测试

```bash
# 先用 dry-run 检查配置（不会启动任何容器）
python3 -m py_llm_bench --config py_llm_bench/config_examples/config_4b_benchmark.yaml --dry-run

# 正式运行
python3 -m py_llm_bench --config py_llm_bench/config_examples/config_4b_benchmark.yaml
```

### 五种运行模式

```bash
# 性能压测
python3 -m py_llm_bench --config config_benchmark.yaml

# 交互对话
python3 -m py_llm_bench --config config_chat.yaml

# 精度评测（lm_eval）
python3 -m py_llm_bench --config config_eval.yaml

# 长文本生成验证
python3 -m py_llm_bench --config config_longform.yaml

# 多轮对话记忆验证
python3 -m py_llm_bench --config config_multiturn.yaml
```

### CLI 参数覆盖

命令行参数优先级最高，可以临时覆盖配置文件中的值：

```bash
# 临时切换 benchmark 后端
python3 -m py_llm_bench --config config.yaml --bench-backend vllm

# 临时切换端口
python3 -m py_llm_bench --config config.yaml --port 9999

# 临时切换 eval 任务
python3 -m py_llm_bench --config config.yaml --run-mode eval --eval-tasks mmlu --eval-num-fewshot 0

# 限制评测样本数（快速调试）
python3 -m py_llm_bench --config config.yaml --run-mode eval --eval-limit 50

# 单次 chat 模式
python3 -m py_llm_bench --config config_chat.yaml --chat-prompt "1+1等于几？"
```

环境变量也可以覆盖（优先级：CLI > 环境变量 > YAML 配置 > 默认值）：

```bash
RUN_MODE=eval EVAL_TASKS=mmlu python3 -m py_llm_bench --config config.yaml
```

### Dry-run 模式

`--dry-run` 只解析配置文件，不启动容器和服务，而是打印完整的运行计划：

```bash
python3 -m py_llm_bench --config config.yaml --dry-run
```

输出包含：
- 配置校验结果
- Docker 容器启动参数（镜像、挂载、环境变量）
- SGLang server 完整启动命令
- 具体测试计划（benchmark 的每个 case、eval 的 lm_eval 命令、chat/longform/multiturn 的详细参数）
- 输出目录路径

建议每次正式运行前先跑一遍 dry-run 确认配置无误。

---

## 目录结构

```
py_llm_bench/
├── cli.py                    # 命令行入口
├── __main__.py               # python -m py_llm_bench 支持
├── config.py                 # YAML 配置加载 + pydantic 校验
├── container.py              # Docker 容器生命周期管理
├── server.py                 # SGLang server 启动/健康检查/预热
├── report.py                 # 性能结果汇总报告
├── run_case.sh               # 容器内 benchmark 执行器（保留）
├── runners/
│   ├── base.py               # BaseRunner 模板方法
│   ├── benchmark.py          # 性能压测
│   ├── chat.py               # 交互对话
│   ├── eval_runner.py        # lm_eval 精度评测
│   ├── longform.py           # 长文本生成
│   └── multiturn.py          # 多轮对话
├── config_examples/          # 配置示例
└── utils/
    └── bench_serving/        # 容器内 benchmark 工具（保留）
```

---

## YAML 配置参数详解

YAML 基础语法：用 `key: value` 表示键值对，用缩进表示层级，用 `-` 表示列表项。`#` 开头为注释。

### 必填参数

每个配置文件都必须包含以下 4 个参数：

```yaml
# Docker 镜像名称
image: "atom-sglang:latest"

# 容器内的模型路径
model_path: "/.cache/huggingface/Qwen3.5-4B"

# 模型简称，用于输出文件命名
model_prefix: "qwen3.5_4b"

# 宿主机模型目录，会挂载到容器的 /.cache/huggingface/
host_model_mount_path: "/raid/models"
```

### 运行模式

```yaml
# 可选值: benchmark / chat / eval / longform / multiturn
run_mode: "benchmark"    # 默认值: benchmark
```

### 通用参数

```yaml
# 精度标注（仅用于结果文件命名，不影响实际推理）
precision: "bf16"        # 默认值: bf16

# GPU 类型标注（仅用于结果文件命名）
runner_type: "mi308x"    # 默认值: mi308x

# 推理框架标注（仅用于结果文件命名）
framework: "sglang"      # 默认值: sglang

# 推理服务监听端口（多实例并行时需指定不同端口）
port: 8888               # 默认值: 8888

# 健康检查最大轮次（每轮 5 秒，默认 240 轮 = 1200 秒）
# 某些模型首次启动需要 JIT 编译，可能需要更大的值
health_timeout: 240      # 默认值: 240

# SGLang watchdog 超时（秒），防止 JIT 编译期间服务器被 watchdog 杀死
watchdog_timeout: 600    # 默认值: 600
```

### 压测参数（run_mode: benchmark）

```yaml
run_mode: "benchmark"

# 压测工具后端
#   vllm:   走 /v1/completions OpenAI 兼容接口
#   sglang: 走 /generate SGLang 原生接口
# 注意：两个后端的 TPOT 有约 1.5-2ms 系统性差异，不可跨后端比较
bench_backend: "sglang"  # 默认值: vllm

# 输入/输出长度随机波动比例
#   1.0 = 固定长度（精确 ISL/OSL）
#   0.8 = 在 [ISL*0.8, ISL] 范围内随机
random_range_ratio: 1.0  # 默认值: 1.0

# 每秒请求发送速率
#   "inf" = closed-loop（尽快发送，不限速）
#   "10"  = 每秒 10 个请求
request_rate: "inf"      # 默认值: inf

# 请求突发程度（仅 vllm 后端支持）
#   1.0 = Poisson 分布
burstiness: 1.0          # 默认值: 1.0

# 测试矩阵，每项定义一个 benchmark case
test_configs:
  - concurrency: 64       # 并发数（同时在飞的最大请求数）
    isl: 2048              # 输入长度 (Input Sequence Length)
    osl: 512               # 输出长度 (Output Sequence Length)
    num_prompts: 128       # 总请求数（建议 >= 2 倍并发数）

  - concurrency: 128
    isl: 4096
    osl: 1024
    num_prompts: 256

  # 可以定义多个 case，会依次运行
```

### Chat 参数（run_mode: chat）

```yaml
run_mode: "chat"

# 单次请求模式：设置此项后直接发送该 prompt 并退出
# 不设置则进入交互式对话模式
chat_prompt: "1+1等于几？"         # 默认值: 不设置（交互模式）

# 流式输出开关
#   true  = 逐 token 实时输出
#   false = 等待完整响应后一次性输出
chat_stream: true                   # 默认值: true

# 最大输出 token 数
chat_max_tokens: 8192               # 默认值: 8192

# 是否启用模型思考过程（thinking/reasoning）
enable_thinking: false              # 默认值: false
```

### 精度评测参数（run_mode: eval）

```yaml
run_mode: "eval"

# lm_eval 评测任务名
eval_tasks: "gsm8k"                 # 默认值: gsm8k
# 其他常见任务: mmlu, hellaswag, arc_easy, winogrande

# Few-shot 示例数量
eval_num_fewshot: 5                 # 默认值: 5

# 最大生成 token 数
eval_max_gen_toks: 2048             # 默认值: 2048

# 并发请求数
eval_num_concurrent: 224            # 默认值: 224

# 批处理大小
eval_batch_size: "auto"             # 默认值: auto

# 限制评测样本数（用于快速调试）
# 不设置或设为 null 则评测完整数据集
eval_limit: 100                     # 默认值: 不限制
```

### 长文本生成参数（run_mode: longform）

```yaml
run_mode: "longform"

# 是否启用模型思考过程
enable_thinking: false              # 默认值: false

# 长文本 prompt 列表（必填，至少 1 个）
longform_prompts:
  - "请写一篇关于人工智能对未来教育影响的深度分析文章，至少包含5个小节。"
  - "Write a detailed tutorial on building a REST API with Python Flask."
```

### 多轮对话参数（run_mode: multiturn）

```yaml
run_mode: "multiturn"

# 是否启用模型思考过程
enable_thinking: false              # 默认值: false

# 方式一：直接在配置文件中定义每轮对话内容（推荐）
multiturn_turns:
  - "我叫李明，来自上海。请记住这些信息，然后介绍一下 Spark 的核心架构。"
  - "我们用 Spark 处理日均10TB日志，存储用 Parquet。请分析优化策略。"
  - "请回答：我叫什么？来自哪里？我们的数据规模和格式是什么？"

# 方式二：指向外部 JSON 文件（适合轮次多或内容包含特殊字符）
# multiturn_turns_file: "./my_turns.json"
# JSON 格式: [{"user": "第一轮"}, {"user": "第二轮"}, ...]
#
# 两者设其一即可，multiturn_turns 优先
```

### 容器环境变量

注入到容器内的环境变量，用于控制推理框架行为：

```yaml
container_env_overrides:
  - "SGLANG_DISABLE_CUDNN_CHECK=1"
  - "SGLANG_USE_AITER=1"
  - "SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1"
  - "AITER_QUICK_REDUCE_QUANTIZATION=INT4"
```

### GPU 选择

通过 `extra_docker_args` 指定使用哪些 GPU，直接写 `KEY=VALUE` 格式：

```yaml
# TP=1（单卡）
extra_docker_args:
  - "HIP_VISIBLE_DEVICES=0"        # AMD GPU
  - "CUDA_VISIBLE_DEVICES=0"       # NVIDIA GPU

# TP=2（双卡）
extra_docker_args:
  - "HIP_VISIBLE_DEVICES=4,5"
  - "CUDA_VISIBLE_DEVICES=4,5"
```

### 额外挂载

将宿主机文件/目录挂载到容器内：

```yaml
extra_container_mounts:
  - "/host/path:/container/path"
  - "/host/path2:/container/path2:ro"    # 只读挂载
```

### SGLang 服务参数

传递给 `sglang.launch_server` 的额外参数：

```yaml
server_args:
  - "--tensor-parallel-size 1"
  - "--trust-remote-code"
  - "--mem-fraction-static 0.9"
  - "--disable-radix-cache"
  - "--attention-backend aiter"
  # 带值的参数写在一行，flag 参数单独一行
```

### 容器启动后自定义指令

在容器启动后、server 启动前执行的自定义指令。适合注入自定义配置、修改容器内文件等：

```yaml
post_start_commands:
  # 示例：注入 tuned GEMM 配置
  - "tail -n +2 /simple-suite/tuned_gemm.csv >> /path/to/aiter.csv"
  # 示例：安装额外依赖
  - "pip install some-package"
  # 示例：修改容器内配置
  - "echo 'export MY_VAR=1' >> /root/.bashrc"
```

---

## 完整配置文件示例

### 性能压测（benchmark）

```yaml
image: "atom-sglang:latest"
model_path: "/.cache/huggingface/Qwen3.5-35B-A3B-FP8"
model_prefix: "qwen3.5_moe"
host_model_mount_path: "/raid/models"

run_mode: benchmark
bench_backend: sglang
precision: fp8
port: 8888

container_env_overrides:
  - "SGLANG_DISABLE_CUDNN_CHECK=1"
  - "SGLANG_USE_AITER=1"

extra_docker_args:
  - "HIP_VISIBLE_DEVICES=4"
  - "CUDA_VISIBLE_DEVICES=4"

extra_container_mounts:
  - "/raid/users/code/ATOM:/code_backup/ATOM"

server_args:
  - "--attention-backend aiter"
  - "--trust-remote-code"
  - "--mem-fraction-static 0.9"
  - "--disable-radix-cache"
  - "--max-running-requests 224"

test_configs:
  - concurrency: 224
    isl: 4096
    osl: 2048
    num_prompts: 448
```

### 精度评测（eval）

```yaml
image: "atom-sglang:latest"
model_path: "/.cache/huggingface/Qwen3.5-4B"
model_prefix: "qwen3.5_4b"
host_model_mount_path: "/raid/models"

run_mode: eval
eval_tasks: gsm8k
eval_num_fewshot: 5
port: 8888

container_env_overrides:
  - "SGLANG_DISABLE_CUDNN_CHECK=1"

extra_docker_args:
  - "HIP_VISIBLE_DEVICES=0"
  - "CUDA_VISIBLE_DEVICES=0"

server_args:
  - "--tensor-parallel-size 1"
  - "--trust-remote-code"
  - "--mem-fraction-static 0.85"
  - "--disable-radix-cache"
```

### 交互对话（chat）

```yaml
image: "atom-sglang:latest"
model_path: "/.cache/huggingface/Qwen3.5-4B"
model_prefix: "qwen3.5_4b"
host_model_mount_path: "/raid/models"

run_mode: chat
chat_stream: true
enable_thinking: false
# chat_prompt: "你好"   # 取消注释启用单次模式
port: 8888

container_env_overrides:
  - "SGLANG_DISABLE_CUDNN_CHECK=1"

extra_docker_args:
  - "HIP_VISIBLE_DEVICES=0"
  - "CUDA_VISIBLE_DEVICES=0"

server_args:
  - "--tensor-parallel-size 1"
  - "--trust-remote-code"
  - "--mem-fraction-static 0.85"
  - "--disable-radix-cache"
```

---

## 输出文件说明

所有测试结果保存在 `py_llm_bench/runs/run_<时间戳>/` 目录下。

### benchmark 模式输出

```
runs/run_20260514_125638/
├── suite_summary_report.txt              # 汇总报告（表格）
└── case_01_conc64_isl2048_osl512_np128/
    ├── *.json                            # 原始 benchmark 结果
    ├── agg_*.json                        # 聚合后的吞吐量数据
    ├── meta_*.json                       # 运行配置元数据
    ├── server_*.log                      # SGLang server 日志
    └── gpu_metrics.csv                   # GPU 监控数据
```

**suite_summary_report.txt 格式**：

```
Run directory : runs/run_20260514_125638
Total cases    : 1
Model path     : /.cache/huggingface/Qwen3.5-4B
Image          : atom-sglang:latest

+-------+------+------+------+----+----+---------+------------+--------+---------+---------+-----------------------+------------------------+------------------------+
| DTYPE | CONC | ISL  | OSL  | TP | EP | PROMPTS | QPS(req/s) | QPS/TP | TPOT(ms)| TTFT(ms)| Input Tput/GPU(tok/s) | Output Tput/GPU(tok/s) | Total Tput/GPU(tok/s)  |
+-------+------+------+------+----+----+---------+------------+--------+---------+---------+-----------------------+------------------------+------------------------+
| bf16  | 64   | 2048 | 512  | 1  | 1  | 128     | 2.7325     | 2.7325 | 42.7    | 2998.2  | 5041.55               | 1262.28                | 6303.83                |
+-------+------+------+------+----+----+---------+------------+--------+---------+---------+-----------------------+------------------------+------------------------+
```

### 关键性能指标

| 指标 | 说明 |
|------|------|
| TPOT (ms) | Time Per Output Token，decode 阶段每个输出 token 的平均耗时 |
| TTFT (ms) | Time To First Token，首 token 延迟，反映 prefill 性能 |
| QPS (req/s) | 每秒完成的请求数 |
| Output Tput/GPU (tok/s) | 每 GPU 输出吞吐量 |
| Total Tput/GPU (tok/s) | 每 GPU 总吞吐量（输入 + 输出） |

### chat 模式输出

```
runs/run_<时间戳>/chat_log.txt
```

### eval 模式输出

```
runs/run_<时间戳>/lm_eval.log              # lm_eval 完整输出
runs/run_<时间戳>/eval_gsm8k_fewshot5/     # lm_eval 结果目录
```

### longform 模式输出

```
runs/run_<时间戳>/longform_results.txt     # 所有 prompt 的生成结果
```

### multiturn 模式输出

```
runs/run_<时间戳>/accuracy_multiturn_multiturn.txt  # 完整对话记录
runs/run_<时间戳>/multiturn.log                     # 运行摘要
runs/run_<时间戳>/_multiturn_turns.json              # 对话轮次定义
```

---

## 编写新配置

1. 复制最接近的示例配置：
   ```bash
   cp py_llm_bench/config_examples/config_4b_benchmark.yaml my_config.yaml
   ```

2. 修改必填参数（`image`、`model_path`、`model_prefix`、`host_model_mount_path`）

3. 修改 GPU 选择（`extra_docker_args` 中的 `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`）

4. 根据需要调整 `server_args`、`test_configs` 等

5. 先 dry-run 确认：
   ```bash
   python3 -m py_llm_bench --config my_config.yaml --dry-run
   ```

6. 正式运行：
   ```bash
   python3 -m py_llm_bench --config my_config.yaml
   ```

---

## BENCH_BACKEND 差异

| 值 | 工具 | API 端点 | 说明 |
|---|---|---|---|
| `vllm` | `benchmark_serving.py` | `/v1/completions` | 走 OpenAI 兼容接口 |
| `sglang` | `sglang.bench_serving` | `/generate` | 走 SGLang 原生接口 |

两个后端在 TPOT 测量上存在约 1.5-2ms 的系统性偏差（sglang 后端偏高），原因是流式 chunk 的 ITL 采集粒度不同。**绝对值不可跨后端比较。** 对比测试时应统一使用同一后端。

---

## 运行前提

- 机器上已安装 Docker
- Python >= 3.10，已安装 `pyyaml`、`pydantic`、`docker`、`httpx`
- 测试镜像包含 `sglang`、`transformers` 等依赖
- 精度测试（eval 模式）需要镜像内安装 `lm_eval`（`pip install lm_eval[api]`）
- `host_model_mount_path` 目录下有对应模型
- 机器具备与镜像匹配的 GPU/驱动环境

---

## 常见问题

### 模型启动超时

某些模型首次启动需要 JIT 编译 GPU kernel（可能 15-30 分钟）。解决方法：在配置中增大 `health_timeout` 和 `watchdog_timeout`：

```yaml
health_timeout: 360     # 360 * 5s = 1800s = 30 分钟
watchdog_timeout: 900   # 15 分钟
```

### lm_eval 分数异常偏低

检查 server 配置是否使用了影响精度的加速选项。某些 FP8 加速、分块 prefill 等可能降低推理精度。建议测精度时使用单独的配置文件，避免使用激进的加速选项。

### 性能数据跨后端不可比

`bench_backend: vllm` 和 `bench_backend: sglang` 测出的 TPOT 有约 1.5-2ms 系统性差异。对比测试必须使用同一后端。

### Dry-run 配置校验失败

`--dry-run` 会检查 `host_model_mount_path` 是否存在。如果在没有模型的机器上编写配置，可以先创建一个空目录通过校验。
