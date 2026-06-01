# sgl_benchmark

配置驱动的推理测试套件，包含两个入口：

- [`sgl_bench/`](sgl_bench/)：SGLang 推理测试套件，覆盖性能压测、模型精度评测、长文本生成、多轮对话验证、Torch Profiler 采集等场景。详细文档见 [`sgl_bench/README.md`](sgl_bench/README.md)。
- [`vllm_bench/`](vllm_bench/)：vLLM 推理测试套件，支持 attach 到已有容器或从镜像创建临时容器，覆盖 vLLM serving 性能压测和 `lm_eval` 精度评测。详细文档见 [`vllm_bench/README.md`](vllm_bench/README.md)。

两个入口都使用 Python + YAML 配置驱动，具有配置校验和 dry-run 预览功能。

## 快速体验

```bash
# 安装依赖
pip install pyyaml pydantic docker httpx

# SGLang: 先 dry-run 预览配置（不启动容器）
python3 -m sgl_bench --config sgl_bench/config_examples/config_4b_benchmark.yaml --dry-run

# SGLang: 正式运行性能压测
python3 -m sgl_bench --config sgl_bench/config_examples/config_4b_benchmark.yaml

# vLLM: 预览 DeepSeek-V4-Pro TP8 1K/1K 性能配置
python3 -m vllm_bench --config vllm_bench/config_examples/deepseek_v4_pro_tp8_1k1k_image.yaml --dry-run

# vLLM: 运行 MTP 性能压测
python3 -m vllm_bench --config vllm_bench/config_examples/deepseek_v4_pro_tp8_1k1k_mtp_perf_acc_base1_image.yaml

# vLLM: 运行 GSM8K 20-shot 精度评测
python3 -m vllm_bench --config vllm_bench/config_examples/deepseek_v4_pro_tp8_gsm8k_eval_mtp_perf_acc_base1_image.yaml
```

## 支持的测试模式

| 模式 | 说明 | 用途 |
|------|------|------|
| `benchmark` | 性能压测 | 测量 TPOT、TTFT、QPS、吞吐量；SGLang/vLLM 均支持 |
| `eval` | 精度评测（lm_eval） | 标准化评测（gsm8k、mmlu 等）；SGLang/vLLM 均支持 |
| `chat` | 交互对话 | 验证模型基本推理能力 |
| `longform` | 长文本生成 | 验证长输出的连贯性和尾部质量 |
| `multiturn` | 多轮对话 | 验证跨轮次的上下文记忆能力 |
| `profile` | Torch Profiler 采集 | 收集 GPU kernel 和 Python 调用栈 trace |

其中 `chat`、`longform`、`multiturn`、`profile` 当前由 `sgl_bench` 提供；`vllm_bench` 当前聚焦 `benchmark` 和 `eval`。

## 运行前提

- Docker 已安装
- 测试镜像包含目标推理框架及相关依赖（SGLang 或 vLLM）
- 机器具备匹配的 GPU/驱动环境
