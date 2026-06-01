# sgl_benchmark

配置驱动的 SGLang 推理测试套件，覆盖性能压测、模型精度评测、长文本生成、多轮对话验证、Torch Profiler 采集五大场景。基于 Docker + SGLang，可独立部署到任意机器使用。

当前项目使用 Python + YAML 配置驱动，具有配置校验（pydantic）和 dry-run 预览功能。详细文档见 [`py_llm_bench/README.md`](py_llm_bench/README.md)。

## 快速体验

```bash
# 安装依赖
pip install pyyaml pydantic docker httpx

# 先 dry-run 预览配置（不启动容器）
python3 -m py_llm_bench --config py_llm_bench/config_examples/config_4b_benchmark.yaml --dry-run

# 正式运行性能压测
python3 -m py_llm_bench --config py_llm_bench/config_examples/config_4b_benchmark.yaml
```

## 支持的测试模式

| 模式 | 说明 | 用途 |
|------|------|------|
| `benchmark` | 性能压测 | 测量 TPOT、TTFT、QPS、吞吐量 |
| `eval` | 精度评测（lm_eval） | 标准化评测（gsm8k、mmlu 等） |
| `chat` | 交互对话 | 验证模型基本推理能力 |
| `longform` | 长文本生成 | 验证长输出的连贯性和尾部质量 |
| `multiturn` | 多轮对话 | 验证跨轮次的上下文记忆能力 |
| `profile` | Torch Profiler 采集 | 收集 GPU kernel 和 Python 调用栈 trace |

## 运行前提

- Docker 已安装
- 测试镜像包含 SGLang 及相关依赖
- 机器具备匹配的 GPU/驱动环境
