# vllm_bench

`vllm_bench` is a Python + YAML runner for vLLM serving performance and
accuracy tests. It starts `vllm serve`, drives the OpenAI-compatible
`/v1/completions` API, and writes raw logs plus summarized results.

It is intentionally separate from `sgl_bench`: `sgl_bench` starts SGLang,
whereas `vllm_bench` starts vLLM.

## Quick Start

Install host-side dependencies:

```bash
pip install pyyaml pydantic docker
```

If the host Python is old, `uv` works well:

```bash
uv run --python 3.12 --with pydantic --with docker --with pyyaml \
  python -m vllm_bench --config vllm_bench/config_examples/deepseek_v4_pro_tp8_1k1k_image.yaml --dry-run
```

Run a benchmark using an already-running container:

```bash
python -m vllm_bench \
  --config vllm_bench/config_examples/deepseek_v4_pro_tp8_1k1k.yaml
```

Run the same benchmark by creating a container from an image:

```bash
python -m vllm_bench \
  --config vllm_bench/config_examples/deepseek_v4_pro_tp8_1k1k_image.yaml
```

Run accuracy evaluation:

```bash
python -m vllm_bench \
  --config vllm_bench/config_examples/deepseek_v4_pro_tp8_gsm8k_eval_nightly_nomtp_image.yaml
```

Preview without creating a container or running a model:

```bash
python -m vllm_bench --config <config.yaml> --dry-run
```

## Run Modes

`vllm_bench` supports:

- `benchmark`: starts `vllm serve` and runs the serving throughput benchmark.
- `eval`: starts `vllm serve` and runs `lm_eval` with the `local-completions`
  backend.
- `chat`: sends a single prompt, or prompts from stdin, through
  `/v1/chat/completions`.
- `longform`: sends configured long-form prompts through `/v1/chat/completions`
  and saves the outputs.
- `multiturn`: sends a configured multi-turn conversation while preserving
  assistant history between turns.
- `profile`: starts `vllm serve` with Torch Profiler enabled, runs a short
  serving benchmark with `--profile`, and saves `.trace.json.gz` files.

The mode can be set in YAML:

```yaml
run_mode: eval
```

or overridden on the command line:

```bash
python -m vllm_bench --config <benchmark-config.yaml> --run-mode eval
```

Command-line overrides currently include:

- `--run-mode benchmark|eval|chat|longform|multiturn|profile`
- `--port`
- `--num-warmups`
- `--eval-tasks`
- `--eval-num-fewshot`
- `--eval-limit`
- `--eval-num-concurrent`
- `--chat-prompt`
- `--chat-max-tokens`

The runner does not rewrite `server_args` when switching modes. If an eval
prompt exceeds a configured `--max-model-len`, the user should increase that
server argument or reduce eval prompt length/few-shot count.

## Container Modes

Set exactly one of `existing_container` or `image`.

### Existing Container

Use this when you manually started a container and want the runner to attach to
it:

```yaml
existing_container: "jjjjjdsv4"
```

The runner injects the local repository into the container at
`suite_path_in_container`, so the container does not need to mount this repo.

### Image Mode

Use this when the runner should create a temporary container:

```yaml
image: "vllm/vllm-openai-rocm:perf_acc_base1"
container_name: "vllm-bench-dsv4"
host_model_mount_path: "/home/sabre/model"
entrypoint: "/bin/bash"
command:
  - "-lc"
  - "while true; do sleep 3600; done"
```

The official vLLM ROCm image normally uses vLLM as the default entrypoint. For
benchmark automation we instead start a long-lived bash container and then use
`docker exec` to launch `vllm serve`. This avoids the image entrypoint consuming
benchmark arguments and makes setup/cleanup predictable.

Common ROCm docker settings:

```yaml
# Same as sgl_bench: image mode automatically mounts this path to
# /.cache/huggingface/ and to the same path inside the container.
host_model_mount_path: "/home/sabre/model"

health_timeout: 240  # 240 polls x 5s = 1200s

docker_run_args:
  group_add: ["video"]
  cap_add: ["SYS_PTRACE"]
  security_opt: ["seccomp=unconfined"]
  devices:
    - "/dev/kfd"
    - "/dev/dri"
  privileged: true
  ipc_mode: "host"
```

## Important YAML Fields

- `model_path`: model repo id or local model path passed to `vllm serve`.
- `model_prefix`: short name used in result filenames.
- `host_model_mount_path`: host-side model/cache directory. In image mode it is
  automatically mounted to `/.cache/huggingface/` and to the same container path,
  matching `sgl_bench`.
- `health_timeout`: server health-check polling rounds. Each round waits 5
  seconds, matching `sgl_bench`.
- `container_env`: environment injected into every `docker exec`.
- `server_args`: arguments appended to `vllm serve`; the framework treats them
  as user-owned and does not auto-correct them.
- `post_start_commands`: commands run after container start and suite injection,
  before starting `vllm serve`; useful for installing `lm_eval`.
- `test_configs`: benchmark cases with `concurrency`, `isl`, `osl`, and
  `num_prompts`.
- `profile_configs`: profiler cases with `concurrency`, `isl`, `osl`,
  optional `num_prompts`, and `profile_with_stack`.
- `eval_tasks`, `eval_num_fewshot`, `eval_limit`, `eval_num_concurrent`: eval
  settings used in `run_mode: eval`.
- `chat_prompt`, `chat_max_tokens`, `chat_temperature`: chat mode settings.
- `longform_prompts`, `longform_max_tokens`: long-form generation settings.
- `multiturn_turns`, `multiturn_turns_file`, `multiturn_max_tokens`: multi-turn
  conversation settings.

For DeepSeek-V4-Pro on the mounted cache used here, set:

```yaml
container_env:
  - "HF_HOME=/.cache/huggingface/"
  - "HF_HUB_CACHE=/.cache/huggingface"
```

## Example Configs

| Config | Purpose |
| --- | --- |
| `deepseek_v4_pro_tp8_1k1k.yaml` | Attach to an existing container and run the original 1K/1K benchmark. |
| `deepseek_v4_pro_tp8_1k1k_image.yaml` | Create a vLLM ROCm container and run the 1K/1K benchmark; can also be used with `--run-mode eval` if eval fields are present. |
| `deepseek_v4_pro_tp8_gsm8k_eval.yaml` | Attach-mode GSM8K eval. |
| `deepseek_v4_pro_tp8_gsm8k_eval_image.yaml` | Image-mode GSM8K eval with the v0.22.0 image. |
| `deepseek_v4_pro_tp8_gsm8k_eval_nightly_nomtp_image.yaml` | No-MTP accuracy recipe using the retagged accuracy-good image. |
| `deepseek_v4_pro_tp8_1k1k_mtp_perf_acc_base1_image.yaml` | MTP 1K/1K benchmark using the accuracy-good image. |
| `deepseek_v4_pro_tp8_gsm8k_eval_mtp_perf_acc_base1_image.yaml` | MTP GSM8K 20-shot eval using the accuracy-good image. |
| `deepseek_v4_pro_tp8_chat_mtp_perf_acc_base1_image.yaml` | MTP single-shot chat example. |
| `deepseek_v4_pro_tp8_longform_mtp_perf_acc_base1_image.yaml` | MTP long-form generation example. |
| `deepseek_v4_pro_tp8_multiturn_mtp_perf_acc_base1_image.yaml` | MTP multi-turn conversation example. |
| `deepseek_v4_pro_tp8_profile_mtp_perf_acc_base1_image.yaml` | MTP Torch Profiler example with a small default workload. |

## Known Good Results

These results were measured on the local MI355X TP8 setup.

| Recipe | Metric | Result |
| --- | --- | ---: |
| No-MTP `perf_acc_base1`, GSM8K 20-shot | flexible exact_match | 0.9492 ± 0.0060 |
| No-MTP `perf_acc_base1`, GSM8K 20-shot | strict exact_match | 0.9484 ± 0.0061 |
| MTP `perf_acc_base1`, GSM8K 20-shot | flexible exact_match | 0.9553 ± 0.0057 |
| MTP `perf_acc_base1`, GSM8K 20-shot | strict exact_match | 0.9560 ± 0.0056 |
| MTP `perf_acc_base1`, 1K/1K conc64 | total tok/s/GPU | 438.01 |
| MTP `perf_acc_base1`, 1K/1K conc64 | output tok/s/GPU | 218.96 |

## Output

Results are written under:

```text
vllm_bench/runs/run_<timestamp>/
├── suite_summary_report.txt
└── case_01_conc64_isl1024_osl1024_np640/
    ├── <result>.json
    ├── agg_<result>.json
    ├── meta_<result>.json
    ├── server_<result>.log
    ├── run_<result>.log
    ├── status.txt
    └── gpu_pids_*.log
```

Eval mode writes:

```text
vllm_bench/runs/run_<timestamp>/
└── eval_gsm8k_fewshot20/
    ├── eval_<result>.log
    ├── eval_summary.txt
    ├── meta_<result>.json
    ├── results_gsm8k_fewshot20/
    └── server_<result>.log
```

Chat mode writes:

```text
vllm_bench/runs/run_<timestamp>/chat/
├── chat_log.txt
├── chat_result.json
└── server_<result>.log
```

Longform mode writes:

```text
vllm_bench/runs/run_<timestamp>/longform/
├── longform_results.txt
├── longform_results.json
└── server_<result>.log
```

Multiturn mode writes:

```text
vllm_bench/runs/run_<timestamp>/multiturn/
├── accuracy_multiturn_multiturn.txt
├── multiturn_results.json
├── _multiturn_turns.json
└── server_<result>.log
```

Profile mode writes:

```text
vllm_bench/runs/run_<timestamp>/profile_01_conc8_isl1024_osl5_np8/
├── traces/
│   └── *.trace.json.gz
├── <result>.json
├── agg_<result>.json
├── meta_<result>.json
├── run_<result>.log
├── server_<result>.log
└── status.txt
```

Profile mode sets `VLLM_TORCH_PROFILER_DIR` for the `vllm serve` process and
uses the benchmark client's `--profile` flag to call `/start_profile` and
`/stop_profile`. Trace files can be opened directly in
[Perfetto](https://ui.perfetto.dev/). Keep `osl` and `num_prompts` small for
first runs because traces can grow quickly.

## Cleanup

For each case, `vllm_bench` terminates vLLM server, engine, worker, and Python
resource tracker processes that match configured cleanup patterns. Containers
created from `image` are removed at the end of the run. Existing containers are
never removed.

## Troubleshooting

- If `lm_eval` is missing in image mode, add a `post_start_commands` entry to
  install it before server startup.
- If eval returns 400 errors about context length, increase `--max-model-len` or
  reduce `eval_num_fewshot`/`eval_max_gen_toks`.
- If ROCm device detection fails, ensure `/dev/kfd`, `/dev/dri`, `group_add:
  ["video"]`, `SYS_PTRACE`, `seccomp=unconfined`, `privileged`, and `ipc=host`
  are present in the container config.
- If a run is interrupted, check for orphaned `VLLM::Worker_TP` processes or KFD
  PIDs and clean them before rerunning.

