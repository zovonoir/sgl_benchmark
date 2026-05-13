#!/usr/bin/env bash
# =============================================================================
# run_case.sh
#
# 容器内执行的单个测试用例脚本。由 run_suite.sh 通过 docker exec 调用。
# 所有参数通过环境变量传入。
#
# 流程:
#   1. 解析环境变量和 SERVER_ARGS
#   2. 启动 GPU 监控
#   3. 启动 sglang server
#   4. 等待 server 健康
#   5. 执行 benchmark_serving
#   6. 生成 meta JSON 和聚合结果
# =============================================================================
set -euo pipefail

CASE_OUTPUT_DIR="${CASE_OUTPUT_DIR:?CASE_OUTPUT_DIR is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
IMAGE="${IMAGE:?IMAGE is required}"

PRECISION="${PRECISION:-bf16}"
MODEL_PREFIX="${MODEL_PREFIX:-$(basename "${MODEL_PATH}")}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
BURSTINESS="${BURSTINESS:-1.0}"
RUNNER_TYPE="${RUNNER_TYPE:-mi308x}"
FRAMEWORK="${FRAMEWORK:-sglang}"
SPEC_DECODING="${SPEC_DECODING:-none}"
DISAGG="${DISAGG:-false}"
DP_ATTENTION="${DP_ATTENTION:-false}"
PORT="${PORT:-8888}"
BENCH_BACKEND="${BENCH_BACKEND:-vllm}"

CONC="${CONC:?CONC is required}"
ISL="${ISL:?ISL is required}"
OSL="${OSL:?OSL is required}"
NUM_PROMPTS="${NUM_PROMPTS:?NUM_PROMPTS is required}"
CASE_NAME="${CASE_NAME:-case}"

mkdir -p "$CASE_OUTPUT_DIR"
cd "$CASE_OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 反序列化 SERVER_ARGS
# ─────────────────────────────────────────────────────────────────────────────
USER_SERVER_ARGS=()
if [[ -n "${SERVER_ARGS_SERIALIZED:-}" ]]; then
  IFS=$'\x1e' read -r -a USER_SERVER_ARGS <<< "$SERVER_ARGS_SERIALIZED"
fi

# ─────────────────────────────────────────────────────────────────────────────
# GPU 监控
# ─────────────────────────────────────────────────────────────────────────────
GPU_MONITOR_PID=""
GPU_METRICS_CSV="$CASE_OUTPUT_DIR/gpu_metrics.csv"

start_gpu_monitor() {
  if command -v amd-smi >/dev/null 2>&1; then
    amd-smi metric -p -c -t -u -w 1 --csv 2>/dev/null \
      | awk '/^timestamp,/{if(!h){print;h=1};next} h{print}' > "$GPU_METRICS_CSV" &
    GPU_MONITOR_PID=$!
    echo "[gpu] Started AMD monitor -> $GPU_METRICS_CSV"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory \
      --format=csv -l 1 > "$GPU_METRICS_CSV" 2>/dev/null &
    GPU_MONITOR_PID=$!
    echo "[gpu] Started NVIDIA monitor -> $GPU_METRICS_CSV"
    return 0
  fi

  echo "[gpu] No GPU monitor tool found, skip."
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Server 管理
# ─────────────────────────────────────────────────────────────────────────────
wait_for_server_ready() {
  local server_pid="$1"
  local server_log="$2"

  while [[ ! -f "$server_log" ]]; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[server] Server exited before log file was created." >&2
      return 1
    fi
    sleep 1
  done

  tail -f -n +1 "$server_log" &
  local tail_pid=$!

  until curl --output /dev/null --silent --fail "http://0.0.0.0:${PORT}/health"; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[server] Server exited before becoming healthy." >&2
      kill "$tail_pid" 2>/dev/null || true
      wait "$tail_pid" 2>/dev/null || true
      return 1
    fi
    sleep 5
  done

  kill "$tail_pid" 2>/dev/null || true
  wait "$tail_pid" 2>/dev/null || true
}

cleanup_server() {
  local server_pid="${1:-}"
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    echo "[cleanup] Killing server PID=$server_pid"
    kill -- -"$server_pid" 2>/dev/null || kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    sleep 5
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 文件名构建
# ─────────────────────────────────────────────────────────────────────────────
sanitize_tag() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  printf '%s' "$value"
}

MODEL_TAG="$(sanitize_tag "$MODEL_PREFIX")"
RESULT_FILENAME="${CASE_NAME}_${MODEL_TAG}_${PRECISION}_${FRAMEWORK}_conc${CONC}_isl${ISL}_osl${OSL}_np${NUM_PROMPTS}"

SERVER_LOG="$CASE_OUTPUT_DIR/server_${RESULT_FILENAME}.log"
META_JSON="$CASE_OUTPUT_DIR/meta_${RESULT_FILENAME}.json"

export CASE_NAME MODEL_PATH MODEL_PREFIX IMAGE PRECISION FRAMEWORK RUNNER_TYPE
export CONC ISL OSL NUM_PROMPTS RESULT_FILENAME META_JSON
export REQUEST_RATE BURSTINESS

echo "[case] MODEL_PATH=$MODEL_PATH"
echo "[case] CONC=$CONC ISL=$ISL OSL=$OSL NUM_PROMPTS=$NUM_PROMPTS"
echo "[case] REQUEST_RATE=$REQUEST_RATE BURSTINESS=$BURSTINESS"
echo "[case] USER_SERVER_ARGS: ${USER_SERVER_ARGS[*]:-<none>}"
echo "[case] RESULT_FILENAME=$RESULT_FILENAME"

# ─────────────────────────────────────────────────────────────────────────────
# 启动 server
# ─────────────────────────────────────────────────────────────────────────────
start_gpu_monitor

SERVER_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host=0.0.0.0
  --port "$PORT"
)

# 所有 server 参数（包括 --tensor-parallel-size 等）由 SERVER_ARGS 传入
SERVER_CMD+=("${USER_SERVER_ARGS[@]}")

echo "[case] Full server command: ${SERVER_CMD[*]}"

"${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

trap 'stop_gpu_monitor; cleanup_server "${SERVER_PID:-}"' EXIT

wait_for_server_ready "$SERVER_PID" "$SERVER_LOG"

# ─────────────────────────────────────────────────────────────────────────────
# 执行 Benchmark
# ─────────────────────────────────────────────────────────────────────────────
echo "[case] BENCH_BACKEND=$BENCH_BACKEND"

if [[ "$BENCH_BACKEND" == "sglang" ]]; then
  # SGLang 原生 bench_serving（走 /generate 原生接口）
  BENCH_CMD=(
    python3 -m sglang.bench_serving
    --backend sglang
    --port "$PORT"
    --model "$MODEL_PATH"
    --dataset-name random
    --random-input-len "$ISL"
    --random-output-len "$OSL"
    --random-range-ratio "$RANDOM_RANGE_RATIO"
    --num-prompts "$NUM_PROMPTS"
    --max-concurrency "$CONC"
    --request-rate "$REQUEST_RATE"
    --warmup-requests 2
    --output-file "$CASE_OUTPUT_DIR/${RESULT_FILENAME}.json"
  )
else
  # vllm 兼容 benchmark（走 /v1/completions OpenAI 接口）
  BENCH_CMD=(
    python3 /simple-suite/utils/bench_serving/benchmark_serving.py
    --model "$MODEL_PATH"
    --backend vllm
    --base-url "http://0.0.0.0:${PORT}"
    --dataset-name random
    --random-input-len "$ISL"
    --random-output-len "$OSL"
    --random-range-ratio "$RANDOM_RANGE_RATIO"
    --num-prompts "$NUM_PROMPTS"
    --max-concurrency "$CONC"
    --request-rate "$REQUEST_RATE"
    --burstiness "$BURSTINESS"
    --ignore-eos
    --save-result
    --num-warmups 2
    --percentile-metrics ttft,tpot,itl,e2el
    --result-dir "$CASE_OUTPUT_DIR"
    --result-filename "${RESULT_FILENAME}.json"
  )
fi

"${BENCH_CMD[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# 生成 meta JSON（记录本次运行的所有配置，用于汇总报告）
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import json
import os

# 收集所有 SGLANG_ / AITER_ / ATOM_ 开头的环境变量
server_env = {k: v for k, v in sorted(os.environ.items())
              if k.startswith(("SGLANG_", "AITER_", "ATOM_"))}

meta = {
    "case_name": os.environ["CASE_NAME"],
    "model_path": os.environ["MODEL_PATH"],
    "model_prefix": os.environ["MODEL_PREFIX"],
    "image": os.environ["IMAGE"],
    "precision": os.environ["PRECISION"],
    "framework": os.environ["FRAMEWORK"],
    "runner_type": os.environ["RUNNER_TYPE"],
    "concurrency": int(os.environ["CONC"]),
    "isl": int(os.environ["ISL"]),
    "osl": int(os.environ["OSL"]),
    "num_prompts": int(os.environ["NUM_PROMPTS"]),
    "request_rate": os.environ.get("REQUEST_RATE", "inf"),
    "burstiness": os.environ.get("BURSTINESS", "1.0"),
    "result_filename": os.environ["RESULT_FILENAME"],
    "server_env": server_env,
}

with open(os.environ["META_JSON"], "w") as fh:
    json.dump(meta, fh, indent=2)
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
# 聚合结果（process_result.py）
# ─────────────────────────────────────────────────────────────────────────────
(
  export RESULT_FILENAME
  export RUNNER_TYPE
  export FRAMEWORK
  export PRECISION
  export SPEC_DECODING
  export DISAGG
  export MODEL_PREFIX
  export IMAGE
  export TP="${TP:-1}"
  export EP_SIZE="${EP:-1}"
  export DP_ATTENTION
  export ISL
  export OSL
  python3 /simple-suite/utils/process_result.py || {
    echo "[case] WARNING: process_result.py failed (non-fatal), generating minimal agg JSON"
    python3 -c "
import json, os
rf = os.environ['RESULT_FILENAME']
with open(f'{rf}.json') as f: raw = json.load(f)
tp = int(os.environ.get('TP', '1'))
agg = {
    'tput_per_gpu': float(raw.get('total_token_throughput',0)) / max(tp,1),
    'output_tput_per_gpu': float(raw.get('output_throughput',0)) / max(tp,1),
    'input_tput_per_gpu': (float(raw.get('total_token_throughput',0)) - float(raw.get('output_throughput',0))) / max(tp,1),
}
with open(f'agg_{rf}.json','w') as f: json.dump(agg, f, indent=2)
print(json.dumps(agg, indent=2))
"
  }
)

echo "[case] Finished: $CASE_OUTPUT_DIR/${RESULT_FILENAME}.json"
echo "[case] Aggregated: $CASE_OUTPUT_DIR/agg_${RESULT_FILENAME}.json"
