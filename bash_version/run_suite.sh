#!/usr/bin/env bash
# =============================================================================
# run_suite.sh
#
# 配置驱动的 LLM 推理测试套件入口。
# 支持三种模式：benchmark（性能压测）、chat（交互对话）、eval（精度评测）。
#
# 使用方法:
#   CONFIG_FILE=./config.sh bash run_suite.sh
#   RUN_MODE=eval CONFIG_FILE=./config.sh bash run_suite.sh
#   RUN_MODE=chat CONFIG_FILE=./config.sh bash run_suite.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-llm_bench_$(whoami 2>/dev/null || echo "uid$(id -u)")_$$}"

# ─────────────────────────────────────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────────────────────────────────────
source_config() {
  local config="${CONFIG_FILE:-${SCRIPT_DIR}/config.sh}"
  if [[ ! -f "$config" ]]; then
    echo "Config file not found: $config" >&2
    exit 1
  fi
  local _cli_run_mode="${RUN_MODE:-}"
  local _cli_chat_prompt="${CHAT_PROMPT:-}"
  local _cli_chat_max_tokens="${CHAT_MAX_TOKENS:-}"

  # shellcheck source=/dev/null
  source "$config"

  [[ -n "$_cli_run_mode" ]] && RUN_MODE="$_cli_run_mode"
  [[ -n "$_cli_chat_prompt" ]] && CHAT_PROMPT="$_cli_chat_prompt"
  [[ -n "$_cli_chat_max_tokens" ]] && CHAT_MAX_TOKENS="$_cli_chat_max_tokens"

  # 可选数组默认值
  declare -p CONTAINER_ENV_OVERRIDES &>/dev/null || CONTAINER_ENV_OVERRIDES=()
  declare -p EXTRA_CONTAINER_MOUNTS &>/dev/null  || EXTRA_CONTAINER_MOUNTS=()
  declare -p EXTRA_DOCKER_ARGS &>/dev/null       || EXTRA_DOCKER_ARGS=()
  declare -p SERVER_ARGS &>/dev/null             || SERVER_ARGS=()
  declare -p POST_START_COMMANDS &>/dev/null     || POST_START_COMMANDS=()

  # 标量默认值
  : "${RUN_MODE:=benchmark}"
  : "${PRECISION:=bf16}"
  : "${RUNNER_TYPE:=mi308x}"
  : "${FRAMEWORK:=sglang}"
  : "${RANDOM_RANGE_RATIO:=1.0}"
  : "${REQUEST_RATE:=inf}"
  : "${BURSTINESS:=1.0}"
  : "${PORT:=8888}"
  : "${BENCH_BACKEND:=vllm}"
  : "${EVAL_TASKS:=gsm8k}"
  : "${EVAL_NUM_FEWSHOT:=5}"
  : "${EVAL_MAX_GEN_TOKS:=2048}"
  : "${EVAL_NUM_CONCURRENT:=224}"
  : "${EVAL_BATCH_SIZE:=auto}"
  : "${HEALTH_TIMEOUT:=240}"
  : "${ENABLE_THINKING:=false}"
  : "${CHAT_STREAM:=true}"
  : "${ENABLE_TUNED_GEMM:=0}"
  : "${TUNED_GEMM_CSV:=}"
  : "${AITER_CSV:=/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_tuned_gemm.csv}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 校验
# ─────────────────────────────────────────────────────────────────────────────
validate_config() {
  local missing=()
  [[ -z "${IMAGE:-}" ]]               && missing+=("IMAGE")
  [[ -z "${MODEL_PATH:-}" ]]          && missing+=("MODEL_PATH")
  [[ -z "${MODEL_PREFIX:-}" ]]        && missing+=("MODEL_PREFIX")
  [[ -z "${HOST_MODEL_MOUNT_PATH:-}" ]] && missing+=("HOST_MODEL_MOUNT_PATH")

  if (( ${#missing[@]} > 0 )); then
    echo "Missing required config variables: ${missing[*]}" >&2
    exit 1
  fi

  if [[ ! -d "$HOST_MODEL_MOUNT_PATH" ]]; then
    echo "HOST_MODEL_MOUNT_PATH not found: $HOST_MODEL_MOUNT_PATH" >&2
    exit 1
  fi

  case "$RUN_MODE" in
    benchmark|chat|eval|longform|multiturn) ;;
    *)
      echo "Unsupported RUN_MODE: ${RUN_MODE}. Supported: benchmark, chat, eval, longform, multiturn." >&2
      exit 1
      ;;
  esac

  if [[ "$ENABLE_TUNED_GEMM" == "1" && ! -f "$TUNED_GEMM_CSV" ]]; then
    echo "ENABLE_TUNED_GEMM=1 but TUNED_GEMM_CSV not found: $TUNED_GEMM_CSV" >&2
    exit 1
  fi
}

validate_test_configs() {
  # 只有 benchmark 模式需要 TEST_CONFIGS
  if [[ "$RUN_MODE" != "benchmark" ]]; then return 0; fi

  if (( ${#TEST_CONFIGS[@]} == 0 )); then
    echo "TEST_CONFIGS is empty." >&2
    exit 1
  fi

  local cfg
  local -a fields
  for cfg in "${TEST_CONFIGS[@]}"; do
    read -r -a fields <<< "$cfg"
    if (( ${#fields[@]} != 4 )); then
      echo "Invalid TEST_CONFIGS entry: '$cfg'" >&2
      echo "Expected 4 fields: concurrency ISL OSL num_prompts" >&2
      exit 1
    fi
  done
}

validate_dependencies() {
  command -v docker >/dev/null 2>&1 || { echo "docker not found" >&2; exit 1; }

  local required_files=(
    "${SCRIPT_DIR}/run_case.sh"
    "${SCRIPT_DIR}/summarize_results.sh"
    "${SCRIPT_DIR}/utils/bench_serving/benchmark_serving.py"
    "${SCRIPT_DIR}/utils/bench_serving/backend_request_func.py"
    "${SCRIPT_DIR}/utils/bench_serving/benchmark_utils.py"
    "${SCRIPT_DIR}/utils/process_result.py"
  )
  local f
  for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "Required file not found: $f" >&2
      exit 1
    fi
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# Docker 容器管理
# ─────────────────────────────────────────────────────────────────────────────
cleanup_container() {
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}" 2>/dev/null; then
    docker exec "${CONTAINER_NAME}" pkill -f sglang.launch_server >/dev/null 2>&1 || true
    sleep 2
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
}

# 清理当前用户历史崩溃遗留的 llm_bench 容器（仅清理父进程已不存在的孤儿容器）
cleanup_stale_containers() {
  local user_prefix="llm_bench_$(whoami 2>/dev/null || echo "uid$(id -u)")"
  local candidates
  candidates=$(docker ps -a --format '{{.Names}}' | grep "^${user_prefix}_" | grep -Fxv "${CONTAINER_NAME}" || true)
  if [[ -z "$candidates" ]]; then return 0; fi

  local name pid_part
  while IFS= read -r name; do
    # 容器名格式: llm_bench_<user>_<pid>，提取 PID 部分
    pid_part="${name##*_}"
    # 仅当对应的父进程已不存在时才清理（避免误杀并行运行的实例）
    if [[ "$pid_part" =~ ^[0-9]+$ ]] && ! kill -0 "$pid_part" 2>/dev/null; then
      echo ">>> Removing stale container (PID $pid_part no longer exists): $name"
      docker rm -f "$name" >/dev/null 2>&1 || true
    fi
  done <<< "$candidates"
}

start_container() {
  local -a docker_cmd=(
    docker run -it -d
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
    --user root --device=/dev/kfd --device=/dev/dri
    --group-add video --ipc=host --pid=host --network host --privileged
    -e "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    -e "HF_HOME=/.cache/huggingface/"
    --name "${CONTAINER_NAME}"
    --mount "type=bind,src=${HOST_MODEL_MOUNT_PATH},dst=/.cache/huggingface/"
    --mount "type=bind,src=${HOST_MODEL_MOUNT_PATH},dst=${HOST_MODEL_MOUNT_PATH}"
    --mount "type=bind,src=${SCRIPT_DIR},dst=/simple-suite"
    --mount "type=bind,src=${RUN_DIR},dst=/simple-suite-output"
    --mount "type=bind,src=/dev/shm,dst=/dev/shm"
  )

  local mount_spec
  for mount_spec in "${EXTRA_CONTAINER_MOUNTS[@]}"; do
    local src dst opts
    IFS=':' read -r src dst opts <<< "$mount_spec"
    if [[ -n "$opts" ]]; then
      docker_cmd+=(--mount "type=bind,src=${src},dst=${dst},${opts}")
    else
      docker_cmd+=(--mount "type=bind,src=${src},dst=${dst}")
    fi
  done

  docker_cmd+=("${EXTRA_DOCKER_ARGS[@]}")
  docker_cmd+=(-t "${IMAGE}")

  "${docker_cmd[@]}" >/dev/null

  local cmd
  for cmd in "${POST_START_COMMANDS[@]}"; do
    docker exec "${CONTAINER_NAME}" bash -c "$cmd"
  done
}

inject_tuned_gemm_if_needed() {
  if [[ "${ENABLE_TUNED_GEMM}" != "1" ]]; then return 0; fi
  local container_csv="/simple-suite/$(basename "${TUNED_GEMM_CSV}")"
  echo ">>> Backing up container aiter CSV ..."
  docker exec "${CONTAINER_NAME}" cp "${AITER_CSV}" "${AITER_CSV}.bak"
  local before after
  before=$(docker exec "${CONTAINER_NAME}" bash -c "wc -l < ${AITER_CSV}")
  echo ">>> Injecting tuned GEMM config: $(basename "${TUNED_GEMM_CSV}") ..."
  docker exec "${CONTAINER_NAME}" bash -c "tail -n +2 '${container_csv}' >> ${AITER_CSV}"
  after=$(docker exec "${CONTAINER_NAME}" bash -c "wc -l < ${AITER_CSV}")
  echo ">>> Injection done: ${before} -> ${after} lines"
}

# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────
serialize_server_args() {
  local IFS=$'\x1e'
  echo "${SERVER_ARGS[*]}"
}

build_case_name() {
  local case_id="$1" conc="$2" isl="$3" osl="$4" np="$5"
  printf 'case_%s_conc%s_isl%s_osl%s_np%s' \
    "$case_id" "$conc" "$isl" "$osl" "$np"
}

# 构建 server 启动命令字符串（所有模式共用）
# 对包含特殊字符的参数做 shell 转义，确保通过 bash -c 传递时不被破坏。
build_server_cmd() {
  local cmd="python3 -m sglang.launch_server --model-path ${MODEL_PATH} --host 0.0.0.0 --port ${PORT}"
  for arg in "${SERVER_ARGS[@]}"; do
    cmd+=" $(printf '%q' "$arg")"
  done
  echo "$cmd"
}

# 等待 server 健康（同时实时显示 server 日志）
# 包含稳定性确认（连续多次 healthy）和容器存活检测
wait_for_health() {
  local timeout="${1:-$HEALTH_TIMEOUT}"
  local stable_required=3  # 需要连续 3 次 healthy 才认为稳定
  local stable_count=0
  echo ">>> Waiting for server to become healthy..."

  # 后台 tail server 日志，实时显示启动过程
  # 用子 shell + exec 避免 set -e 下 kill 后台进程导致脚本退出
  ( docker exec "${CONTAINER_NAME}" bash -c "tail -f /tmp/server.log 2>/dev/null" ) &
  local tail_pid=$!

  _stop_tail() {
    kill "$tail_pid" 2>/dev/null
    wait "$tail_pid" 2>/dev/null
    return 0
  }

  for i in $(seq 1 "$timeout"); do
    # 检查容器是否还存在
    if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}" 2>/dev/null; then
      _stop_tail || true
      echo ""
      echo ">>> Container '${CONTAINER_NAME}' no longer running. Server may have crashed." >&2
      return 1
    fi

    if docker exec "${CONTAINER_NAME}" curl -s -f "http://0.0.0.0:${PORT}/health" >/dev/null 2>&1; then
      stable_count=$((stable_count + 1))
      if (( stable_count >= stable_required )); then
        _stop_tail || true
        echo ""
        echo ">>> Server is ready! (stable after $((i * 5))s, ${stable_count} consecutive checks)"
        return 0
      fi
    else
      # 不健康时重置稳定计数
      if (( stable_count > 0 )); then
        echo ">>> Server health unstable (was healthy ${stable_count}x, then failed). Resetting..."
      fi
      stable_count=0
    fi

    if (( i == timeout )); then
      _stop_tail || true
      echo ""
      echo ">>> Server failed to start within $((timeout * 5))s" >&2
      return 1
    fi
    sleep 5
  done
}

# 构建容器环境变量参数
build_env_args() {
  local -n _env_args=$1
  for env_spec in "${CONTAINER_ENV_OVERRIDES[@]}"; do
    _env_args+=(-e "$env_spec")
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
source_config
validate_config
validate_test_configs
validate_dependencies

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/runs/run_${RUN_TIMESTAMP}"
mkdir -p "${RUN_DIR}"

cleanup_stale_containers
trap cleanup_container EXIT

SERIALIZED_SERVER_ARGS="$(serialize_server_args)"

echo "============================================================"
echo "  sgl_benchmark"
echo "============================================================"
echo "Run dir       : ${RUN_DIR}"
echo "Image         : ${IMAGE}"
echo "Model         : ${MODEL_PATH}"
echo "RUN_MODE      : ${RUN_MODE}"
echo "SERVER_ARGS   : ${SERVER_ARGS[*]}"
if [[ "$RUN_MODE" == "benchmark" ]]; then
  echo "BENCH_BACKEND : ${BENCH_BACKEND}"
  echo "Test cases    : ${#TEST_CONFIGS[@]}"
fi
echo "============================================================"

case "$RUN_MODE" in
  benchmark)
    for idx in "${!TEST_CONFIGS[@]}"; do
      cfg="${TEST_CONFIGS[$idx]}"
      read -r conc isl osl num_prompts <<< "$cfg"

      case_id="$(printf '%02d' "$((idx + 1))")"
      case_name="$(build_case_name "$case_id" "$conc" "$isl" "$osl" "$num_prompts")"
      case_host_dir="${RUN_DIR}/${case_name}"
      mkdir -p "$case_host_dir"

      echo ""
      echo ">>> [${case_id}/${#TEST_CONFIGS[@]}] ${case_name}"
      echo ">>> CONC=${conc} ISL=${isl} OSL=${osl} NP=${num_prompts}"

      cleanup_container
      start_container
      inject_tuned_gemm_if_needed

      local_env_args=(
        -e "CASE_NAME=${case_name}"
        -e "CASE_OUTPUT_DIR=/simple-suite-output/${case_name}"
        -e "MODEL_PATH=${MODEL_PATH}"
        -e "MODEL_PREFIX=${MODEL_PREFIX}"
        -e "IMAGE=${IMAGE}"
        -e "PRECISION=${PRECISION}"
        -e "RUNNER_TYPE=${RUNNER_TYPE}"
        -e "FRAMEWORK=${FRAMEWORK}"
        -e "RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO}"
        -e "REQUEST_RATE=${REQUEST_RATE}"
        -e "BURSTINESS=${BURSTINESS}"
        -e "PORT=${PORT}"
        -e "CONC=${conc}"
        -e "ISL=${isl}"
        -e "OSL=${osl}"
        -e "NUM_PROMPTS=${num_prompts}"
        -e "SERVER_ARGS_SERIALIZED=${SERIALIZED_SERVER_ARGS}"
        -e "BENCH_BACKEND=${BENCH_BACKEND}"
      )

      for env_spec in "${CONTAINER_ENV_OVERRIDES[@]}"; do
        local_env_args+=(-e "$env_spec")
      done

      docker exec \
        -w /simple-suite \
        "${local_env_args[@]}" \
        "${CONTAINER_NAME}" \
        bash /simple-suite/run_case.sh

      cleanup_container
    done

    bash "${SCRIPT_DIR}/summarize_results.sh" "${RUN_DIR}"

    echo ""
    echo ">>> All cases finished"
    echo ">>> Summary report: ${RUN_DIR}/suite_summary_report.txt"
    ;;

  chat)
    echo ""
    echo ">>> Chat mode"

    cleanup_container
    start_container
    inject_tuned_gemm_if_needed

    env_args=()
    build_env_args env_args

    WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-600}"
    SERVER_CMD="$(build_server_cmd) --skip-server-warmup --watchdog-timeout ${WATCHDOG_TIMEOUT} --soft-watchdog-timeout ${WATCHDOG_TIMEOUT}"
    echo ">>> Starting server: ${SERVER_CMD}"
    docker exec -d "${env_args[@]}" "${CONTAINER_NAME}" bash -c "${SERVER_CMD} > /tmp/server.log 2>&1"

    wait_for_health 120

    send_chat() {
      local prompt="$1"
      local max_tokens="${CHAT_MAX_TOKENS:-8192}"
      local escaped_prompt
      escaped_prompt=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.buffer.read().decode('utf-8').rstrip('\n')))" <<< "$prompt")

      if [[ "${CHAT_STREAM}" == "true" ]]; then
        # 流式输出：逐 token 打印到终端 + 同时写日志文件
        # 不走 tee 管道，在 Python 内部直接写两路，避免缓冲破坏流式体验
        docker exec "${env_args[@]}" "${CONTAINER_NAME}" \
          curl -sN "http://0.0.0.0:${PORT}/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d "{\"model\":\"${MODEL_PATH}\",\"messages\":[{\"role\":\"user\",\"content\":${escaped_prompt}}],\"max_tokens\":${max_tokens},\"stream\":true,\"chat_template_kwargs\":{\"enable_thinking\":${ENABLE_THINKING}}}" \
        | python3 -c "
import sys, json, time, os

log_file = os.environ.get('CHAT_LOG', '')
log_fh = open(log_file, 'a', encoding='utf-8') if log_file else None

t_start = time.time()
t_first_token = None
n_tokens = 0

for line in sys.stdin:
    line = line.strip()
    if not line or not line.startswith('data: '):
        continue
    payload = line[6:]
    if payload == '[DONE]':
        break
    try:
        chunk = json.loads(payload)
        delta = chunk['choices'][0].get('delta', {})
        text = delta.get('content', '')
        if text:
            if t_first_token is None:
                t_first_token = time.time()
            n_tokens += 1
            sys.stdout.write(text)
            sys.stdout.flush()
            if log_fh:
                log_fh.write(text)
    except Exception:
        pass

t_end = time.time()
print()
if log_fh:
    log_fh.write('\n')

# 性能统计
if n_tokens > 0 and t_first_token is not None:
    ttft_ms = (t_first_token - t_start) * 1000
    total_ms = (t_end - t_start) * 1000
    if n_tokens > 1:
        tpot_ms = (t_end - t_first_token) * 1000 / (n_tokens - 1)
        tps = (n_tokens - 1) / (t_end - t_first_token)
    else:
        tpot_ms = 0
        tps = 0
    stats = f'[tokens: {n_tokens} | TTFT: {ttft_ms:.0f}ms | TPOT: {tpot_ms:.1f}ms | {tps:.1f} tok/s | total: {total_ms:.0f}ms]'
    print(stats)
    if log_fh:
        log_fh.write(stats + '\n')

if log_fh:
    log_fh.close()
"
      else
        # 非流式输出：等待完整响应 + 性能统计
        docker exec "${env_args[@]}" "${CONTAINER_NAME}" \
          curl -s "http://0.0.0.0:${PORT}/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d "{\"model\":\"${MODEL_PATH}\",\"messages\":[{\"role\":\"user\",\"content\":${escaped_prompt}}],\"max_tokens\":${max_tokens},\"chat_template_kwargs\":{\"enable_thinking\":${ENABLE_THINKING}}}" \
        | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    content = data['choices'][0]['message']['content']
    usage = data.get('usage', {})
    print(content)
    prompt_tokens = usage.get('prompt_tokens', '?')
    completion_tokens = usage.get('completion_tokens', '?')
    print(f'[tokens: prompt={prompt_tokens}, completion={completion_tokens}]')
except Exception as e:
    print(f'[Error] {e}', file=sys.stderr)
"
      fi
    }

    CHAT_LOG="${RUN_DIR}/chat_log.txt"
    export CHAT_LOG
    {
      echo "========== Chat Session =========="
      echo "Config: ${CONFIG_FILE}"
      echo "Model: ${MODEL_PATH}"
      echo "Date: $(date)"
      echo "Stream: ${CHAT_STREAM}"
      echo "Enable thinking: ${ENABLE_THINKING}"
      echo "=================================="
      echo ""
    } > "$CHAT_LOG"

    # 辅助函数：同时输出到终端和日志
    _log() { echo "$*" | tee -a "$CHAT_LOG"; }

    if [[ -n "${CHAT_PROMPT:-}" ]]; then
      _log ""
      _log ">>> Prompt: ${CHAT_PROMPT}"
      _log "------------------------------------------------------------"
      # 流式模式下 Python 内部直接写 CHAT_LOG，非流式走 tee
      if [[ "${CHAT_STREAM}" == "true" ]]; then
        send_chat "$CHAT_PROMPT"
      else
        send_chat "$CHAT_PROMPT" 2>&1 | tee -a "$CHAT_LOG"
      fi
      _log "------------------------------------------------------------"
    else
      _log ""
      _log ">>> Interactive chat mode. Type 'quit' or 'exit' to stop."
      _log "------------------------------------------------------------"
      while true; do
        printf "\n[You] "
        read -r user_input
        [[ "$user_input" == "quit" || "$user_input" == "exit" ]] && break
        [[ -z "$user_input" ]] && continue
        echo "" >> "$CHAT_LOG"
        echo "[You] ${user_input}" >> "$CHAT_LOG"
        echo ""
        echo "[Model]"
        echo "[Model]" >> "$CHAT_LOG"
        if [[ "${CHAT_STREAM}" == "true" ]]; then
          send_chat "$user_input"
        else
          send_chat "$user_input" 2>&1 | tee -a "$CHAT_LOG"
        fi
      done
    fi

    _log ""
    _log ">>> Chat session ended."
    echo ">>> Chat log saved to: ${CHAT_LOG}"
    ;;

  eval)
    echo ""
    echo ">>> Eval mode: tasks=${EVAL_TASKS} fewshot=${EVAL_NUM_FEWSHOT}"

    cleanup_container
    start_container
    inject_tuned_gemm_if_needed

    env_args=()
    build_env_args env_args

    # eval 模式需要更长的 watchdog timeout，因为首次请求可能触发 JIT kernel 编译（耗时 60s+）
    WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-600}"
    SERVER_CMD="$(build_server_cmd) --watchdog-timeout ${WATCHDOG_TIMEOUT} --soft-watchdog-timeout ${WATCHDOG_TIMEOUT}"
    echo ">>> Starting server: ${SERVER_CMD}"
    docker exec -d "${env_args[@]}" "${CONTAINER_NAME}" bash -c "${SERVER_CMD} > /tmp/server.log 2>&1"

    wait_for_health "$HEALTH_TIMEOUT"

    EVAL_MODEL_ARGS="{\"base_url\": \"http://localhost:${PORT}/v1/completions\", \"model\": \"${MODEL_PATH}\", \"num_concurrent\": ${EVAL_NUM_CONCURRENT}, \"max_retries\": 10, \"max_gen_toks\": ${EVAL_MAX_GEN_TOKS}}"

    EVAL_CMD=(
      python3 -m lm_eval
      --model local-completions
      --model_args "'${EVAL_MODEL_ARGS}'"
      --tasks "$EVAL_TASKS"
      --batch_size "$EVAL_BATCH_SIZE"
      --num_fewshot "$EVAL_NUM_FEWSHOT"
      --trust_remote_code
    )

    EVAL_OUTPUT_DIR="${RUN_DIR}/eval_${EVAL_TASKS}_fewshot${EVAL_NUM_FEWSHOT}"
    EVAL_CMD+=(--output_path "$EVAL_OUTPUT_DIR")

    echo ">>> Running: ${EVAL_CMD[*]}"
    docker exec \
      "${env_args[@]}" \
      "${CONTAINER_NAME}" \
      bash -c "${EVAL_CMD[*]}" 2>&1 | tee "${RUN_DIR}/lm_eval.log"

    echo ""
    echo ">>> Eval finished."
    echo ">>> Results: ${RUN_DIR}/lm_eval.log"
    echo ""
    echo ">>> Results summary:"
    grep -E '^\|' "${RUN_DIR}/lm_eval.log" | tail -4
    ;;

  longform)
    echo ""
    echo ">>> Long-form generation test"

    cleanup_container
    start_container
    inject_tuned_gemm_if_needed

    env_args=()
    build_env_args env_args

    WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-600}"
    SERVER_CMD="$(build_server_cmd) --skip-server-warmup --watchdog-timeout ${WATCHDOG_TIMEOUT} --soft-watchdog-timeout ${WATCHDOG_TIMEOUT}"
    echo ">>> Starting server: ${SERVER_CMD}"
    docker exec -d "${env_args[@]}" "${CONTAINER_NAME}" bash -c "${SERVER_CMD} > /tmp/server.log 2>&1"

    wait_for_health "$HEALTH_TIMEOUT"

    # 发送 chat 请求的辅助函数
    _send_longform() {
      local messages_json="$1"
      local max_tokens="${2:-8192}"
      docker exec "${env_args[@]}" "${CONTAINER_NAME}" \
        curl -s "http://0.0.0.0:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL_PATH}\",\"messages\":${messages_json},\"max_tokens\":${max_tokens},\"chat_template_kwargs\":{\"enable_thinking\":${ENABLE_THINKING}}}" \
      | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    content = data['choices'][0]['message']['content']
    usage = data.get('usage', {})
    print(f'[tokens: prompt={usage.get(\"prompt_tokens\",\"?\")}, completion={usage.get(\"completion_tokens\",\"?\")}]')
    print(content)
except Exception as e:
    print(f'[Error] {e}', file=sys.stderr)
"
    }

    OUT_FILE="${RUN_DIR}/longform_results.txt"
    echo "========== Long-form Accuracy Test ==========" > "$OUT_FILE"
    echo "Config: ${CONFIG_FILE}" >> "$OUT_FILE"
    echo "Date: $(date)" >> "$OUT_FILE"
    echo "" >> "$OUT_FILE"

    # LONGFORM_PROMPTS 必须在配置文件中定义
    if ! declare -p LONGFORM_PROMPTS &>/dev/null || (( ${#LONGFORM_PROMPTS[@]} == 0 )); then
      echo "ERROR: LONGFORM_PROMPTS array is not set in config file." >&2
      echo "Example:" >&2
      echo '  LONGFORM_PROMPTS=(' >&2
      echo '      "请写一篇关于XXX的深度分析文章..."' >&2
      echo '      "Write a detailed tutorial about..."' >&2
      echo '  )' >&2
      exit 1
    fi
    PROMPTS=("${LONGFORM_PROMPTS[@]}")

    for i in "${!PROMPTS[@]}"; do
      echo ""
      echo ">>> Test $((i+1))/${#PROMPTS[@]}: ${PROMPTS[$i]:0:60}..."
      echo "================================================================" >> "$OUT_FILE"
      echo "TEST $((i+1)): ${PROMPTS[$i]:0:80}..." >> "$OUT_FILE"
      echo "================================================================" >> "$OUT_FILE"

      escaped=$(printf '%s' "${PROMPTS[$i]}" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
      messages_json="[{\"role\":\"user\",\"content\":${escaped}}]"

      echo "Prompt: ${PROMPTS[$i]}" >> "$OUT_FILE"
      echo "---" >> "$OUT_FILE"
      response=$(_send_longform "$messages_json" 8192)
      echo "$response" >> "$OUT_FILE"
      echo "" >> "$OUT_FILE"
      echo "$response" | head -3
      echo "  ... ($(echo "$response" | wc -c) chars total)"
    done

    echo ""
    echo ">>> Long-form test finished."
    echo ">>> Results: ${OUT_FILE}"
    ;;

  multiturn)
    echo ""
    echo ">>> Multi-turn conversation test"

    cleanup_container
    start_container
    inject_tuned_gemm_if_needed

    env_args=()
    build_env_args env_args

    WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-600}"
    SERVER_CMD="$(build_server_cmd) --skip-server-warmup --watchdog-timeout ${WATCHDOG_TIMEOUT} --soft-watchdog-timeout ${WATCHDOG_TIMEOUT}"
    echo ">>> Starting server: ${SERVER_CMD}"
    docker exec -d "${env_args[@]}" "${CONTAINER_NAME}" bash -c "${SERVER_CMD} > /tmp/server.log 2>&1"

    wait_for_health "$HEALTH_TIMEOUT"

    # 支持两种方式指定对话轮次：
    #   1. MULTITURN_TURNS 数组（直接在配置文件中定义每轮内容）
    #   2. MULTITURN_TURNS_FILE（指向外部 JSON 文件）
    # 优先使用 MULTITURN_TURNS 数组
    TURNS_FILE=""
    if declare -p MULTITURN_TURNS &>/dev/null && (( ${#MULTITURN_TURNS[@]} > 0 )); then
      # 从数组生成临时 JSON 文件
      TURNS_FILE="${RUN_DIR}/_multiturn_turns.json"
      python3 -c "
import json, sys
turns = []
for line in sys.stdin:
    line = line.strip()
    if line:
        turns.append({'user': line})
with open('${TURNS_FILE}', 'w', encoding='utf-8') as f:
    json.dump(turns, f, ensure_ascii=False, indent=2)
print(f'Generated {len(turns)} turns -> ${TURNS_FILE}')
" <<EOF
$(printf '%s\n' "${MULTITURN_TURNS[@]}")
EOF
    elif [[ -n "${MULTITURN_TURNS_FILE:-}" ]]; then
      if [[ ! -f "${MULTITURN_TURNS_FILE}" ]]; then
        echo "ERROR: MULTITURN_TURNS_FILE not found: ${MULTITURN_TURNS_FILE}" >&2
        exit 1
      fi
      TURNS_FILE="${MULTITURN_TURNS_FILE}"
    else
      echo "ERROR: Neither MULTITURN_TURNS array nor MULTITURN_TURNS_FILE is set." >&2
      echo 'Set MULTITURN_TURNS in config:' >&2
      echo '  MULTITURN_TURNS=(' >&2
      echo '      "第一轮用户输入"' >&2
      echo '      "第二轮用户输入"' >&2
      echo '  )' >&2
      echo 'Or set MULTITURN_TURNS_FILE="/path/to/turns.json"' >&2
      exit 1
    fi

    MULTITURN_CMD=(
      python3 "${SCRIPT_DIR}/accuracy_multiturn_test.py"
      --port "${PORT}"
      --label "multiturn"
      --model "${MODEL_PATH}"
      --out-dir "${RUN_DIR}"
      --turns-file "${TURNS_FILE}"
    )
    if [[ "${ENABLE_THINKING}" == "true" ]]; then
      MULTITURN_CMD+=(--enable-thinking)
    fi

    "${MULTITURN_CMD[@]}" 2>&1 | tee "${RUN_DIR}/multiturn.log"

    echo ""
    echo ">>> Multi-turn test finished."
    echo ">>> Results: ${RUN_DIR}/"
    ;;

  *)
    echo "Unsupported RUN_MODE: ${RUN_MODE}" >&2
    exit 1
    ;;
esac
