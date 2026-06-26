#!/bin/bash
# 启动 vLLM Disaggregated Prefill-Decode with MooncakeConnector
# 使用 Qwen3-VL 模型

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${PROJECT_DIR}/config"

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
TP_SIZE=1
MAX_MODEL_LEN=4096

echo "======================================"
echo "vLLM Disaggregated EPD Serving"
echo "Model: ${MODEL}"
echo "======================================"

# Activate venv
source "${PROJECT_DIR}/../venv_mooncake/bin/activate"

# 1. Start etcd (if not running)
if ! pgrep -x etcd > /dev/null; then
    echo "[1/4] Starting etcd..."
    etcd --listen-client-urls http://0.0.0.0:2379 \
         --advertise-client-urls http://localhost:2379 \
         &
    sleep 3
else
    echo "[1/4] etcd already running"
fi

# 2. Start Prefill Node
echo "[2/4] Starting Prefill node on port ${PREFILL_PORT}..."
CUDA_VISIBLE_DEVICES=0 \
MOONCAKE_CONFIG_PATH="${CONFIG_DIR}/mooncake.json" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port ${PREFILL_PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' \
    &
PREFILL_PID=$!
echo "Prefill started (PID: $PREFILL_PID)"
sleep 5

# 3. Start Decode Node
echo "[3/4] Starting Decode node on port ${DECODE_PORT}..."
CUDA_VISIBLE_DEVICES=1 \
MOONCAKE_CONFIG_PATH="${CONFIG_DIR}/mooncake.json" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port ${DECODE_PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}' \
    &
DECODE_PID=$!
echo "Decode started (PID: $DECODE_PID)"
sleep 5

# 4. Start Proxy Server
echo "[4/4] Starting Proxy server on port ${PROXY_PORT}..."
PROXY_SCRIPT="${PROJECT_DIR}/mooncake-wheel/mooncake/vllm_v1_proxy_server.py"
if [ -f "$PROXY_SCRIPT" ]; then
    python3 "${PROXY_SCRIPT}" \
        --prefiller-host 127.0.0.1 --prefiller-port ${PREFILL_PORT} \
        --decoder-host 127.0.0.1 --decoder-port ${DECODE_PORT} \
        --port ${PROXY_PORT} \
        &
    PROXY_PID=$!
    echo "Proxy started (PID: $PROXY_PID)"
else
    echo "Proxy script not found at ${PROXY_SCRIPT}"
    echo "Please run proxy server manually"
    PROXY_PID=""
fi

echo ""
echo "======================================"
echo "All services started!"
echo "  Prefill: http://127.0.0.1:${PREFILL_PORT}"
echo "  Decode:  http://127.0.0.1:${DECODE_PORT}"
echo "  Proxy:   http://127.0.0.1:${PROXY_PORT}"
echo ""
echo "Test with:"
echo "  curl http://127.0.0.1:${PROXY_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d @${CONFIG_DIR}/test_request.json"
echo "======================================"
echo ""
echo "Press Ctrl+C to stop all services."

cleanup() {
    echo "Stopping services..."
    [ -n "$PREFILL_PID" ] && kill $PREFILL_PID 2>/dev/null
    [ -n "$DECODE_PID" ] && kill $DECODE_PID 2>/dev/null
    [ -n "$PROXY_PID" ] && kill $PROXY_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

wait
