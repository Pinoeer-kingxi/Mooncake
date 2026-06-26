#!/bin/bash
# 启动 Mooncake 基础服务
# 包括 mooncake_master 和 store client

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting Mooncake environment..."
echo "Protocol: tcp"
echo "Master port: 50051"
echo "Metadata port: 8080"

# Activate venv
source "${PROJECT_DIR}/../venv_mooncake/bin/activate"

# Start mooncake_master
echo "[1/2] Starting mooncake_master..."
mooncake_master \
    --rpc_port=50051 \
    --enable_http_metadata_server=true \
    --http_metadata_server_port=8080 \
    --http_metadata_server_host=0.0.0.0 \
    &
MASTER_PID=$!
echo "mooncake_master started (PID: $MASTER_PID)"
sleep 2

# Start store client
echo "[2/2] Starting store client..."
MOONCAKE_MASTER=127.0.0.1:50051 \
MOONCAKE_TE_META_DATA_SERVER=http://127.0.0.1:8080/metadata \
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_GLOBAL_SEGMENT_SIZE=16gb \
MOONCAKE_LOCAL_BUFFER_SIZE=4gb \
MOONCAKE_LOCAL_HOSTNAME=localhost \
python3 -m mooncake.mooncake_store_service \
    &
CLIENT_PID=$!
echo "Store client started (PID: $CLIENT_PID)"

echo ""
echo "Mooncake environment is running."
echo "  Master: 127.0.0.1:50051"
echo "  Metadata: http://127.0.0.1:8080/metadata"
echo ""
echo "Press Ctrl+C to stop."

# Wait for processes
wait $MASTER_PID $CLIENT_PID
