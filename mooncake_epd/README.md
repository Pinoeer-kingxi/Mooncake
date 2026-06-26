# Mooncake EPD Disaggregation Framework

## 赛题四：Agent 与多模态推理场景下的 KVCache 分离与协同调度

基于 [Mooncake](https://github.com/kvcache-ai/Mooncake) 的 EPD 三阶段分离与 Agent 状态协同调度框架，面向下一代 AI 多模态推理工作负载。

## 项目概述

本框架将多模态大模型推理拆分为三个独立阶段（EPD），通过 Mooncake Transfer Engine 实现跨节点高性能数据传输：

```
Image → [Encoder GPU] → Hidden States → [Transfer E→P] →
        [Prefill GPU] → KV Cache → [Transfer P→D] →
        [Decode GPU] → Text
```

同时实现了 Agent 状态克隆、PD 调度策略和 Hidden State 前缀缓存等进阶功能。

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | RTX A6000 48GB x 8 (或同等) |
| CPU | 64 cores |
| RAM | 503 GB |
| CUDA | 12.9 |
| Python | 3.10+ |
| Mooncake | 0.3.11+ |
| vLLM | latest (V1 backend) |

### 硬件说明

当前环境为 RTX A6000 工作站，**无 RDMA/InfiniBand 硬件**。框架使用 TCP 协议作为传输层，功能完整但带宽受限。在 RDMA 环境下可获得最高 142 GB/s 的传输带宽。

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
python3.10 -m venv venv_mooncake
source venv_mooncake/bin/activate

# 安装 Mooncake
pip install mooncake-transfer-engine

# 安装 vLLM
pip install vllm

# 安装其他依赖
pip install torch numpy Pillow pyyaml requests aiohttp
```

### 2. 运行 EPD Demo

```bash
cd mooncake_epd
python demo/run_qwenvl_epd.py
```

### 3. 运行性能基准测试

```bash
python benchmarks/benchmark.py
```

### 4. 启动 Mooncake 基础服务

```bash
bash scripts/start_mooncake.sh
```

### 5. 启动 vLLM EPD 分离推理

```bash
bash scripts/start_vllm_disagg.sh
```

## 项目结构

```
mooncake_epd/
├── __init__.py
├── config/
│   ├── config.yaml              # 总配置文件
│   └── mooncake.json            # Mooncake Transfer Engine 配置
├── core/
│   ├── __init__.py
│   ├── transfer_engine.py       # Mooncake Transfer Engine 封装
│   ├── encoder_worker.py        # Vision Encoder Worker
│   ├── prefill_worker.py        # Prefill Worker
│   ├── decode_worker.py         # Decode Worker
│   └── epd_pipeline.py          # EPD 流水线编排
├── agent/
│   ├── __init__.py
│   ├── state_clone.py           # Agent KVCache 零拷贝克隆
│   ├── scheduler.py             # Agent PD 调度策略
│   └── prefix_cache.py          # Hidden State 前缀缓存
├── demo/
│   ├── run_qwenvl_epd.py        # Qwen3-VL EPD 端到端 Demo
│   └── vllm_integration.py      # vLLM 集成配置生成
├── benchmarks/
│   └── benchmark.py             # 性能基准测试
├── scripts/
│   ├── start_mooncake.sh        # 启动 Mooncake 服务
│   ├── start_vllm_disagg.sh     # 启动 vLLM EPD 分离
│   └── setup_mooncake.py        # Mooncake 环境管理
└── requirements.txt
```

## 基础任务实现

### 1. EPD 三阶段分离原型

**文件**: `core/encoder_worker.py`, `core/prefill_worker.py`, `core/decode_worker.py`, `core/epd_pipeline.py`

**实现要点**:
- Vision Encoder (ViT) 在独立 GPU 上运行，输出 Hidden States
- Prefill Worker 接收视觉特征 + 文本 token，生成 KV Cache
- Decode Worker 接收 KV Cache，执行自回归解码
- 通过 Mooncake Transfer Engine 实现 E→P (Hidden States) 和 P→D (KV Cache) 传输
- 支持 TCP 和 RDMA 协议

### 2. Agent State Cloning

**文件**: `agent/state_clone.py`

**实现要点**:
- 零拷贝克隆：通过引用计数共享 KV Cache 物理内存
- 写时复制 (CoW)：仅在修改时才分配新内存
- 生命周期管理：引用计数为 0 时自动回收
- 支持 Tree-of-Thought 剪枝（保留 top-k 分支）

**性能数据**:
- 2 分支克隆: 0.098 ms (0.049 ms/branch)
- 4 分支克隆: 0.142 ms (0.036 ms/branch)
- 8 分支克隆: 0.248 ms (0.031 ms/branch)
- 16 分支克隆: 0.471 ms (0.029 ms/branch)

### 3. Qwen-VL 端到端 Demo

**文件**: `demo/run_qwenvl_epd.py`

四个 Demo 模块：
1. **Basic EPD**: 多模态输入经过 E→P→D 三阶段处理
2. **Agent Cloning**: Tree-of-Thought 思考分支 fork 与剪枝
3. **Prefix Caching**: 图像编码结果缓存，避免重复计算
4. **PD Scheduling**: 思考型/交互型 Agent 动态路由

## 进阶任务实现

### 1. Agent PD Disaggregation 调度策略

**文件**: `agent/scheduler.py`

- 思考型 Agent → 高算力 Prefill Worker（选择 GPU utilization 最低的）
- 交互型 Agent → 低延迟 Decode Worker（选择 avg_latency 最低的）
- 支持优先级调度和动态负载均衡

### 2. Hidden State Prefix Caching

**文件**: `agent/prefix_cache.py`

- 基于 SHA-256 图像 hash 的精确匹配
- LRU 淘汰策略
- 可配置缓存大小 (默认 4GB) 和 TTL (默认 1 小时)
- 相同图像命中率 100%

### 3. vLLM MooncakeConnector 集成

**文件**: `demo/vllm_integration.py`, `scripts/start_vllm_disagg.sh`

- 使用 vLLM V1 后端的 `MooncakeConnector`
- Prefill (kv_producer) 和 Decode (kv_consumer) 分离部署
- Proxy Server 路由请求

## 性能数据

### Benchmark 结果 (Mock 模型, A6000, TCP)

| 指标 | 数值 |
|------|------|
| **EPD Pipeline** | |
| Avg Latency | 167.31 ms |
| P50 Latency | 160.26 ms |
| P99 Latency | 207.98 ms |
| Avg TTFT | 2.54 ms |
| Throughput | 204.59 tokens/s |
| **Transfer Bandwidth (Local CUDA)** | |
| 4KB Tensor | 1.284 Gbps |
| 40KB Tensor | 13.417 Gbps |
| 400KB Tensor | 120.452 Gbps |
| 4MB Tensor | 1240.088 Gbps |
| **Agent Cloning** | |
| 2 branches | 0.049 ms/branch |
| 16 branches | 0.029 ms/branch |
| **Prefix Caching** | |
| Cache Hit Rate | 100% |

> 注：以上数据基于 Mock 模型的演示性测试。实际模型（如 Qwen3-VL-8B）的数据会有所不同。

## vLLM 集成指南

### 使用 MooncakeConnector 实现 PD 分离

1. **配置 mooncake.json**:
```json
{
  "prefill_url": "127.0.0.1:9100",
  "decode_url": "127.0.0.1:9200",
  "metadata_server": "http://127.0.0.1:8080/metadata",
  "protocol": "tcp",
  "device_name": ""
}
```

2. **启动 Prefill**:
```bash
MOONCAKE_CONFIG_PATH=mooncake.json \
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8100 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

3. **启动 Decode**:
```bash
MOONCAKE_CONFIG_PATH=mooncake.json \
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8200 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

4. **启动 Proxy**:
```bash
python mooncake/vllm_v1_proxy_server.py \
  --prefiller-host 127.0.0.1 --prefiller-port 8100 \
  --decoder-host 127.0.0.1 --decoder-port 8200 \
  --port 8000
```

## 已知限制

1. **无 RDMA**: A6000 工作站无 InfiniBand/RoCE 硬件，TCP 模式带宽受限
2. **Mock 模型**: Demo 使用模拟模型验证架构，实际 Qwen3-VL 需要 vLLM 集成
3. **跨节点**: 当前仅验证单节点多 GPU 场景，跨节点需要网络配置
4. **MooncakeConnector Proxy**: 当前使用 vLLM 自带的 toy_proxy_server，生产环境需要更健壮的方案

## 模型依赖

| 模型 | 用途 | VRAM 需求 |
|------|------|-----------|
| Qwen2.5-VL-7B-Instruct | 多模态推理 | ~16 GB (FP16) |
| Qwen2.5-VL-32B-Instruct | 高性能推理 | ~64 GB (FP16) |
| Qwen3-VL-8B (预期) | 最新模型 | ~18 GB (FP16) |

## 框架版本

| 组件 | 版本 |
|------|------|
| Mooncake | 0.3.11.post1 |
| vLLM | latest (V1) |
| PyTorch | 2.x |
| CUDA | 12.9 |
| Python | 3.10 |

## 部署拓扑

### 单机三 GPU EPD 分离
```
GPU 0: Vision Encoder (E)
GPU 1: LLM Prefill (P)
GPU 2: LLM Decode (D)
```

### 多机 EPD 分离
```
Node 1 (GPU 0,1): Encoder + Prefill
Node 2 (GPU 2,3): Decode
Transfer: Mooncake (TCP/RDMA)
```

### 生产环境建议
```
Node 1-2: Vision Encoder Pool
Node 3-6: Prefill Pool (high compute)
Node 7-8: Decode Pool (low latency)
Load Balancer: Agent PD Scheduler
```
