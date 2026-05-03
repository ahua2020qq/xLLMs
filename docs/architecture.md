<!--
  nxtLLM — Next-Generation LLM Inference Engine
  Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
  SPDX-License-Identifier: Apache-2.0
  This header must not be removed. All derivative works must retain this notice.
-->

# nxtLLM V0.5 架构设计文档

## 1. 系统总览

nxtLLM 是面向大语言模型（LLM）推理场景的高性能内存管理引擎。核心目标是解决 LLM 推理过程中 KV-cache 和模型权重对显存/内存的巨大压力，通过分层存储、智能驱逐和前缀共享策略实现高效的页面级内存管理。

### 1.1 设计目标

| 目标 | 说明 |
|------|------|
| 高效内存利用 | 三层缓冲池（GPU HBM → CPU DRAM → NVMe SSD）统一管理 |
| 智能驱逐 | LRU-K (K=3) + O(1) 哈希索引，避免热点页面被误驱逐 |
| 前缀共享 | 基数树索引 token 序列，跨请求复用 KV-cache |
| 请求感知 | 每请求页面目录 + 准入控制，防止内存超载 |
| 后台碎片整理 | Per-tier 紧凑整理，指针交换，自动重建空闲链表 |
| GPU 流水线 | Paged Attention V2 — cp.async 3-stage pipeline (SM80+) |
| 模块化分层 | 5 层架构，各层独立可替换 |

### 1.2 五层架构

```
┌─────────────────────────────────────────────────────┐
│  Layer 5: 推理引擎          ← V0.3 已实现           │
│  transformer_block.c / tokenizer.c / decode_loop.c  │
│  TransformerBlock 前向传播、BPE 分词、自回归解码      │
├─────────────────────────────────────────────────────┤
│  Layer 4: 请求调度器        ← V0.4 部分实现          │
│  prefix_sharing.c (基数树 KV-cache 索引)             │
│  前缀匹配、节点插入/分裂、LRU 淘汰、引用锁定          │
├─────────────────────────────────────────────────────┤
│  Layer 3: 页面分配器        ← V0.5 已增强            │
│  memory_manager.c                                     │
│  页面分配/释放、哈希索引、碎片整理                       │
├─────────────────────────────────────────────────────┤
│  Layer 2: 驱逐管理器        ← V0.5 已优化            │
│  lru_k.h / lru_k.c                                    │
│  LRU-K (K=3) 驱逐 + O(1) 哈希查找                     │
├─────────────────────────────────────────────────────┤
│  Layer 2: 驱逐管理器        ← V0.1 已实现            │
│  lru_k.h / lru_k.c                                    │
│  LRU-K (K=3) 驱逐、时间戳追踪、victim 选择             │
├─────────────────────────────────────────────────────┤
│  Layer 1: 分层缓冲池        ← V0.1 已实现            │
│  page_pool.h                                          │
│  GPU HBM / CPU DRAM / NVMe SSD 三层存储               │
└─────────────────────────────────────────────────────┘
```

## 2. 模块依赖关系

```
page_pool.h ─────► lru_k.h
     │                  │
     ▼                  ▼
memory_manager.c    lru_k.c
     │
     ▼
prefix_sharing.h ───► prefix_sharing.c
     │
     ▼
operator_api.h ───► page_attention.cu / activation_kernels.cu
     │
     ▼
transformer_block.c / tokenizer.c / decode_loop.c
```

## 3. 版本演进

| 版本 | 核心交付 | 代码量 |
|------|---------|--------|
| V0.1 | 三层缓冲池 + LRU-K 驱逐 + 页面分配器 | ~725 行 |
| V0.2 | GPU 算子库（Paged Attention, SiLU/GELU） | +818 行 |
| V0.3 | GPT-2 推理引擎 + CUDA 可选构建 | +1553 行 |
| V0.4 | 前缀共享基数树（SGLang RadixCache 集成） | +1104 行 |
| V0.5 | 哈希索引 + 碎片整理 + Paged Attention V2 | +929 行 |

## 4. 构建系统

- **语言标准**: C11 / C++17 / CUDA17（可选）
- **构建工具**: CMake >= 3.16
- **编译选项**: `-Wall -Wextra -g`
- **CUDA**: `-DUSE_CUDA=ON` 启用（默认 OFF），支持 SM 75/80/86/89/90
- **产物**:
  - `nxtllm_core` — 静态库（memory_manager + lru_k + prefix_sharing）
  - `nxtllm_operators` — CUDA 静态库（page_attention + page_attention_v2 + activation_kernels）
  - `nxtllm_inference` — 静态库（transformer_block + tokenizer + decode_loop）
  - `test_page_pool` / `test_prefix_sharing` / `test_attention` — 测试
  - `run_gpt2` — 推理示例

## 5. 实现范围

| 模块 | 文件 | 状态 |
|------|------|------|
| 分层缓冲池 | page_pool.h | ✅ |
| LRU-K 驱逐 | lru_k.h / lru_k.c | ✅ |
| 页面分配器 | memory_manager.c | ✅（哈希索引 + 碎片整理已实现） |
| 前缀共享基数树 | prefix_sharing.h / prefix_sharing.c | ✅ |
| GPU 算子 | operator_api.h / *.cu | ✅（V1 + V2 pipelined attention） |
| GPT-2 推理 | transformer_block.c / tokenizer.c / decode_loop.c | ✅ |
| 请求调度器 | scheduler.c / admission.c | ❌ 计划中 |
| KV-cache 管理器 | kv_cache.c | ❌ 计划中 |
