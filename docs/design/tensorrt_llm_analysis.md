# TensorRT-LLM 核心技术分析报告

> 分析日期: 2026-05-04
> 源码路径: `/mnt/d/deepseek/TensorRT-LLM/tensorrt_llm/`
> 目标: 提取层融合、量化、CUDA Graph、In-Flight Batching 核心设计，指导 nxtLLM 适配

---

## 1. 层融合 (Layer Fusion)

TensorRT-LLM 在 Python 模型定义层实现了多种算子融合，消除冗余的内存读写和 kernel launch 开销。

### 1.1 融合 QKV 投影

**文件**: `layers/attention.py` (L514-525)

将 Query、Key、Value 三个线性投影合并为单个 `ColumnLinear`：

```
传统: W_q @ x, W_k @ x, W_v @ x  → 3 次 GEMM
融合: W_qkv @ x → split → Q, K, V  → 1 次 GEMM
```

- `self.qkv = ColumnLinear(hidden_size, hidden_size * 3)` 一次计算全部投影
- `enable_qkv=False` 时回退到独立投影（`unfuse_qkv_gemm`, L841）

### 1.2 融合门控 MLP (SwiGLU)

**文件**: `layers/mlp.py` (L267-442)

`FusedGatedMLP` 将 FC 和 Gate 投影合并：

```
传统: FC(x) * sigmoid(Gate(x))  → 2 次 GEMM + 1 次 element-wise
融合: fused_gemm(x) → split → swiglu  → 1 次 GEMM + 1 次激活
```

- 输出维度 `ffn_hidden_size * 2`，激活函数内部 split
- 可选 FP8 `gemm_swiglu_plugin` (L318) 进一步融合 GEMM+激活到单个 TRT plugin

### 1.3 融合 MoE 专家权重

**文件**: `layers/moe.py` (L867-891)

`MixtureOfExperts` 中，gate 和 fc 权重水平拼接为 `MOEWeightWrapper`：

- `expert_inter_size * 2` 输出维度
- `_moe_plugin` TRT plugin 在单个 kernel 中完成路由、计算、缩放

### 1.4 融合 AllReduce + Bias + Residual

**文件**: `layers/linear.py` (RowLinear, L523-551)

`RowLinear.collect_and_bias()` 将 all-reduce 通信与 bias 加法、残差 RMS norm 合并：

- `AllReduceFusionOp.RESIDUAL_RMS_NORM` 模式在 all-reduce kernel 中同时完成 bias 和 norm
- 消除额外的 element-wise kernel launch

### 1.5 DeepSeekV2 MLA 融合投影

**文件**: `layers/attention.py` (L2107-2120)

`DeepseekV2Attention` 的 `self.fused_a` 将 Q 压缩、KV 压缩、QK RoPE 三个投影合并为单次 GEMM：

```
输出维度 = q_lora_rank + kv_lora_rank + qk_rope_head_dim
```

---

## 2. 量化 (Quantization)

### 2.1 量化模式体系

**文件**: `quantization/mode.py`

`QuantAlgo` 枚举定义了 26 种量化算法：

| 家族 | 算法 | 说明 |
|------|------|------|
| W8A16 | `W8A16`, `W8A16_GPTQ` | INT8 权重量化 |
| W4A16 | `W4A16`, `W4A16_AWQ`, `W4A16_GPTQ` | INT4 权重量化 |
| SmoothQuant | `W8A8_SQ_PER_CHANNEL` + 4 插件变体 | INT8 权重+激活量化 |
| FP8 | `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_BLOCK_SCALES` | FP8 QDQ / Rowwise |
| NVFP4 | `NVFP4`, `NVFP4_AWQ`, `NVFP4_ARC` | NVIDIA FP4 精度 |
| QServe | `W4A8_QSERVE_PER_GROUP/PER_CHANNEL` | W4A8 QServe kernel |
| MXFP4 | `W4A8_NVFP4_FP8`, `W4A8_MXFP4_FP8` 等 | 混合 FP4/FP8 |
| KV Cache | `INT8`, `FP8`, `NVFP4` | KV-cache 专用量化 |

`QuantMode` (IntFlag) 将算法分解为可组合的位掩码：`INT4_WEIGHTS`, `INT8_WEIGHTS`, `ACTIVATIONS`, `PER_CHANNEL`, `PER_TOKEN`, `PER_GROUP`, `FP8_QDQ`, `FP8_ROWWISE`, `NVFP4` 等。

### 2.2 量化管线

**Phase A — ModelOpt 离线量化** (`quantize_by_modelopt.py`):
- 加载 HuggingFace 模型 → NVIDIA ModelOpt PTQ/AWQ 校准 → 导出 TRT-LLM checkpoint

**Phase B — 构建时层替换** (`quantize.py`, L569):
- 读取 `QuantConfig` → 遍历模型树 → 用量化版本替换标准层

```
QuantMode flag              → 替换函数             → 量化层类型
has_fp8_qdq()              → fp8_quantize()       → FP8Linear / FP8RowLinear
has_fp8_rowwise()          → fp8_rowwise_quantize() → Fp8RowwiseAttention/MLP
has_nvfp4()                → fp4_quantize()       → FP4Linear / FP4RowLinear
has_act_and_weight_quant() → smooth_quantize()    → Int8SmoothQuantLinear
is_weight_only() + group   → weight_only_groupwise_quantize()
is_weight_only()           → weight_only_quantize()
```

### 2.3 量化算子

**文件**: `quantization/functional.py`

核心量化 GEMM:
- `smooth_quant_gemm()` — INT8 权重 × INT8 激活
- `weight_only_quant_matmul()` — INT4/INT8 权重 × FP16 激活
- `qserve_gemm_per_group()` — W4A8 分组量化 GEMM
- `fp8_rowwise_gemm()` — FP8 逐行量化 GEMM
- `fp4_gemm()` — NVFP4 GEMM

融合量化+Norm:
- `smooth_quant_rms_norm()` — RMSNorm + INT8 量化融合
- `fp8_rowwise_rms_norm()` — RMSNorm + FP8 量化融合

---

## 3. CUDA Graph

### 3.1 实现位置

**文件**: `runtime/generation.py` (class `GenerationSession`, L860+)

### 3.2 双缓冲 Ping-Pong 设计

```
cuda_graph_instances = [None, None]   # 两个 graph 槽位
instance_idx = step % 2               # 当前步选择槽位
```

- **槽位 0**: 捕获奇数步的 graph
- **槽位 1**: 捕获偶数步的 graph
- 允许一边 replay 一边 capture 下一轮

### 3.3 捕获与更新流程

1. **Capture**: `_capture_cuda_graph_and_instantiate()` (L1360)
   - `cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)`
   - 执行 TensorRT engine context (`execute_async_v3`)
   - `cudaStreamEndCapture(stream, &graph)`
   - `cudaGraphInstantiate(&instance, graph)`

2. **Update**: `_update_cuda_graph_instance()` (L145)
   - `cudaGraphExecUpdate(instance, graph)` — 增量更新
   - 失败时回退到 destroy + re-instantiate

3. **Launch**: `cudaGraphLaunch(instance, stream)` (L3622-3627)
   - 替代 `self.runtime._run()` 的正常执行路径

### 3.4 上下文模式处理

Context phase 的 tensor shapes 与 generation 不同 → 清除所有 graph instances (L3613-3615)，用正常执行路径。

### 3.5 配置传递链

```
ModelRunnerCpp.cuda_graph_mode (L122)
  → ExtendedRuntimePerfKnobConfig
  → GenerationSession.__init__(cuda_graph_mode=...) (L906)
```

---

## 4. In-Flight Batching

### 4.1 架构层次

In-flight batching 的核心在 **C++ 层**，Python 层提供封装：

```
GenerationRequest (Python)
  → GenerationExecutor.submit() / generate_async()
    → GenerationExecutorProxy (MPI, L39 executor/proxy.py)
      → GenerationExecutorWorker (L36 executor/worker.py)
        → tllm.Executor (C++ bindings)
          → [真正的 in-flight batching 调度]
```

### 4.2 Python 层接口

**文件**: `executor/executor.py`

- `GenerationExecutor` (ABC, L80) — 抽象基类
- `submit()` — 提交单个请求
- `generate_async()` (L123) — 异步生成
- `generate()` — 同步生成

**文件**: `executor/request.py`

`GenerationRequest` 携带：
- `sampling_params` — 温度、top-k、top-p 等
- `scheduling_params` — 调度优先级
- `kv_cache_retention_config` — KV-cache 保留策略
- `disaggregated_params` — 分离式 prefill/decode 参数
- `priority` — 默认 0.5，优先级调度

### 4.3 调度策略

C++ Executor 负责：
- **Continuous batching**: 请求完成即退出，新请求即时加入
- **Chunked prefill**: 长 context 分块 prefill，与 decode 交错
- **Priority scheduling**: 基于 priority 值的调度排序
- **Iteration-level scheduling**: 每步决定哪些请求参与计算

### 4.4 KV-Cache 管理 (V2)

**目录**: `runtime/kv_cache_manager_v2/`

多层级缓存：
- **GPU tier** (`GpuCacheTierConfig`) — 热数据，HBM
- **Host tier** (`HostCacheTierConfig`) — 温数据，CPU RAM
- **Disk tier** (`DiskCacheTierConfig`) — 冷数据，SSD

BlockRadixTree (`_block_radix_tree.py`):
- 基数树索引 KV-cache blocks
- 跨请求 block 复用
- 写时复制 (Copy-on-Write) 支持 beam search

---

## 5. nxtLLM 适配方案

### 5.1 适配优先级矩阵

| 优先级 | 技术 | 收益 | 难度 | nxtLLM 结合点 |
|--------|------|------|------|---------------|
| P0 | 融合 QKV 投影 | 减少 66% GEMM launch | 低 | `transformer_block.c` attention 路径 |
| P0 | 融合门控 MLP | 减少 50% FC kernel | 低 | `transformer_block.c` FFN 路径 |
| P1 | INT8/FP8 权重量化 | 50% 显存节省 | 中 | `weight_loader.c` + 量化 GEMM |
| P1 | CUDA Graph 双缓冲 | 消除 kernel launch 开销 | 中 | `decode_loop.c` 添加 graph 槽位 |
| P2 | In-Flight Batching | 吞吐提升 2-5x | 高 | 新增 `scheduler.c` 调度模块 |
| P2 | 融合 AllReduce+Bias+Norm | 减少通信轮次 | 中 | 多 GPU TP 场景 |
| P3 | BlockRadixTree KV Cache | 跨请求前缀复用 | 高 | 增强现有 `prefix_sharing.c` |
| P3 | FP4/NVFP4 量化 | 更极致的显存压缩 | 高 | Blackwell GPU 优化 |

### 5.2 存根接口设计

适配头文件 `tensorrt_adapter.h` 定义了最小接口集合：
- Builder 配置接口（量化、融合开关）
- Runtime 接口（CUDA Graph、执行）
- Executive 接口（请求提交、结果获取）
- 环境检测宏 `NXTLLM_HAS_TENSORRT`

### 5.3 渐进式集成路线

1. **Phase 1 (存根)**: 编译通过的空实现，不影响现有功能
2. **Phase 2 (量化)**: INT8/FP8 权重量化 + 量化 GEMM
3. **Phase 3 (融合)**: QKV 融合 + 门控 MLP 融合
4. **Phase 4 (图优化)**: CUDA Graph 双缓冲 decode loop
5. **Phase 5 (调度)**: In-flight batching scheduler

---

## 6. 与现有组件的协同

| nxtLLM 组件 | TensorRT 增强点 |
|-------------|----------------|
| `page_attention.cu` | TRT 融合 QKV attention plugin |
| `transformer_block.c` | 融合 GatedMLP → 减少 GEMM 调用 |
| `decode_loop.c` | CUDA Graph 双缓冲 replay |
| `memory_manager.c` | BlockRadixTree 增强前缀复用 |
| `prefix_sharing.c` | 基数树升级为多层级 KV-cache 索引 |
| `weight_loader.c` | 量化权重加载 + 解量化 |
