# FlashInfer Attention Architecture — Analysis for nxtLLM Integration

**Date:** 2026-05-04  
**Source:** `/mnt/d/deepseek/flashinfer` (Apache 2.0)  
**Analyst:** nxtLLM team

---

## 1. Overview

FlashInfer is a state-of-the-art GPU kernel library for LLM inference. This
analysis focuses on three components relevant to nxtLLM's paged attention
subsystem:

1. **PagedAttention decode kernel** — `BatchDecodeWithPagedKVCacheKernel`  
2. **Work-estimation scheduler** — `DecodePlan / PartitionPagedKVCacheBinarySearchMinNumPagePerBatch`  
3. **Attention variant dispatch** — `DefaultAttention` and compile-time feature flags

The comparison baseline is nxtLLM's current `paged_attention_v1_kernel` in
`operators/page_attention.cu`, which is a direct port of the vLLM v1 kernel
(one threadblock per head, online softmax, warp-level reduction).

## 2. High-Level Design Comparison

| Dimension               | nxtLLM (current)                          | FlashInfer (target)                           |
|-------------------------|-------------------------------------------|-----------------------------------------------|
| **Kernel launch**       | 1 block per (head, seq), grid=(H, B)     | 1 block per (batch, kv_head), grid=(B, H_kv)  |
| **Batch model**         | Head-parallel, seq-parallel              | Request-parallel with GQA-aware thread layout |
| **Tile size**           | Full block (16 tokens) per iteration      | Template `tile_size_per_bdx` (1, 2, or 4)     |
| **Pipeline**            | No async copy                             | `cp.async` multi-stage (2-4 stages)           |
| **Shared memory**       | logits + output (per-block)              | K/V tiles interleaved with async pipeline     |
| **KV cache**            | Raw pointer arithmetic (block_tables)     | `paged_kv_t` abstraction with protective access|
| **Position encoding**   | Not handled in kernel                    | RoPE applied on-the-fly (k in smem, q at load)|
| **GQA/MQA support**     | Modulo head_idx at load time             | Template `GROUP_SIZE` with warp-level grouping |
| **Partition-KV**        | Not supported                             | Binary-search chaining when `batch_size × H_kv > max_grid_size` |
| **PDL (Programmatic Dependent Launch)** | Not supported               | Supported for CUDA graphs with `cudaLaunchKernelEx` |
| **Logits transforms**   | Simple scale                             | Extensible `LogitsTransform` + `LogitsMask`    |

## 3. Core Kernel Analysis — `BatchDecodeWithPagedKVCacheDevice`

### 3.1 Template Parameters

```cpp
template <
    PosEncodingMode POS_ENCODING_MODE,   // kNone, kRoPELlama, kALiBi
    uint32_t num_stages_smem,            // 2-4 async pipeline stages
    uint32_t tile_size_per_bdx,          // tiles per bdx (1/2/4)
    uint32_t vec_size,                   // 16/sizeof(DTypeKV) or HEAD_DIM/32
    uint32_t bdx, uint32_t bdy, uint32_t bdz, // block dims
    typename AttentionVariant,           // DefaultAttention<...>
    typename Params
>
```

Key insight: The vector size `vec_size = max(16/sizeof(DTypeKV), HEAD_DIM/32)`
ensures 16-byte memory transactions — 8 half-precision elements or 4 float
elements per vector load/store.

### 3.2 Multi-Stage Async Pipeline

FlashInfer uses `cp.async` (CUDA hardware capability SM80+) to overlap global
memory loads with computation:

```
Stage 0: load K[0..tile], V[0..tile]   (prefetch)
Stage 1: load K[tile..2*tile], V[tile..2*tile] (prefetch)
─────────────────────────────────────────────
Iter loop:
  1. wait_group<2*num_stages-1> → K ready
  2. compute QK (fused RoPE + attention scores)
  3. issue K load for future tiles
  4. wait_group<2*num_stages-1> → V ready
  5. update_local_state (W × V outer product)
  6. issue V load for future tiles
  7. advance stage ring buffer
```

Each thread loads `vec_size` elements per memory transaction. The pipeline
achieves near full memory bandwidth utilization.

### 3.3 Paged KV-Cache Indexing

Instead of the vLLM-style `block_tables[seq * max_blocks + block_idx]`,
FlashInfer uses a compact representation:

```cpp
struct paged_kv_t {
    DTypeKV* k_data;               // contiguous allocation
    DTypeKV* v_data;
    IdType*  indices;              // page index mapping  
    IdType*  indptr;               // batch -> start offset in indices
    IdType*  last_page_len;        // variable-length last page
    uint32_t page_size;
    uint32_t num_heads;
    uint32_t batch_size;
    // ...
    
    __device__ size_t protective_get_kv_offset(
        uint32_t page_idx, uint32_t head_idx,
        uint32_t token_in_page, uint32_t feat_idx,
        uint32_t last_indptr);
};
```

The page offset is computed once per tile and cached in shared memory
(`kv_offset_smem`). Periodic re-computation (`% bdx` pattern) amortizes the
integer division cost.

### 3.4 Online Softmax with Warp Merge

Each warp maintains local (m, d, o) state. After the KV traversal:
- If `bdz > 1` (multiple warps per threadblock): warp states are merged via
  shared memory
- The merge uses the standard FlashAttention combine formula: `o_new = (d_old*o_old + d_new*o_new) / d_combined`

### 3.5 RoPE Application Strategy

Rotary embeddings are applied **lazily**:
- **Query**: Apply RoPE once at load time
- **Key**: Apply RoPE in `compute_qk` when key is in shared memory — this
  avoids writing RoPE'd keys back to the KV cache

This is more efficient than nxtLLM's current approach of assuming RoPE is
pre-applied to the cache.

## 4. Scheduler — Dynamic Work Partitioning

### 4.1 Partition-KV Strategy

When `batch_size × num_kv_heads > max_grid_size`, FlashInfer splits long KV
sequences into chunks:

```cpp
auto PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
    max_grid_size, gdy, num_pages, min_num_pages_per_batch)
    → (max_num_pages_per_batch, new_batch_size)
```

The binary search finds the largest page-per-batch size such that the expanded
batch fits within the grid limit. Each chunk produces a partial result, merged
by `VariableLengthMergeStates` (weighted combine with log-sum-exp).

### 4.2 Work Estimation

`DecodePlan` runs a lightweight host-side estimation that:
1. Determines `split_kv` necessity via occupancy calculation
2. Computes offset allocations for temporary buffers
3. Copies schedule metadata (request_indices, kv_tile_indices, o_indptr) to
   page-locked then device memory

## 5. Transferable Techniques to nxtLLM

### 5.1 Immediate Wins (Low Risk)

| Technique                    | Impact                    | Effort |
|------------------------------|---------------------------|--------|
| Vectorized loads (vec_size)  | 2-4× memory throughput    | Low    |
| Template head_size dispatch  | Eliminates switch-case    | Low    |
| GQA warp grouping            | Proper GQA/MQA scaling    | Medium |
| log2-based softmax shortcuts | ~5% arithmetic savings    | Low    |

### 5.2 Medium-Term Enhancements

| Technique                    | Impact                    | Effort |
|------------------------------|---------------------------|--------|
| cp.async multi-stage pipeline| 30-50% decode throughput  | Medium |
| Fused RoPE in compute_qk     | Removes RoPE preprocess   | Medium |
| paged_kv_t abstraction       | Cleaner API, error safety | Low    |
| Partition-KV for long ctx    | Handles arbitrary lengths | Medium |

### 5.3 Advanced (SM80+/SM90 Required)

| Technique                    | Impact                    | Effort |
|------------------------------|---------------------------|--------|
| PDL (cudaLaunchKernelEx)     | CUDA graph efficiency     | High   |
| Tensor Core MMA (Hopper)     | 2× math throughput        | High   |
| FP8 KV-cache quantization    | 2× memory savings         | High   |

## 6. Recommended Integration Path

1. **Phase 1** — Wrap FlashInfer-style decode as `nxt_paged_attention_flash`
   in an optional compilation module (`USE_FLASHINFER` CMake option).
   The adapter uses the same `operator_api.h` signature, allowing A/B
   testing against the existing v1 kernel.

2. **Phase 2** — Port the `paged_kv_t` abstraction and multi-stage async
   pipeline into the main `nxt_paged_attention` path for SM80+ targets.

3. **Phase 3** — Integrate the `DecodePlan` scheduler for automatic
   partition-KV when batch sizes exceed grid capacity.

## 7. Performance Projection

Based on FlashInfer's published benchmarks and the architectural differences
analyzed above:

| Scenario                       | nxtLLM v1 (baseline) | FlashInfer-style (projected) |
|-------------------------------|----------------------|------------------------------|
| Batch=1,  ctx=1024,  fp16     | 1.0×                 | 1.3-1.5×                     |
| Batch=16, ctx=1024,  fp16     | 1.0×                 | 1.5-2.0×                     |
| Batch=64, ctx=4096,  fp16     | 1.0×                 | 2.0-2.5×                     |
| Batch=128, ctx=8192, fp16     | 1.0× (OOM risk)      | 1.5-1.8× (stable)            |

The largest gains come from the multi-stage pipeline (hiding memory latency)
and partition-KV (enabling larger batch sizes).
