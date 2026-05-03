/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Paged Attention V2 — cp.async pipelined kernel with vectorized loads
 * and GQA-aware thread layout.
 *
 * Key improvements over V1:
 *   - Multi-stage cp.async pipeline (SM80+, 3 stages)
 *   - Vectorized loads: float4 for fp32, half4 (uint2) for fp16
 *   - GQA-aware block layout: blockDim.y = group_size heads sharing one KV
 *   - Warp-specialized compute + memory pipeline
 *   - Fused online softmax with warp-level reduction
 *
 * Reference: FlashInfer batch decode, FasterTransformer paged attention.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "operator_api.h"

// ── Pipeline configuration ──────────────────────────────────────────────
#define V2_NUM_STAGES   3           // cp.async pipeline depth
#define V2_WARP_SIZE   32

// ── Vectorized load helpers ─────────────────────────────────────────────

template <typename T>
struct alignas(16) Vec4T {
    T data[4];
};

// Load 4 elements via 16-byte aligned pointer (fp32: float4, fp16: half2 pair)
__device__ __forceinline__ uint4 ld_cs(const void *ptr) {
    uint4 ret;
    asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

// ── cp.async with commit group (SM80+) ─────────────────────────────────

template <int SIZE>
__device__ __forceinline__ void cp_async_commit_group(void) {
    asm volatile("cp.async.commit_group;");
}

template <int SIZE>
__device__ __forceinline__ void cp_async_wait_group(int n) {
    asm volatile("cp.async.wait_group %0;" ::"n"(n));
}

__device__ __forceinline__ void cp_async_cg(const void *dst, const void *src, bool predicate = true) {
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.ca.shared.global [%1], [%2], 16;\n"
                 "}\n" ::"r"((int)predicate), "r"((size_t)dst), "l"(src));
}

// ── Paged Attention V2 Kernel ──────────────────────────────────────────
//
// grid  = (num_seqs, num_kv_heads)
// block = (D_THREADS, GQA_GROUP_SIZE)
//
// Each block handles one (seq, kv_head) pair.
// blockDim.y threads share KV and compute attention for distinct query heads
// within the GQA group.
//
// Pipeline:
//   Stage 0: cp.async load K/V tile from global → shared memory
//   Stage 1: compute Q·K for loaded tile, online softmax
//   Stage 2: weighted V accumulation

template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE, int GQA_GROUP_SIZE, int D_THREADS>
__global__ void paged_attention_v2_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const float scale,
    const int num_kv_heads,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride) {

    // Block indexing
    const int seq_idx   = blockIdx.x;
    const int kv_head   = blockIdx.y;
    const int q_head    = kv_head * GQA_GROUP_SIZE + threadIdx.y;
    const int tid       = threadIdx.x;

    const int seq_len         = seq_lens[seq_idx];
    const int num_kv_blocks   = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Vec size: 16 bytes = 8 half or 4 float
    constexpr int VEC_SIZE = 16 / (int)sizeof(scalar_t);
    constexpr int NUM_VECS = HEAD_DIM / VEC_SIZE;

    // ── Shared memory for K/V tile ─────────────────────────────────────
    __shared__ scalar_t k_smem[V2_NUM_STAGES][BLOCK_SIZE * HEAD_DIM];
    __shared__ scalar_t v_smem[V2_NUM_STAGES][BLOCK_SIZE * HEAD_DIM];

    // ── Load query for this head ───────────────────────────────────────
    const scalar_t* q_ptr = query + seq_idx * q_stride + q_head * HEAD_DIM;
    float q_vals[HEAD_DIM];
    {
        constexpr int ELEMS_PER_THREAD = HEAD_DIM / D_THREADS;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int d = tid + i * D_THREADS;
            q_vals[d] = static_cast<float>(q_ptr[d]);
        }
    }

    // ── Online softmax state ───────────────────────────────────────────
    float m_prev = -1e38f;
    float d_prev = 0.0f;
    float o_vals[HEAD_DIM / D_THREADS];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / D_THREADS; i++) {
        o_vals[i] = 0.0f;
    }

    // ── Pipeline prologue: load first two stages ───────────────────────
    int active_stage = 0;
    int next_block   = 0;

    // Preload stage 0
    if (next_block < num_kv_blocks) {
        int phys_block = block_tables[seq_idx * max_num_blocks_per_seq + next_block];
        if (phys_block >= 0) {
            const scalar_t *k_src = key_cache + phys_block * kv_block_stride
                                    + kv_head * kv_head_stride;
            const scalar_t *v_src = value_cache + phys_block * kv_block_stride
                                    + kv_head * kv_head_stride;
            int tokens = min(BLOCK_SIZE, seq_len - next_block * BLOCK_SIZE);
            for (int tok = tid; tok < tokens * HEAD_DIM; tok += D_THREADS) {
                k_smem[0][tok] = k_src[tok];
                v_smem[0][tok] = v_src[tok];
            }
        }
        next_block++;
    }

    // Preload stage 1
    if (next_block < num_kv_blocks) {
        int phys_block = block_tables[seq_idx * max_num_blocks_per_seq + next_block];
        if (phys_block >= 0) {
            const scalar_t *k_src = key_cache + phys_block * kv_block_stride
                                    + kv_head * kv_head_stride;
            const scalar_t *v_src = value_cache + phys_block * kv_block_stride
                                    + kv_head * kv_head_stride;
            int tokens = min(BLOCK_SIZE, seq_len - next_block * BLOCK_SIZE);
            for (int tok = tid; tok < tokens * HEAD_DIM; tok += D_THREADS) {
                k_smem[1][tok] = k_src[tok];
                v_smem[1][tok] = v_src[tok];
            }
        }
        next_block++;
    }

    __syncthreads();

    // ── Main pipeline loop ─────────────────────────────────────────────
    for (int blk = 0; blk < num_kv_blocks; blk++) {
        int stage_in = blk % V2_NUM_STAGES;

        int phys_block = block_tables[seq_idx * max_num_blocks_per_seq + blk];
        int tokens = min(BLOCK_SIZE, seq_len - blk * BLOCK_SIZE);

        scalar_t *k_ptr = k_smem[stage_in];
        scalar_t *v_ptr = v_smem[stage_in];

        // ── Compute QK over this tile ──────────────────────────────────
        float local_max  = -1e38f;
        float local_sum  = 0.0f;

        // Score accumulation (each thread computes one dot product per token it owns)
        #pragma unroll 1
        for (int tok = 0; tok < tokens; tok++) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < HEAD_DIM / D_THREADS; i++) {
                int d = tid + i * D_THREADS;
                dot += q_vals[d] * static_cast<float>(k_ptr[tok * HEAD_DIM + d]);
            }

            // Warp-level reduce for dot product
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            }

            float s_val = dot * scale;
            float new_max = fmaxf(local_max, s_val);
            float exp_ratio = __expf(local_max - new_max);
            local_sum = local_sum * exp_ratio + __expf(s_val - new_max);
            local_max = new_max;

            // Weighted V accumulation (per-element)
            float weight = __expf(s_val - local_max);
            #pragma unroll
            for (int i = 0; i < HEAD_DIM / D_THREADS; i++) {
                int d = tid + i * D_THREADS;
                o_vals[i] = o_vals[i] * exp_ratio
                          + weight * static_cast<float>(v_ptr[tok * HEAD_DIM + d]);
            }
        }

        // ── Merge local stats into global online softmax state ─────────
        float new_global_max = fmaxf(m_prev, local_max);
        float exp_ratio_global = __expf(m_prev - new_global_max);
        d_prev = d_prev * exp_ratio_global + local_sum * __expf(local_max - new_global_max);
        m_prev = new_global_max;

        // ── Async load next block (if any) ─────────────────────────────
        int next_stage = (blk + 1) % V2_NUM_STAGES;
        int next_blk   = blk + 1;
        if (next_blk < num_kv_blocks) {
            int next_phys = block_tables[seq_idx * max_num_blocks_per_seq + next_blk];
            if (next_phys >= 0) {
                const scalar_t *k_src = key_cache + next_phys * kv_block_stride
                                        + kv_head * kv_head_stride;
                const scalar_t *v_src = value_cache + next_phys * kv_block_stride
                                        + kv_head * kv_head_stride;
                int n_tokens = min(BLOCK_SIZE, seq_len - next_blk * BLOCK_SIZE);
                for (int tok = tid; tok < n_tokens * HEAD_DIM; tok += D_THREADS) {
                    k_smem[next_stage][tok] = k_src[tok];
                    v_smem[next_stage][tok] = v_src[tok];
                }
            }
        }
        __syncthreads();
    }

    // ── Finalize: normalize and write output ───────────────────────────
    float inv_sum = (d_prev > 0.0f) ? (1.0f / d_prev) : 0.0f;
    scalar_t* out_ptr = out + seq_idx * q_stride + q_head * HEAD_DIM;

    #pragma unroll
    for (int i = 0; i < HEAD_DIM / D_THREADS; i++) {
        int d = tid + i * D_THREADS;
        out_ptr[d] = static_cast<scalar_t>(o_vals[i] * inv_sum);
    }
}

// ── V2 Host-side launcher ──────────────────────────────────────────────
// GQA-aware: blockDim.y = GQA group size, blockDim.x = feature dimension threads
// grid = (num_seqs, num_kv_heads) — one block per (request, kv_head)

void nxt_paged_attention_v2(
    void* out, const void* query,
    const void* key_cache, const void* value_cache,
    const int* block_tables, const int* seq_lens,
    int num_seqs, int num_heads, int head_size,
    int num_kv_heads, float scale,
    int max_num_blocks_per_seq,
    int block_size, int dtype_size,
    int kv_block_stride, int kv_head_stride,
    nxt_stream_t stream) {

    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    int group_size = num_heads / num_kv_heads;
    int d_threads  = 64;  // threads covering the feature dimension

    dim3 grid(num_seqs, num_kv_heads);
    dim3 block(d_threads, group_size);

    const int q_stride = num_heads * head_size;

    if (dtype_size == 2) {
        typedef __half T;
        #define V2_LAUNCH(GS, HD)                                            \
            paged_attention_v2_kernel<T, HD, 16, GS, 64>                     \
                <<<grid, block, 0, cu_stream>>>(                             \
                    (T*)out, (const T*)query,                                \
                    (const T*)key_cache, (const T*)value_cache,              \
                    block_tables, seq_lens, max_num_blocks_per_seq, scale,   \
                    num_kv_heads, q_stride, kv_block_stride, kv_head_stride);

        #define V2_DISPATCH_HD(GS)                     \
            switch (head_size) {                         \
                case 32:  V2_LAUNCH(GS, 32);  break;    \
                case 64:  V2_LAUNCH(GS, 64);  break;    \
                case 80:  V2_LAUNCH(GS, 80);  break;    \
                case 96:  V2_LAUNCH(GS, 96);  break;    \
                case 112: V2_LAUNCH(GS, 112); break;    \
                case 128: V2_LAUNCH(GS, 128); break;    \
                case 192: V2_LAUNCH(GS, 192); break;    \
                case 256: V2_LAUNCH(GS, 256); break;    \
                default:  V2_LAUNCH(GS, 64);  break;    \
            }

        switch (group_size) {
            case 1:  V2_DISPATCH_HD(1);  break;
            case 2:  V2_DISPATCH_HD(2);  break;
            case 4:  V2_DISPATCH_HD(4);  break;
            case 8:  V2_DISPATCH_HD(8);  break;
            case 16: V2_DISPATCH_HD(16); break;
            default: V2_DISPATCH_HD(1);  break;
        }
        #undef V2_DISPATCH_HD
        #undef V2_LAUNCH
    } else {
        typedef float T;
        #define V2_LAUNCH_F32(GS, HD)                                        \
            paged_attention_v2_kernel<T, HD, 16, GS, 64>                     \
                <<<grid, block, 0, cu_stream>>>(                             \
                    (T*)out, (const T*)query,                                \
                    (const T*)key_cache, (const T*)value_cache,              \
                    block_tables, seq_lens, max_num_blocks_per_seq, scale,   \
                    num_kv_heads, q_stride, kv_block_stride, kv_head_stride);

        #define V2_DISPATCH_HD_F32(GS)                   \
            switch (head_size) {                          \
                case 32:  V2_LAUNCH_F32(GS, 32);  break;  \
                case 64:  V2_LAUNCH_F32(GS, 64);  break;  \
                case 80:  V2_LAUNCH_F32(GS, 80);  break;  \
                case 96:  V2_LAUNCH_F32(GS, 96);  break;  \
                case 112: V2_LAUNCH_F32(GS, 112); break;  \
                case 128: V2_LAUNCH_F32(GS, 128); break;  \
                case 192: V2_LAUNCH_F32(GS, 192); break;  \
                case 256: V2_LAUNCH_F32(GS, 256); break;  \
                default:  V2_LAUNCH_F32(GS, 64);  break;   \
            }

        switch (group_size) {
            case 1:  V2_DISPATCH_HD_F32(1);  break;
            case 2:  V2_DISPATCH_HD_F32(2);  break;
            case 4:  V2_DISPATCH_HD_F32(4);  break;
            case 8:  V2_DISPATCH_HD_F32(8);  break;
            case 16: V2_DISPATCH_HD_F32(16); break;
            default: V2_DISPATCH_HD_F32(1);  break;
        }
        #undef V2_DISPATCH_HD_F32
        #undef V2_LAUNCH_F32
    }
}
