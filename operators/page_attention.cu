/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Paged Attention Kernel — Adapted for xLLM
 * Original source: vLLM csrc/attention/paged_attention_v1.cu
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#include "operator_api.h"

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

// ── Utility: float4 add ────────────────────────────────────────────────
__device__ __forceinline__ float4 add(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// ── Utility: float4 fma ────────────────────────────────────────────────
__device__ __forceinline__ float4 fma(float4 a, float4 b, float4 c) {
    return make_float4(a.x * b.x + c.x, a.y * b.y + c.y,
                       a.z * b.z + c.z, a.w * b.w + c.w);
}

// ── Utility: float4 to half4 ───────────────────────────────────────────
__device__ __forceinline__ void float4_to_half4(uint2* out, const float4* in) {
    half2* dst = reinterpret_cast<half2*>(out);
    dst[0] = __halves2half2(__float2half_rn(in->x), __float2half_rn(in->y));
    dst[1] = __halves2half2(__float2half_rn(in->z), __float2half_rn(in->w));
}

// ── Paged Attention V1 Kernel ──────────────────────────────────────────
// Computes attention over paged KV-cache blocks.
// grid:  (num_heads, num_seqs)
// block: (NUM_THREADS)
template <typename scalar_t, typename cache_t, int HEAD_SIZE,
          int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ query,   // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ key_cache,   // [num_blocks, num_kv_heads, ...]
    const cache_t* __restrict__ value_cache, // [num_blocks, num_kv_heads, ...]
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,     // [num_seqs]
    const int max_num_blocks_per_seq,
    const float scale,
    const int num_kv_heads,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride) {

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    const int num_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);

    // ── Shared memory layout ───────────────────────────────────────────
    extern __shared__ char shared_mem[];
    float* logits = reinterpret_cast<float*>(shared_mem);
    float* output = reinterpret_cast<float*>(
        shared_mem + DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE) * BLOCK_SIZE * sizeof(float));

    // ── Initialize output accumulators ────────────────────────────────
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    constexpr int ROWS_PER_WARP = HEAD_SIZE / (NUM_WARPS / 2);
    float qk_max = -1e9f;
    float sum_exp = 0.0f;

    constexpr int SAFE_ROWS = ROWS_PER_WARP > 0 ? ROWS_PER_WARP : 1;
    float out_vals[SAFE_ROWS];
#pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        out_vals[i] = 0.0f;
    }

    // ── Load query for this head ──────────────────────────────────────
    const scalar_t* q_ptr = query + seq_idx * q_stride + head_idx * HEAD_SIZE;
    constexpr int Q_ELEMS = (HEAD_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    float q[Q_ELEMS];
#pragma unroll
    for (int i = 0; i < HEAD_SIZE / NUM_THREADS; i++) {
        int idx = thread_idx + i * NUM_THREADS;
        if (idx < HEAD_SIZE) {
            q[idx] = static_cast<float>(q_ptr[idx]);
        }
    }

    // ── Iterate over KV-cache blocks ──────────────────────────────────
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        if (physical_block < 0) break;  // invalid block

        // ── Q*K ──────────────────────────────────────────────────────
        const int tokens_in_block = MIN(BLOCK_SIZE, seq_len - block_idx * BLOCK_SIZE);
        for (int token = thread_idx; token < tokens_in_block; token += NUM_THREADS) {
            float dot = 0.0f;
            const cache_t* k_ptr = key_cache +
                physical_block * kv_block_stride +
                (head_idx % num_kv_heads) * kv_head_stride +
                token * HEAD_SIZE;

#pragma unroll
            for (int d = 0; d < HEAD_SIZE; d++) {
                dot += q[d] * static_cast<float>(k_ptr[d]);
            }
            logits[token] = dot * scale;
        }
        __syncthreads();

        // ── Softmax (online) ─────────────────────────────────────────
        for (int token = 0; token < tokens_in_block; token++) {
            float val = logits[token];
            float new_max = fmaxf(qk_max, val);
            float exp_ratio = expf(qk_max - new_max);
            sum_exp = sum_exp * exp_ratio + expf(val - new_max);
            qk_max = new_max;

#pragma unroll
            for (int i = 0; i < ROWS_PER_WARP; i++) {
                out_vals[i] *= exp_ratio;
            }

            // ── Weighted value sum ───────────────────────────────────
            const cache_t* v_ptr = value_cache +
                physical_block * kv_block_stride +
                (head_idx % num_kv_heads) * kv_head_stride +
                token * HEAD_SIZE;

            float weight = expf(val - qk_max);
            int row = (thread_idx / WARP_SIZE) * ROWS_PER_WARP;
#pragma unroll
            for (int i = 0; i < ROWS_PER_WARP; i++) {
                int d = row + i;
                if (d < HEAD_SIZE) {
                    out_vals[i] += weight * static_cast<float>(v_ptr[d]);
                }
            }
        }
    }

    // ── Normalize and write output ────────────────────────────────────
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    int row = (thread_idx / WARP_SIZE) * ROWS_PER_WARP;
#pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        int d = row + i;
        if (d < HEAD_SIZE) {
            scalar_t* out_ptr = out + seq_idx * q_stride + head_idx * HEAD_SIZE;
            out_ptr[d] = static_cast<scalar_t>(out_vals[i] * inv_sum);
        }
    }
}

// ── Host-side launcher ─────────────────────────────────────────────────
void nxt_paged_attention(
    void* out, const void* query,
    const void* key_cache, const void* value_cache,
    const int* block_tables, const int* seq_lens,
    int num_seqs, int num_heads, int head_size,
    int num_kv_heads, float scale,
    int max_num_blocks_per_seq,
    int block_size, int dtype_size,
    int kv_block_stride, int kv_head_stride,
    cudaStream_t stream) {

    constexpr int NUM_THREADS = 128;
    constexpr int BLOCK_SIZE = 16;

    dim3 grid(num_heads, num_seqs);
    dim3 block(NUM_THREADS);

    int padded_seq_len = DIVIDE_ROUND_UP(max_num_blocks_per_seq * BLOCK_SIZE, WARP_SIZE) * WARP_SIZE;
    int shared_mem_size = padded_seq_len * sizeof(float) + head_size * sizeof(float);

    const int q_stride = num_seqs * num_heads * head_size;

    // Dispatch by dtype, then by head_size
    if (dtype_size == 2) {
        // ── fp16 ──────────────────────────────────────────────────────
        typedef __half scalar_t;
        typedef __half cache_t;
        switch (head_size) {
            case 32:
                paged_attention_v1_kernel<scalar_t, cache_t, 32, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 64:
                paged_attention_v1_kernel<scalar_t, cache_t, 64, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 80:
                paged_attention_v1_kernel<scalar_t, cache_t, 80, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 96:
                paged_attention_v1_kernel<scalar_t, cache_t, 96, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 112:
                paged_attention_v1_kernel<scalar_t, cache_t, 112, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 128:
                paged_attention_v1_kernel<scalar_t, cache_t, 128, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 192:
                paged_attention_v1_kernel<scalar_t, cache_t, 192, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 256:
                paged_attention_v1_kernel<scalar_t, cache_t, 256, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            default:
                break;
        }
    } else {
        // ── fp32 ──────────────────────────────────────────────────────
        typedef float scalar_t;
        typedef float cache_t;
        switch (head_size) {
            case 32:
                paged_attention_v1_kernel<scalar_t, cache_t, 32, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 64:
                paged_attention_v1_kernel<scalar_t, cache_t, 64, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 80:
                paged_attention_v1_kernel<scalar_t, cache_t, 80, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 96:
                paged_attention_v1_kernel<scalar_t, cache_t, 96, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 112:
                paged_attention_v1_kernel<scalar_t, cache_t, 112, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            case 128:
                paged_attention_v1_kernel<scalar_t, cache_t, 128, 16, 128>
                    <<<grid, block, shared_mem_size, stream>>>(
                        (scalar_t*)out, (const scalar_t*)query,
                        (const cache_t*)key_cache, (const cache_t*)value_cache,
                        block_tables, seq_lens, max_num_blocks_per_seq, scale,
                        num_kv_heads, q_stride, kv_block_stride, kv_head_stride);
                break;
            default:
                break;
        }
    }
}
