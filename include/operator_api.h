/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * nxtLLM Operator API — Adapter Header
 *
 * Provides a unified C API for GPU operators (paged attention,
 * activation functions, quantization kernels) consumed by the
 * nxtLLM inference engine.
 *
 * All functions accept void* tensor pointers so callers are not
 * required to include CUDA headers directly.  The dtype_size
 * parameter distinguishes fp16 (2), fp32 (4), bf16 (2).
 */

#ifndef NXTLLM_OPERATOR_API_H_
#define NXTLLM_OPERATOR_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ── Opaque stream handle ──────────────────────────────────────────── */
typedef void* nxt_stream_t;

/* ═══════════════════════════════════════════════════════════════════════
 * Paged Attention
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Compute multi-head paged attention over KV-cache blocks.
 *
 * Dispatches at runtime by dtype_size (2 → fp16, 4 → fp32) and
 * head_size.  Supported head_size values:
 *   32, 64, 80, 96, 112, 128, 192, 256
 *
 * Kernel: paged_attention_v1_kernel<scalar_t, cache_t, HEAD_SIZE, 16, 128>
 * Uses online-softmax with warp-parallel reduction over paged KV blocks.
 *
 * @param out                [num_seqs, num_heads, head_size] output tensor
 * @param query              [num_seqs, num_heads, head_size] query tensor
 * @param key_cache          [num_blocks, ...] paged key tensor
 * @param value_cache        [num_blocks, ...] paged value tensor
 * @param block_tables       [num_seqs, max_num_blocks_per_seq] int32 mapping
 * @param seq_lens           [num_seqs] int32 sequence lengths
 * @param num_seqs           batch size
 * @param num_heads          number of query heads
 * @param head_size          dimension per head (must be in supported set)
 * @param num_kv_heads       number of KV heads (for GQA/MQA)
 * @param scale              softmax scale (1/sqrt(head_size))
 * @param max_num_blocks_per_seq  max blocks per sequence
 * @param block_size         tokens per KV-cache block (typically 16)
 * @param dtype_size         sizeof(element): 2 (fp16), 4 (fp32)
 * @param kv_block_stride    stride between KV blocks (in elements)
 * @param kv_head_stride     stride between KV heads (in elements)
 * @param stream             CUDA stream
 */
void nxt_paged_attention(
    void* out, const void* query,
    const void* key_cache, const void* value_cache,
    const int* block_tables, const int* seq_lens,
    int num_seqs, int num_heads, int head_size,
    int num_kv_heads, float scale,
    int max_num_blocks_per_seq,
    int block_size, int dtype_size,
    int kv_block_stride, int kv_head_stride,
    nxt_stream_t stream);

#ifdef USE_FLASHINFER
/**
 * FlashInfer-style batch decode paged attention (optional module).
 *
 * Launches a GQA-aware batch-decode kernel with one block per (request, KV-head).
 * Internally dispatches by dtype_size (2 → fp16, 4 → fp32), head_size
 * (64, 80, 96, 112, 128, 192, 256), and GQA group_size (1, 2, 4, 8).
 *
 * Same signature as nxt_paged_attention for drop-in A/B comparison.
 * Requires SM80+, USE_FLASHINFER=ON, USE_CUDA=ON.
 */
void nxt_paged_attention_flash(
    void* out, const void* query,
    const void* key_cache, const void* value_cache,
    const int* block_tables, const int* seq_lens,
    int num_seqs, int num_heads, int head_size,
    int num_kv_heads, float scale,
    int max_num_blocks_per_seq,
    int block_size, int dtype_size,
    int kv_block_stride, int kv_head_stride,
    nxt_stream_t stream);
#endif  /* USE_FLASHINFER */

/* ═══════════════════════════════════════════════════════════════════════
 * Activation + Gating Kernels
 * ═══════════════════════════════════════════════════════════════════════ */

/** out = silu(gate) * up   (gate = first half of input) */
void nxt_silu_and_mul(void* out, const void* input,
                      int num_tokens, int d, int dtype_size,
                      nxt_stream_t stream);

/** out = gate * silu(up)   (up = second half of input) */
void nxt_mul_and_silu(void* out, const void* input,
                      int num_tokens, int d, int dtype_size,
                      nxt_stream_t stream);

/** out = gelu(gate) * up */
void nxt_gelu_and_mul(void* out, const void* input,
                      int num_tokens, int d, int dtype_size,
                      nxt_stream_t stream);

/** out = gelu_tanh(gate) * up */
void nxt_gelu_tanh_and_mul(void* out, const void* input,
                           int num_tokens, int d, int dtype_size,
                           nxt_stream_t stream);

/** out = GELU(input)  (element-wise) */
void nxt_gelu_elementwise(void* out, const void* input,
                          int num_tokens, int d, int dtype_size,
                          nxt_stream_t stream);

/** out = SiLU(input)  (element-wise) */
void nxt_silu_elementwise(void* out, const void* input,
                          int num_tokens, int d, int dtype_size,
                          nxt_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // NXTLLM_OPERATOR_API_H_
