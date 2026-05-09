/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA backend — shared library implementing the xLLM backend API.
 * Exports nxt_backend_init / nxt_backend_run / nxt_backend_fini.
 */

#ifndef XLLM_CUDA_BACKEND_H
#define XLLM_CUDA_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

#include "operator_api.h"

/* Forward-declare NxtBackend to avoid pulling in backend.h's conflicting
 * nxt_backend_run declaration (public API vs. backend-plugin symbol). */
typedef struct NxtBackend NxtBackend;

/* ── Paged-attention input descriptor ────────────────────────────────────
 *
 * Callers pack all tensor pointers and shape parameters into this struct
 * and pass it as the `input` argument to nxt_backend_run().  The `output`
 * argument points to the [num_seqs, num_heads, head_size] result tensor.
 */
typedef struct {
    const void   *query;                  /* [num_seqs, num_heads, head_size]   */
    const void   *key_cache;              /* [num_blocks, ...] paged K cache   */
    const void   *value_cache;            /* [num_blocks, ...] paged V cache   */
    const int    *block_tables;           /* [num_seqs, max_num_blocks_per_seq]*/
    const int    *seq_lens;               /* [num_seqs] actual sequence lengths*/
    int           num_seqs;               /* batch size                        */
    int           num_heads;              /* number of query heads             */
    int           head_size;              /* dimension per head                */
    int           num_kv_heads;           /* KV heads (for GQA/MQA)            */
    float         scale;                  /* softmax scale, e.g. 1/sqrt(d)     */
    int           max_num_blocks_per_seq; /* block-table columns               */
    int           block_size;             /* tokens per KV-cache block         */
    int           dtype_size;             /* sizeof(element): 2=fp16, 4=fp32   */
    int           kv_block_stride;        /* stride between KV blocks (elems)  */
    int           kv_head_stride;         /* stride between KV heads  (elems)  */
    nxt_stream_t  stream;                /* CUDA stream handle                */
} NxtPagedAttentionInput;

/* Backend-implemented entry points (exported from .so) */
int nxt_backend_init(NxtBackend *backend);
int nxt_backend_run(NxtBackend *backend, void *input, void *output);
int nxt_backend_fini(NxtBackend *backend);

#ifdef __cplusplus
}
#endif

#endif /* XLLM_CUDA_BACKEND_H */
