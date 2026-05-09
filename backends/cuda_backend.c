/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA backend — calls nxt_paged_attention in its run path.
 * Compiled as a shared library and loaded at runtime via dlopen / LoadLibrary
 * by the backend manager.
 */

/* Suppress name-clash with the public API's nxt_backend_run */
#define nxt_backend_run  nxt_public_backend_run
#include "backend.h"
#undef nxt_backend_run

#include "cuda_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

/* ── Internal per-backend state ─────────────────────────────────────────── */

typedef struct {
    int    device_id;
    int    sm_count;
    char   model_name[256];
} CudaBackendState;

/* ── nxt_backend_init ───────────────────────────────────────────────────── */

int nxt_backend_init(NxtBackend *backend) {
    if (!backend) return -1;

    CudaBackendState *state = calloc(1, sizeof(CudaBackendState));
    if (!state) return -1;

#ifdef USE_CUDA
    cudaError_t err = cudaGetDevice(&state->device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "[cuda_backend] cudaGetDevice failed: %s\n",
                cudaGetErrorString(err));
        free(state);
        return -1;
    }

    struct cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, state->device_id);
    if (err == cudaSuccess) {
        state->sm_count = props.multiProcessorCount;
        snprintf(state->model_name, sizeof(state->model_name), "%s", props.name);
    }
#else
    state->device_id = -1;
    state->sm_count  = 0;
    snprintf(state->model_name, sizeof(state->model_name), "stub (no CUDA)");
#endif

    backend->backend_state = state;
    backend->state = NXT_BACKEND_STATE_READY;
    fprintf(stderr, "[cuda_backend] initialized device=%d sm=%d gpu=%s\n",
            state->device_id, state->sm_count, state->model_name);
    return 0;
}

/* ── nxt_backend_run ────────────────────────────────────────────────────── */

int nxt_backend_run(NxtBackend *backend, void *input, void *output) {
    if (!backend || !backend->backend_state) return -1;
    if (!input || !output) return -1;

    const NxtPagedAttentionInput *pa = (const NxtPagedAttentionInput *)input;

#ifdef USE_CUDA
    nxt_paged_attention(
        /* out          */ output,
        /* query        */ pa->query,
        /* key_cache    */ pa->key_cache,
        /* value_cache  */ pa->value_cache,
        /* block_tables */ pa->block_tables,
        /* seq_lens     */ pa->seq_lens,
        /* num_seqs     */ pa->num_seqs,
        /* num_heads    */ pa->num_heads,
        /* head_size    */ pa->head_size,
        /* num_kv_heads */ pa->num_kv_heads,
        /* scale        */ pa->scale,
        /* max_blocks   */ pa->max_num_blocks_per_seq,
        /* block_size   */ pa->block_size,
        /* dtype_size   */ pa->dtype_size,
        /* kv_blk_stride*/ pa->kv_block_stride,
        /* kv_hd_stride */ pa->kv_head_stride,
        /* stream       */ pa->stream);
#else
    (void)pa;
    (void)output;
    fprintf(stderr, "[cuda_backend] nxt_backend_run: CUDA not available (stub)\n");
    return -1;
#endif

    return 0;
}

/* ── nxt_backend_fini ───────────────────────────────────────────────────── */

int nxt_backend_fini(NxtBackend *backend) {
    if (!backend) return -1;

    CudaBackendState *state = (CudaBackendState *)backend->backend_state;
    if (state) {
        fprintf(stderr, "[cuda_backend] unloading device=%d\n", state->device_id);
        free(state);
        backend->backend_state = NULL;
    }

    backend->state = NXT_BACKEND_STATE_UNINITIALIZED;
    return 0;
}
