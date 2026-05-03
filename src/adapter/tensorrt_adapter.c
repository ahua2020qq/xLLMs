/*
 * tensorrt_adapter.c — TensorRT-LLM Stub Implementation
 *
 * All functions return safe defaults when compiled without NXTLLM_HAS_TENSORRT.
 * This allows the test suite and any calling code to compile and run without
 * TensorRT installed. When NXTLLM_HAS_TENSORRT is defined, this file should be
 * replaced with the real TRT integration (Phase 2+).
 */

#include <stdlib.h>
#include "tensorrt_adapter.h"

/* ── Runtime (opaque, unused in stub) ────────────────────────────────── */

struct tr_runtime { int dummy; };
struct tr_executor { int dummy; };

/* ── Builder API ────────────────────────────────────────────────────── */

tr_runtime_t* tr_builder_build(const tr_builder_config_t* config)
{
    (void)config;
#ifdef NXTLLM_HAS_TENSORRT
    /* TODO Phase 2+: create TRT builder, set config, build engine */
    return NULL;
#else
    return NULL;
#endif
}

tr_adapter_status_t tr_builder_serialize(const tr_runtime_t* runtime, const char* path)
{
    (void)runtime;
    (void)path;
#ifdef NXTLLM_HAS_TENSORRT
    /* TODO Phase 2+: serialize engine to file */
    return TR_ADAPTER_ERR_NOT_SUPPORTED;
#else
    return TR_ADAPTER_ERR_NOT_SUPPORTED;
#endif
}

tr_runtime_t* tr_runtime_load(const char* path)
{
    (void)path;
#ifdef NXTLLM_HAS_TENSORRT
    /* TODO Phase 2+: deserialize engine from file */
    return NULL;
#else
    return NULL;
#endif
}

void tr_runtime_destroy(tr_runtime_t* runtime)
{
    /* Safe no-op on NULL (common in stub mode) */
    if (runtime == NULL) return;
#ifdef NXTLLM_HAS_TENSORRT
    /* TODO Phase 2+: free TRT engine and context */
#endif
}

/* ── Runtime / CUDA Graph API ───────────────────────────────────────── */

int tr_runtime_supports_cuda_graph(const tr_runtime_t* runtime)
{
    (void)runtime;
    return 0;
}

tr_adapter_status_t tr_runtime_cuda_graph_begin(tr_runtime_t* runtime, int slot)
{
    (void)runtime;
    (void)slot;
    if (runtime == NULL) return TR_ADAPTER_ERR_NOT_INITIALIZED;
    return TR_ADAPTER_ERR_NOT_SUPPORTED;
}

tr_adapter_status_t tr_runtime_cuda_graph_end(tr_runtime_t* runtime)
{
    (void)runtime;
    if (runtime == NULL) return TR_ADAPTER_ERR_NOT_INITIALIZED;
    return TR_ADAPTER_ERR_NOT_SUPPORTED;
}

tr_adapter_status_t tr_runtime_cuda_graph_launch(tr_runtime_t* runtime, int slot)
{
    (void)runtime;
    (void)slot;
    if (runtime == NULL) return TR_ADAPTER_ERR_NOT_INITIALIZED;
    return TR_ADAPTER_ERR_NOT_SUPPORTED;
}

tr_adapter_status_t tr_runtime_run(
    tr_runtime_t*      runtime,
    const int32_t*     input_ids,
    size_t             batch_size,
    size_t             seq_len,
    float*             logits_out,
    void*              kv_cache)
{
    (void)input_ids;
    (void)batch_size;
    (void)seq_len;
    (void)logits_out;
    (void)kv_cache;
    if (runtime == NULL) return TR_ADAPTER_ERR_NOT_INITIALIZED;
    return TR_ADAPTER_ERR_NOT_INITIALIZED;
}

/* ── Executor / In-Flight Batching API ──────────────────────────────── */

tr_executor_t* tr_executor_create(tr_runtime_t* runtime, size_t max_active_requests)
{
    (void)runtime;
    (void)max_active_requests;
#ifdef NXTLLM_HAS_TENSORRT
    /* TODO Phase 5+: create C++ executor wrapper */
    return NULL;
#else
    return NULL;
#endif
}

tr_request_id_t tr_executor_submit(
    tr_executor_t*              executor,
    const int32_t*              prompt_ids,
    size_t                      prompt_len,
    const tr_sampling_params_t* params)
{
    (void)prompt_ids;
    (void)prompt_len;
    (void)params;
    if (executor == NULL) return 0;
    return 0;
}

size_t tr_executor_await_completions(
    tr_executor_t*    executor,
    int32_t*          token_ids_out,
    size_t*           lengths_out,
    tr_request_id_t*  request_ids,
    size_t            max_requests)
{
    (void)executor;
    (void)token_ids_out;
    (void)lengths_out;
    (void)request_ids;
    (void)max_requests;
    return 0;
}

tr_adapter_status_t tr_executor_cancel(tr_executor_t* executor, tr_request_id_t request_id)
{
    (void)request_id;
    if (executor == NULL) return TR_ADAPTER_ERR_NOT_INITIALIZED;
    return TR_ADAPTER_ERR_NOT_INITIALIZED;
}

tr_adapter_status_t tr_executor_get_stats(tr_executor_t* executor, tr_executor_stats_t* stats)
{
    (void)stats;
    if (executor == NULL) return TR_ADAPTER_ERR_NOT_INITIALIZED;
    return TR_ADAPTER_ERR_NOT_INITIALIZED;
}

void tr_executor_destroy(tr_executor_t* executor)
{
    if (executor == NULL) return;
#ifdef NXTLLM_HAS_TENSORRT
    /* TODO Phase 5+: clean up C++ executor */
#endif
}
