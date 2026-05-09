#ifndef XLLM_TENSORRT_ADAPTER_H
#define XLLM_TENSORRT_ADAPTER_H

/*
 * tensorrt_adapter.h — TensorRT-LLM Stub Adapter Interface
 *
 * This header defines the minimum interface for optionally integrating
 * TensorRT-LLM optimizations (layer fusion, quantization, CUDA Graph,
 * in-flight batching) into xLLM. When XLLM_HAS_TENSORRT is not defined,
 * all functions are no-op stubs that allow the codebase to compile
 * without TensorRT dependencies.
 *
 * Progressive integration phases:
 *   Phase 1 (stub):       Compile-time stubs — no TensorRT required
 *   Phase 2 (quant):      INT8/FP8 weight quantization
 *   Phase 3 (fusion):     Fused QKV + GatedMLP kernels
 *   Phase 4 (graph):      CUDA Graph dual-buffer decode loop
 *   Phase 5 (batching):   In-flight batching scheduler
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Compile-time feature detection ─────────────────────────────────── */

#ifdef XLLM_HAS_TENSORRT
#  define XLLM_TR_VERSION_MAJOR 0
#  define XLLM_TR_VERSION_MINOR 1
#  define XLLM_TR_VERSION_PATCH 0
#  define XLLM_TR_HAS_QUANT     1
#  define XLLM_TR_HAS_CUDA_GRAPH 0
#  define XLLM_TR_HAS_INFLIGHT_BATCHING 0
#endif

/* ── Error codes ────────────────────────────────────────────────────── */

typedef enum {
    TR_ADAPTER_OK = 0,
    TR_ADAPTER_ERR_NOT_INITIALIZED = -1,
    TR_ADAPTER_ERR_INVALID_CONFIG  = -2,
    TR_ADAPTER_ERR_BUILD_FAILED    = -3,
    TR_ADAPTER_ERR_RUNTIME_ERROR   = -4,
    TR_ADAPTER_ERR_OOM             = -5,
    TR_ADAPTER_ERR_NOT_SUPPORTED   = -6,
} tr_adapter_status_t;

/* ── Quantization configuration ─────────────────────────────────────── */

typedef enum {
    TR_QUANT_NONE    = 0,  /* No quantization (FP16/BF16) */
    TR_QUANT_INT8_W8A16 = 1,  /* INT8 weight-only */
    TR_QUANT_INT4_W4A16 = 2,  /* INT4 weight-only (AWQ/GPTQ) */
    TR_QUANT_FP8      = 3,  /* FP8 QDQ (weights + activations) */
    TR_QUANT_INT8_SQ  = 4,  /* INT8 SmoothQuant (W8A8) */
    TR_QUANT_NVFP4    = 5,  /* NVFP4 weight-only */
    TR_QUANT_MIXED    = 6,  /* Per-layer mixed precision */
} tr_quant_mode_t;

/* ── Layer fusion flags (bitmask) ───────────────────────────────────── */

typedef enum {
    TR_FUSE_NONE        = 0,
    TR_FUSE_QKV         = 1 << 0,  /* Fuse Q/K/V projections */
    TR_FUSE_GATED_MLP   = 1 << 1,  /* Fuse FC + Gate in SwiGLU */
    TR_FUSE_BIAS_RESID  = 1 << 2,  /* Fuse AllReduce + Bias + Residual */
    TR_FUSE_GEMM_ACT    = 1 << 3,  /* Fuse GEMM + Activation */
    TR_FUSE_RMSNORM_Q   = 1 << 4,  /* Fuse RMSNorm + Quantization */
    TR_FUSE_ROPE_ATTN   = 1 << 5,  /* Fuse RoPE into attention */
    TR_FUSE_ALL         = 0x3F,    /* All fusion flags */
} tr_fusion_flags_t;

/* ── Builder configuration ──────────────────────────────────────────── */

typedef struct {
    /* Model dimensions */
    size_t hidden_size;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    size_t intermediate_size;
    size_t num_layers;
    size_t vocab_size;
    size_t max_seq_len;

    /* Optimization knobs */
    tr_quant_mode_t   quant_mode;
    uint32_t          fusion_flags;  /* tr_fusion_flags_t bitmask */
    int               use_cuda_graph;
    int               use_inflight_batching;

    /* Precision */
    int               use_fp16;   /* 1 = FP16, 0 = FP32 */
    int               use_bf16;   /* 1 = BF16 */

    /* Memory */
    size_t            max_batch_size;
    size_t            kv_cache_block_size;
    size_t            num_kv_cache_blocks;
} tr_builder_config_t;

/* ── Runtime handle (opaque) ────────────────────────────────────────── */

typedef struct tr_runtime tr_runtime_t;
typedef struct tr_executor tr_executor_t;

/* ── Builder API ────────────────────────────────────────────────────── */

/** Create a TRT engine builder from config.
 *  Returns NULL (stub) when XLLM_HAS_TENSORRT is not defined. */
tr_runtime_t* tr_builder_build(const tr_builder_config_t* config);

/** Serialize built engine to file. Returns TR_ADAPTER_ERR_NOT_SUPPORTED in stub mode. */
tr_adapter_status_t tr_builder_serialize(const tr_runtime_t* runtime, const char* path);

/** Load engine from serialized file. Returns NULL (stub) when XLLM_HAS_TENSORRT is not defined. */
tr_runtime_t* tr_runtime_load(const char* path);

/** Destroy runtime and free resources. */
void tr_runtime_destroy(tr_runtime_t* runtime);

/* ── Runtime / CUDA Graph API ───────────────────────────────────────── */

/** Check if CUDA Graph capture is supported. Returns 0 in stub mode. */
int tr_runtime_supports_cuda_graph(const tr_runtime_t* runtime);

/** Begin CUDA Graph capture for the current step. Returns TR_ADAPTER_ERR_NOT_SUPPORTED in stub mode. */
tr_adapter_status_t tr_runtime_cuda_graph_begin(tr_runtime_t* runtime, int slot);

/** End CUDA Graph capture and instantiate. */
tr_adapter_status_t tr_runtime_cuda_graph_end(tr_runtime_t* runtime);

/** Launch a previously captured CUDA Graph instance. */
tr_adapter_status_t tr_runtime_cuda_graph_launch(tr_runtime_t* runtime, int slot);

/** Run a single forward pass (generation step).
 *  input_ids:   [batch_size] token IDs
 *  logits_out:  [batch_size, vocab_size] output logits (may be NULL for decode-only)
 *  kv_cache:    opaque pointer to KV-cache state
 *  Returns TR_ADAPTER_ERR_NOT_INITIALIZED in stub mode. */
tr_adapter_status_t tr_runtime_run(
    tr_runtime_t*      runtime,
    const int32_t*     input_ids,
    size_t             batch_size,
    size_t             seq_len,
    float*             logits_out,
    void*              kv_cache
);

/* ── Executor / In-Flight Batching API ──────────────────────────────── */

/** Request handle for in-flight batching. */
typedef uint64_t tr_request_id_t;

/** Sampling parameters for a generation request. */
typedef struct {
    float   temperature;
    int     top_k;
    float   top_p;
    int     max_new_tokens;
    int     n_beams;           /* 1 = greedy/sampling */
    int     repetition_penalty_enable;
    float   repetition_penalty;
} tr_sampling_params_t;

/** Scheduler / executor statistics. */
typedef struct {
    size_t  active_requests;
    size_t  queued_requests;
    size_t  completed_requests;
    size_t  total_tokens_generated;
    float   tokens_per_second;
    float   gpu_memory_used_mb;
} tr_executor_stats_t;

/** Create an executor for in-flight batching. */
tr_executor_t* tr_executor_create(tr_runtime_t* runtime, size_t max_active_requests);

/** Submit a generation request. Returns request ID (0 on failure in stub mode). */
tr_request_id_t tr_executor_submit(
    tr_executor_t*              executor,
    const int32_t*              prompt_ids,
    size_t                      prompt_len,
    const tr_sampling_params_t* params
);

/** Wait for the next batch of completions.
 *  token_ids_out: caller-allocated buffer [max_tokens * max_requests]
 *  lengths_out:   number of tokens per completed request [max_requests]
 *  request_ids:   matching request IDs [max_requests]
 *  Returns number of completed requests (0 = no completions yet). */
size_t tr_executor_await_completions(
    tr_executor_t*  executor,
    int32_t*        token_ids_out,
    size_t*         lengths_out,
    tr_request_id_t* request_ids,
    size_t          max_requests
);

/** Cancel a pending request. */
tr_adapter_status_t tr_executor_cancel(tr_executor_t* executor, tr_request_id_t request_id);

/** Get executor statistics. */
tr_adapter_status_t tr_executor_get_stats(tr_executor_t* executor, tr_executor_stats_t* stats);

/** Destroy executor and free resources. */
void tr_executor_destroy(tr_executor_t* executor);

#ifdef __cplusplus
}
#endif

#endif /* XLLM_TENSORRT_ADAPTER_H */
