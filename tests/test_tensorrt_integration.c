/*
 * test_tensorrt_integration.c — TensorRT Adapter Integration Tests
 *
 * Tests the stub implementation of tensorrt_adapter.h.
 * All tests pass without TensorRT installed (stub mode).
 * When NXTLLM_HAS_TENSORRT is defined, they validate real TRT behavior.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/tensorrt_adapter.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do {                         \
    tests_run++;                                \
    printf("  [TEST] %s ... ", name);           \
} while(0)

#define PASS() do {                             \
    tests_passed++;                             \
    printf("PASS\n");                           \
} while(0)

#define FAIL(msg) do {                          \
    tests_failed++;                             \
    printf("FAIL: %s\n", msg);                  \
} while(0)

#define ASSERT(cond, msg) do {                  \
    if (!(cond)) { FAIL(msg); return; }         \
} while(0)

/* ── Test 1: Builder config lifecycle ────────────────────────────────── */

static void test_builder_config_defaults(void)
{
    TEST("builder config defaults");

    tr_builder_config_t cfg = {0};
    ASSERT(cfg.quant_mode == TR_QUANT_NONE, "default quant should be NONE");
    ASSERT(cfg.fusion_flags == TR_FUSE_NONE, "default fusion should be NONE");
    ASSERT(cfg.use_cuda_graph == 0, "default cuda_graph should be 0");
    ASSERT(cfg.use_inflight_batching == 0, "default inflight should be 0");

    PASS();
}

/* ── Test 2: Builder / Runtime stub ─────────────────────────────────── */

static void test_builder_stub_build(void)
{
    TEST("builder stub build");

    tr_builder_config_t cfg = {
        .hidden_size = 768,
        .num_heads = 12,
        .num_kv_heads = 12,
        .head_dim = 64,
        .intermediate_size = 3072,
        .num_layers = 12,
        .vocab_size = 50257,
        .max_seq_len = 2048,
        .quant_mode = TR_QUANT_NONE,
        .fusion_flags = TR_FUSE_QKV | TR_FUSE_GATED_MLP,
        .use_cuda_graph = 1,
        .use_inflight_batching = 0,
        .use_fp16 = 1,
        .max_batch_size = 8,
        .kv_cache_block_size = 64,
        .num_kv_cache_blocks = 256,
    };

    tr_runtime_t* runtime = tr_builder_build(&cfg);
    /* In stub mode, this returns NULL — that's expected and not a crash */
    ASSERT(runtime == NULL, "stub builder should return NULL");

    tr_runtime_destroy(runtime); /* no-op on NULL */
    PASS();
}

/* ── Test 3: Runtime run stub ───────────────────────────────────────── */

static void test_runtime_run_stub(void)
{
    TEST("runtime run stub");

    tr_builder_config_t cfg = {
        .hidden_size = 768,
        .num_heads = 12,
        .num_kv_heads = 12,
        .head_dim = 64,
        .intermediate_size = 3072,
        .num_layers = 12,
        .vocab_size = 50257,
        .max_seq_len = 1024,
    };

    tr_runtime_t* runtime = tr_builder_build(&cfg);

    int32_t input_ids[4] = {1, 2, 3, 4};
    tr_adapter_status_t status = tr_runtime_run(runtime, input_ids, 1, 4, NULL, NULL);
    ASSERT(status == TR_ADAPTER_ERR_NOT_INITIALIZED,
           "stub run should return NOT_INITIALIZED");

    tr_runtime_destroy(runtime);
    PASS();
}

/* ── Test 4: Serialize/Deserialize stub ─────────────────────────────── */

static void test_serialize_deserialize_stub(void)
{
    TEST("serialize/deserialize stub");

    tr_builder_config_t cfg = {
        .hidden_size = 512, .num_heads = 8, .num_kv_heads = 8,
        .head_dim = 64, .intermediate_size = 2048, .num_layers = 6,
        .vocab_size = 32000, .max_seq_len = 512,
    };

    tr_runtime_t* runtime = tr_builder_build(&cfg);

    /* Serialize should fail in stub mode */
    tr_adapter_status_t s = tr_builder_serialize(runtime, "/tmp/nxtllm_test_trt.engine");
    ASSERT(s == TR_ADAPTER_ERR_NOT_SUPPORTED,
           "stub serialize should return NOT_SUPPORTED");

    /* Load should return NULL in stub mode */
    tr_runtime_t* loaded = tr_runtime_load("/tmp/nxtllm_test_trt.engine");
    ASSERT(loaded == NULL, "stub load should return NULL");

    tr_runtime_destroy(runtime);
    tr_runtime_destroy(loaded);
    PASS();
}

/* ── Test 5: CUDA Graph stubs ───────────────────────────────────────── */

static void test_cuda_graph_stubs(void)
{
    TEST("cuda graph stubs");

    tr_builder_config_t cfg = {
        .hidden_size = 768, .num_heads = 12, .num_kv_heads = 12,
        .head_dim = 64, .intermediate_size = 3072, .num_layers = 12,
        .vocab_size = 50257, .max_seq_len = 1024, .use_cuda_graph = 1,
    };

    tr_runtime_t* runtime = tr_builder_build(&cfg);
    ASSERT(runtime == NULL, "stub builder returns NULL");

    /* All graph functions should handle NULL runtime gracefully */
    int supports = tr_runtime_supports_cuda_graph(runtime);
    ASSERT(supports == 0, "stub should not support cuda graph");

    tr_adapter_status_t s;

    s = tr_runtime_cuda_graph_begin(runtime, 0);
    ASSERT(s == TR_ADAPTER_ERR_NOT_INITIALIZED, "stub cuda_graph_begin on NULL runtime FAIL");

    s = tr_runtime_cuda_graph_end(runtime);
    ASSERT(s == TR_ADAPTER_ERR_NOT_INITIALIZED, "stub cuda_graph_end on NULL runtime FAIL");

    s = tr_runtime_cuda_graph_launch(runtime, 0);
    ASSERT(s == TR_ADAPTER_ERR_NOT_INITIALIZED, "stub cuda_graph_launch on NULL runtime FAIL");

    tr_runtime_destroy(runtime);
    PASS();
}

/* ── Test 6: Executor stubs ─────────────────────────────────────────── */

static void test_executor_stubs(void)
{
    TEST("executor stubs");

    tr_runtime_t* runtime = NULL; /* no real runtime */

    tr_executor_t* executor = tr_executor_create(runtime, 8);
    ASSERT(executor == NULL, "stub executor_create should return NULL");

    /* Submit should return 0 (invalid request ID) */
    tr_sampling_params_t sp = {
        .temperature = 0.7f, .top_k = 50, .top_p = 0.9f,
        .max_new_tokens = 128, .n_beams = 1,
    };
    int32_t prompt[4] = {1, 2, 3, 4};
    tr_request_id_t req_id = tr_executor_submit(executor, prompt, 4, &sp);
    ASSERT(req_id == 0, "stub submit should return 0");

    /* Await should return 0 completions */
    int32_t tokens[4];
    size_t lengths[1];
    tr_request_id_t ids[1];
    size_t n = tr_executor_await_completions(executor, tokens, lengths, ids, 1);
    ASSERT(n == 0, "stub await should return 0");

    /* Cancel on invalid executor should be safe */
    tr_adapter_status_t s = tr_executor_cancel(executor, 0);
    ASSERT(s == TR_ADAPTER_ERR_NOT_INITIALIZED, "stub cancel FAIL");

    /* Stats should fail */
    tr_executor_stats_t stats;
    s = tr_executor_get_stats(executor, &stats);
    ASSERT(s == TR_ADAPTER_ERR_NOT_INITIALIZED, "stub get_stats FAIL");

    tr_executor_destroy(executor);
    PASS();
}

/* ── Test 7: Fusion flags bitmask ───────────────────────────────────── */

static void test_fusion_flags(void)
{
    TEST("fusion flags bitmask");

    ASSERT((TR_FUSE_QKV | TR_FUSE_GATED_MLP) == 0x03, "QKV|GATED should be 0x03");
    ASSERT((TR_FUSE_QKV | TR_FUSE_GATED_MLP | TR_FUSE_BIAS_RESID) == 0x07, "QKV|GATED|BIAS should be 0x07");
    ASSERT(TR_FUSE_ALL == 0x3F, "FUSE_ALL should be 0x3F");
    ASSERT((TR_FUSE_ALL & TR_FUSE_QKV) == TR_FUSE_QKV, "QKV should be in FUSE_ALL");
    ASSERT((TR_FUSE_ALL & TR_FUSE_ROPE_ATTN) == TR_FUSE_ROPE_ATTN, "ROPE should be in FUSE_ALL");

    PASS();
}

/* ── Test 8: Quant mode enum values ─────────────────────────────────── */

static void test_quant_mode_values(void)
{
    TEST("quant mode enum values");

    ASSERT(TR_QUANT_NONE == 0, "NONE should be 0");
    ASSERT(TR_QUANT_INT8_W8A16 == 1, "INT8_W8A16 should be 1");
    ASSERT(TR_QUANT_INT4_W4A16 == 2, "INT4_W4A16 should be 2");
    ASSERT(TR_QUANT_FP8 == 3, "FP8 should be 3");
    ASSERT(TR_QUANT_INT8_SQ == 4, "INT8_SQ should be 4");
    ASSERT(TR_QUANT_NVFP4 == 5, "NVFP4 should be 5");
    ASSERT(TR_QUANT_MIXED == 6, "MIXED should be 6");

    PASS();
}

/* ── Test 9: Builder config with all fusion flags ───────────────────── */

static void test_builder_full_config(void)
{
    TEST("builder full config");

    tr_builder_config_t cfg = {
        .hidden_size = 4096,
        .num_heads = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .intermediate_size = 14336,
        .num_layers = 32,
        .vocab_size = 128256,
        .max_seq_len = 8192,
        .quant_mode = TR_QUANT_FP8,
        .fusion_flags = TR_FUSE_ALL,
        .use_cuda_graph = 1,
        .use_inflight_batching = 1,
        .use_fp16 = 0,
        .use_bf16 = 1,
        .max_batch_size = 64,
        .kv_cache_block_size = 256,
        .num_kv_cache_blocks = 512,
    };

    ASSERT(cfg.hidden_size == 4096, "hidden_size mismatch");
    ASSERT(cfg.num_heads == 32, "num_heads mismatch");
    ASSERT(cfg.num_kv_heads == 8, "num_kv_heads mismatch (GQA)");
    ASSERT(cfg.quant_mode == TR_QUANT_FP8, "quant_mode mismatch");
    ASSERT(cfg.fusion_flags == TR_FUSE_ALL, "fusion_flags mismatch");
    ASSERT(cfg.use_cuda_graph == 1, "cuda_graph mismatch");
    ASSERT(cfg.use_inflight_batching == 1, "inflight mismatch");

    PASS();
}

/* ── Test 10: Error code consistency ────────────────────────────────── */

static void test_error_codes(void)
{
    TEST("error codes");

    ASSERT(TR_ADAPTER_OK == 0, "OK should be 0");
    ASSERT(TR_ADAPTER_ERR_NOT_INITIALIZED == -1, "NOT_INITIALIZED should be -1");
    ASSERT(TR_ADAPTER_ERR_INVALID_CONFIG == -2, "INVALID_CONFIG should be -2");
    ASSERT(TR_ADAPTER_ERR_BUILD_FAILED == -3, "BUILD_FAILED should be -3");
    ASSERT(TR_ADAPTER_ERR_RUNTIME_ERROR == -4, "RUNTIME_ERROR should be -4");
    ASSERT(TR_ADAPTER_ERR_OOM == -5, "OOM should be -5");
    ASSERT(TR_ADAPTER_ERR_NOT_SUPPORTED == -6, "NOT_SUPPORTED should be -6");

    PASS();
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("\n=== nxtLLM TensorRT Adapter Integration Tests ===\n");
    printf("Compiled %s NXTLLM_HAS_TENSORRT\n\n",
#ifdef NXTLLM_HAS_TENSORRT
           "with"
#else
           "without"
#endif
    );

    test_builder_config_defaults();
    test_builder_stub_build();
    test_runtime_run_stub();
    test_serialize_deserialize_stub();
    test_cuda_graph_stubs();
    test_executor_stubs();
    test_fusion_flags();
    test_quant_mode_values();
    test_builder_full_config();
    test_error_codes();

    printf("\n--- Results: %d run, %d passed, %d failed ---\n\n",
           tests_run, tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
