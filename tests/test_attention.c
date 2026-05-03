/*
 * test_attention.c — Validate nxtLLM paged attention operator API
 *
 * This is a lightweight C-level smoke test.  Real numerical validation
 * should compare against a PyTorch reference on GPU hardware.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "operator_api.h"

#ifdef CUDART_VERSION
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

/* ── Minimal CUDA host helpers (no runtime dep in this test stub) ─── */

static int test_paged_attention_signature(void) {
    /* Verify that function pointer is linkable */
    void* fn = (void*)nxt_paged_attention;
    assert(fn != NULL);  /* linker resolved */
    return 0;
}

static int test_silu_and_mul_signature(void) {
    void* fn = (void*)nxt_silu_and_mul;
    assert(fn != NULL);
    return 0;
}

static int test_mul_and_silu_signature(void) {
    void* fn = (void*)nxt_mul_and_silu;
    assert(fn != NULL);
    return 0;
}

static int test_gelu_and_mul_signature(void) {
    void* fn = (void*)nxt_gelu_and_mul;
    assert(fn != NULL);
    return 0;
}

static int test_gelu_tanh_and_mul_signature(void) {
    void* fn = (void*)nxt_gelu_tanh_and_mul;
    assert(fn != NULL);
    return 0;
}

static int test_gelu_elementwise_signature(void) {
    void* fn = (void*)nxt_gelu_elementwise;
    assert(fn != NULL);
    return 0;
}

static int test_silu_elementwise_signature(void) {
    void* fn = (void*)nxt_silu_elementwise;
    assert(fn != NULL);
    return 0;
}

/* ── Argument layout sanity ───────────────────────────────────────── */

static int test_paged_attention_arg_layout(void) {
    /*
     * Verify that the operator_api.h signature is callable with
     * zero-initialized pointers (no GPU launch — only CPU-side
     * argument passing is exercised here).
     *
     * Full GPU integration tests require a CUDA-capable device and
     * should be run separately (e.g. via ctest on GPU CI).
     */
    int block_table = 0;
    int seq_len = 0;

    nxt_paged_attention(
        NULL, NULL, NULL, NULL,
        &block_table, &seq_len,
        0, 0, 0,   /* num_seqs, num_heads, head_size */
        0, 0.0f,   /* num_kv_heads, scale */
        0, 0, 0,   /* max_num_blocks_per_seq, block_size, dtype_size */
        0, 0,      /* kv_block_stride, kv_head_stride */
        NULL);     /* stream */

    return 0;
}

/* ══════════════════════════════════════════════════════════════════════
 * CUDA kernel-launch tests (compiled only when CUDA runtime available)
 * ══════════════════════════════════════════════════════════════════════ */

#ifdef CUDART_VERSION

static int test_paged_attention_cuda_launch(void) {
    /*
     * Smoke-test the host launcher dispatch with a real GPU kernel
     * launch.  Allocates minimal device buffers for a single-head,
     * single-seq, single-block forward pass and checks that the
     * kernel completes without error.
     */
    const int num_seqs = 1;
    const int num_heads = 1;
    const int head_size = 64;
    const int num_kv_heads = 1;
    const int max_num_blocks_per_seq = 1;
    const int block_size = 16;
    const int dtype_size = 2;  // fp16

    const int q_stride = num_seqs * num_heads * head_size;
    const int kv_block_stride = num_kv_heads * block_size * head_size;
    const int kv_head_stride = block_size * head_size;

    const size_t out_bytes = num_seqs * num_heads * head_size * dtype_size;
    const size_t q_bytes = out_bytes;
    const size_t kv_bytes = max_num_blocks_per_seq * kv_block_stride * dtype_size;

    __half *d_out, *d_query, *d_key_cache, *d_value_cache;
    int *d_block_tables, *d_seq_lens;

    cudaError_t err;

    err = cudaMalloc(&d_out, out_bytes);
    if (err != cudaSuccess) goto cleanup_none;
    err = cudaMalloc(&d_query, q_bytes);
    if (err != cudaSuccess) goto cleanup_out;
    err = cudaMalloc(&d_key_cache, kv_bytes);
    if (err != cudaSuccess) goto cleanup_query;
    err = cudaMalloc(&d_value_cache, kv_bytes);
    if (err != cudaSuccess) goto cleanup_key;
    err = cudaMalloc(&d_block_tables, num_seqs * max_num_blocks_per_seq * sizeof(int));
    if (err != cudaSuccess) goto cleanup_value;
    err = cudaMalloc(&d_seq_lens, num_seqs * sizeof(int));
    if (err != cudaSuccess) goto cleanup_block;

    err = cudaMemset(d_out, 0, out_bytes);
    if (err != cudaSuccess) goto cleanup_all;
    err = cudaMemset(d_query, 0, q_bytes);
    if (err != cudaSuccess) goto cleanup_all;
    err = cudaMemset(d_key_cache, 0, kv_bytes);
    if (err != cudaSuccess) goto cleanup_all;
    err = cudaMemset(d_value_cache, 0, kv_bytes);
    if (err != cudaSuccess) goto cleanup_all;

    {
        int h_block_table[1] = {0};
        int h_seq_lens[1] = {block_size};
        err = cudaMemcpy(d_block_tables, h_block_table, sizeof(h_block_table), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup_all;
        err = cudaMemcpy(d_seq_lens, h_seq_lens, sizeof(h_seq_lens), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup_all;
    }

    cudaStream_t stream = NULL;
    nxt_paged_attention(
        d_out, d_query,
        d_key_cache, d_value_cache,
        d_block_tables, d_seq_lens,
        num_seqs, num_heads, head_size,
        num_kv_heads, 1.0f / sqrtf((float)head_size),
        max_num_blocks_per_seq,
        block_size, dtype_size,
        kv_block_stride, kv_head_stride,
        stream);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup_all;

    cudaFree(d_seq_lens);
    cudaFree(d_block_tables);
    cudaFree(d_value_cache);
    cudaFree(d_key_cache);
    cudaFree(d_query);
    cudaFree(d_out);
    return 0;

cleanup_all:
    cudaFree(d_seq_lens);
cleanup_block:
    cudaFree(d_block_tables);
cleanup_value:
    cudaFree(d_value_cache);
cleanup_key:
    cudaFree(d_key_cache);
cleanup_query:
    cudaFree(d_query);
cleanup_out:
    cudaFree(d_out);
cleanup_none:
    printf(" (CUDA error: %s) ", cudaGetErrorString(err));
    return 1;
}

static int test_paged_attention_head_sizes(void) {
    /*
     * Verify that every supported head_size dispatches without error.
     */
    const int supported_sizes[] = {32, 64, 80, 96, 112, 128, 192, 256};
    const int num_sizes = sizeof(supported_sizes) / sizeof(supported_sizes[0]);
    int failures = 0;

    for (int i = 0; i < num_sizes; i++) {
        int hs = supported_sizes[i];
        const int num_seqs = 1;
        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int max_blocks = 1;
        const int bs = 16;
        const int ds = 2;

        const int q_stride = num_seqs * num_heads * hs;
        const int kv_block_stride = num_kv_heads * bs * hs;
        const int kv_head_stride = bs * hs;

        size_t out_bytes = num_seqs * num_heads * hs * ds;
        size_t q_bytes = out_bytes;
        size_t kv_bytes = max_blocks * kv_block_stride * ds;

        __half *d_out, *d_query, *d_key, *d_val;
        int *d_bt, *d_sl;
        cudaError_t err;

        if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) { failures++; continue; }
        if (cudaMalloc(&d_query, q_bytes) != cudaSuccess) { cudaFree(d_out); failures++; continue; }
        if (cudaMalloc(&d_key, kv_bytes) != cudaSuccess) { cudaFree(d_query); cudaFree(d_out); failures++; continue; }
        if (cudaMalloc(&d_val, kv_bytes) != cudaSuccess) { cudaFree(d_key); cudaFree(d_query); cudaFree(d_out); failures++; continue; }
        if (cudaMalloc(&d_bt, num_seqs * max_blocks * sizeof(int)) != cudaSuccess) {
            cudaFree(d_val); cudaFree(d_key); cudaFree(d_query); cudaFree(d_out); failures++; continue;
        }
        if (cudaMalloc(&d_sl, num_seqs * sizeof(int)) != cudaSuccess) {
            cudaFree(d_bt); cudaFree(d_val); cudaFree(d_key); cudaFree(d_query); cudaFree(d_out); failures++; continue;
        }

        cudaMemset(d_out, 0, out_bytes);
        cudaMemset(d_query, 0, q_bytes);
        cudaMemset(d_key, 0, kv_bytes);
        cudaMemset(d_val, 0, kv_bytes);
        int h_bt = 0, h_sl = bs;
        cudaMemcpy(d_bt, &h_bt, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sl, &h_sl, sizeof(int), cudaMemcpyHostToDevice);

        nxt_paged_attention(d_out, d_query, d_key, d_val, d_bt, d_sl,
                            num_seqs, num_heads, hs, num_kv_heads,
                            1.0f / sqrtf((float)hs),
                            max_blocks, bs, ds,
                            kv_block_stride, kv_head_stride, NULL);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) failures++;

        cudaFree(d_sl); cudaFree(d_bt);
        cudaFree(d_val); cudaFree(d_key);
        cudaFree(d_query); cudaFree(d_out);
    }
    return failures;
}

#endif  /* CUDART_VERSION */

/* ══════════════════════════════════════════════════════════════════════ */

int main(void) {
    int failures = 0;

    printf("nxtLLM operator API — signature tests\n");
    printf("======================================\n\n");

    #define RUN_TEST(t) do {                              \
        printf("  %-42s ... ", #t);                       \
        if (test_##t() == 0) {                            \
            printf("PASS\n");                             \
        } else {                                          \
            printf("FAIL\n");                             \
            failures++;                                   \
        }                                                 \
    } while (0)

    RUN_TEST(paged_attention_signature);
    RUN_TEST(silu_and_mul_signature);
    RUN_TEST(mul_and_silu_signature);
    RUN_TEST(gelu_and_mul_signature);
    RUN_TEST(gelu_tanh_and_mul_signature);
    RUN_TEST(gelu_elementwise_signature);
    RUN_TEST(silu_elementwise_signature);
    RUN_TEST(paged_attention_arg_layout);

#ifdef CUDART_VERSION
    RUN_TEST(paged_attention_cuda_launch);
    RUN_TEST(paged_attention_head_sizes);
#endif

    #undef RUN_TEST

    printf("\n======================================\n");
    if (failures == 0) {
        printf("All tests PASSED.\n");
    } else {
        printf("%d test(s) FAILED.\n", failures);
    }
    return failures;
}
