/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * test_attention_bench.c — Performance comparison between paged attention
 * kernels (v1 baseline vs FlashInfer-style adapter).
 *
 * Build: cmake -DUSE_CUDA=ON -DUSE_FLASHINFER=ON ..
 * Run:   ./test_attention_bench [--iterations N] [--batch B] [--ctx L]
 *
 * The test allocates GPU buffers, runs warmup iterations, then measures
 * average kernel time for both implementations.  Reports speedup ratio.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "operator_api.h"

#ifdef CUDART_VERSION
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

/* ── Default test parameters ────────────────────────────────────────── */

#define DEFAULT_NUM_ITERS   100
#define DEFAULT_BATCH_SIZE  16
#define DEFAULT_CTX_LEN     1024
#define DEFAULT_NUM_HEADS   32
#define DEFAULT_NUM_KV_HEADS 8
#define DEFAULT_HEAD_SIZE   128
#define DEFAULT_BLOCK_SIZE  16
#define DEFAULT_DTYPE_SIZE   2   /* fp16 */

/* ── CUDA event-based timer ─────────────────────────────────────────── */

#ifdef CUDART_VERSION

typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} gpu_timer_t;

static void gpu_timer_init(gpu_timer_t* t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

static void gpu_timer_destroy(gpu_timer_t* t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

static void gpu_timer_start(gpu_timer_t* t, cudaStream_t stream) {
    cudaEventRecord(t->start, stream);
}

static float gpu_timer_stop(gpu_timer_t* t, cudaStream_t stream) {
    float ms;
    cudaEventRecord(t->stop, stream);
    cudaEventSynchronize(t->stop);
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}

/* ── Device buffer allocation helper ────────────────────────────────── */

typedef struct {
    __half* out;
    __half* query;
    __half* key_cache;
    __half* value_cache;
    int*    block_tables;
    int*    seq_lens;
    int     num_seqs;
    int     num_heads;
    int     head_size;
    int     num_kv_heads;
    int     max_pages_per_seq;
    int     block_size;
    int     seq_len;
} bench_buffers_t;

static cudaError_t bench_buffers_alloc(
    bench_buffers_t* b,
    int num_seqs, int num_heads, int head_size,
    int num_kv_heads, int seq_len, int block_size)
{
    b->num_seqs = num_seqs;
    b->num_heads = num_heads;
    b->head_size = head_size;
    b->num_kv_heads = num_kv_heads;
    b->block_size = block_size;
    b->seq_len = seq_len;
    b->max_pages_per_seq = (seq_len + block_size - 1) / block_size;
    const int total_pages = b->max_pages_per_seq * num_seqs;

    cudaError_t err;
    size_t out_bytes = (size_t)num_seqs * num_heads * head_size * sizeof(__half);
    size_t kv_bytes  = (size_t)total_pages * num_kv_heads * block_size * head_size * sizeof(__half);
    size_t bt_bytes  = (size_t)num_seqs * b->max_pages_per_seq * sizeof(int);
    size_t sl_bytes  = (size_t)num_seqs * sizeof(int);

    #define ALLOC_OR_FAIL(ptr, bytes, label) do {          \
        err = cudaMalloc(&(ptr), bytes);                    \
        if (err != cudaSuccess) goto label;                 \
    } while(0)

    ALLOC_OR_FAIL(b->out,         out_bytes, cleanup_none);
    ALLOC_OR_FAIL(b->query,       out_bytes, cleanup_out);
    ALLOC_OR_FAIL(b->key_cache,   kv_bytes,  cleanup_query);
    ALLOC_OR_FAIL(b->value_cache, kv_bytes,  cleanup_key);
    ALLOC_OR_FAIL(b->block_tables, bt_bytes, cleanup_value);
    ALLOC_OR_FAIL(b->seq_lens,    sl_bytes,  cleanup_block);

    /* zero-initialize */
    cudaMemset(b->out, 0, out_bytes);
    cudaMemset(b->query, 0, out_bytes);
    cudaMemset(b->key_cache, 0, kv_bytes);
    cudaMemset(b->value_cache, 0, kv_bytes);

    /* fill block_tables and seq_lens on host then copy */
    {
        int* h_bt = (int*)malloc(bt_bytes);
        int* h_sl = (int*)malloc(sl_bytes);
        for (int s = 0; s < num_seqs; s++) {
            h_sl[s] = seq_len;
            for (int p = 0; p < b->max_pages_per_seq; p++) {
                h_bt[s * b->max_pages_per_seq + p] = s * b->max_pages_per_seq + p;
            }
        }
        cudaMemcpy(b->block_tables, h_bt, bt_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(b->seq_lens, h_sl, sl_bytes, cudaMemcpyHostToDevice);
        free(h_bt);
        free(h_sl);
    }

    return cudaSuccess;

cleanup_block:
    cudaFree(b->seq_lens);
cleanup_value:
    cudaFree(b->value_cache);
cleanup_key:
    cudaFree(b->key_cache);
cleanup_query:
    cudaFree(b->query);
cleanup_out:
    cudaFree(b->out);
cleanup_none:
    return err;
    #undef ALLOC_OR_FAIL
}

static void bench_buffers_free(bench_buffers_t* b) {
    cudaFree(b->out);
    cudaFree(b->query);
    cudaFree(b->key_cache);
    cudaFree(b->value_cache);
    cudaFree(b->block_tables);
    cudaFree(b->seq_lens);
}

/* ── Benchmark runner ───────────────────────────────────────────────── */

typedef void (*attn_kernel_fn)(
    void*, const void*, const void*, const void*,
    const int*, const int*,
    int, int, int, int, float, int, int, int, int, int,
    nxt_stream_t);

static double run_benchmark(
    attn_kernel_fn kernel_fn,
    bench_buffers_t* b,
    const char* name,
    int num_iters,
    int warmup_iters)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    gpu_timer_t timer;
    gpu_timer_init(&timer);

    const int q_stride = b->num_seqs * b->num_heads * b->head_size;
    const int kv_block_stride = b->num_kv_heads * b->block_size * b->head_size;
    const int kv_head_stride = b->block_size * b->head_size;
    const float scale = 1.0f / sqrtf((float)b->head_size);

    /* warmup */
    for (int i = 0; i < warmup_iters; i++) {
        kernel_fn(
            b->out, b->query, b->key_cache, b->value_cache,
            b->block_tables, b->seq_lens,
            b->num_seqs, b->num_heads, b->head_size,
            b->num_kv_heads, scale,
            b->max_pages_per_seq, b->block_size, DEFAULT_DTYPE_SIZE,
            kv_block_stride, kv_head_stride, stream);
    }
    cudaDeviceSynchronize();

    /* timed iterations */
    gpu_timer_start(&timer, stream);
    for (int i = 0; i < num_iters; i++) {
        kernel_fn(
            b->out, b->query, b->key_cache, b->value_cache,
            b->block_tables, b->seq_lens,
            b->num_seqs, b->num_heads, b->head_size,
            b->num_kv_heads, scale,
            b->max_pages_per_seq, b->block_size, DEFAULT_DTYPE_SIZE,
            kv_block_stride, kv_head_stride, stream);
    }
    float total_ms = gpu_timer_stop(&timer, stream);
    double avg_us = (double)(total_ms * 1000.0) / num_iters;

    printf("  %-42s  %8.1f us  (%d iters, batch=%d, ctx=%d, heads=%d/%d, hd=%d)\n",
           name, avg_us, num_iters,
           b->num_seqs, b->seq_len,
           b->num_heads, b->num_kv_heads,
           b->head_size);

    gpu_timer_destroy(&timer);
    cudaStreamDestroy(stream);

    return avg_us;
}

/* ── Test: signature validation ─────────────────────────────────────── */

static int test_flash_adapter_signature(void) {
    void* fn = (void*)nxt_paged_attention_flash;
    assert(fn != NULL);
    return 0;
}

static int test_flash_adapter_null_launch(void) {
    /* Verify the host dispatcher handles null pointers gracefully */
    int bt = 0, sl = 0;
    nxt_paged_attention_flash(
        NULL, NULL, NULL, NULL,
        &bt, &sl,
        0, 1, 64,
        1, 0.125f,
        1, 16, 2,
        1024, 1024, NULL);
    return 0;
}

/* ── Test: benchmark comparison ─────────────────────────────────────── */

static int test_benchmark_comparison(
    int num_iters, int batch_size, int ctx_len,
    int num_heads, int num_kv_heads, int head_size)
{
    printf("\n─── Benchmark: v1 vs FlashInfer-style ───\n\n");

    bench_buffers_t buffers;
    cudaError_t err = bench_buffers_alloc(
        &buffers, batch_size, num_heads, head_size,
        num_kv_heads, ctx_len, DEFAULT_BLOCK_SIZE);

    if (err != cudaSuccess) {
        printf("  GPU allocation failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int warmup_iters = num_iters / 10;
    if (warmup_iters < 3) warmup_iters = 3;

    double v1_us = run_benchmark(
        nxt_paged_attention, &buffers,
        "nxt_paged_attention (v1 baseline)", num_iters, warmup_iters);

    double fl_us = run_benchmark(
        nxt_paged_attention_flash, &buffers,
        "nxt_paged_attention_flash (FlashInfer-style)", num_iters, warmup_iters);

    if (v1_us > 0.0 && fl_us > 0.0) {
        double speedup = v1_us / fl_us;
        printf("\n  Speedup: %.2fx %s\n",
               speedup,
               (speedup > 1.0) ? "(FlashInfer-style faster)" :
               (speedup < 1.0) ? "(v1 faster)" : "(equal)");
    }

    bench_buffers_free(&buffers);
    return 0;
}

/* ── Test: correctness smoke test ───────────────────────────────────── */

static int test_correctness_smoke(void) {
    /*
     * Minimal correctness check: run both kernels on the same tiny input
     * and compare outputs within tolerance.  This is NOT a full numerical
     * test — that requires a PyTorch golden reference.
     */
    printf("\n─── Correctness smoke test ───\n\n");

    const int num_seqs = 1;
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_size = 64;
    const int seq_len = 32;
    const int block_size = 16;
    const int max_pages = (seq_len + block_size - 1) / block_size;
    const int q_stride = num_seqs * num_heads * head_size;
    const int kv_block_stride = num_kv_heads * block_size * head_size;
    const int kv_head_stride = block_size * head_size;
    const float scale = 1.0f / sqrtf((float)head_size);

    const size_t out_bytes = (size_t)num_seqs * num_heads * head_size * sizeof(__half);
    const size_t kv_bytes  = (size_t)max_pages * kv_block_stride * sizeof(__half);
    const size_t bt_bytes  = (size_t)num_seqs * max_pages * sizeof(int);
    const size_t sl_bytes  = (size_t)num_seqs * sizeof(int);

    __half *d_out_v1, *d_out_fl, *d_query, *d_key, *d_val;
    int *d_bt, *d_sl;

    cudaMalloc(&d_out_v1, out_bytes);
    cudaMalloc(&d_out_fl, out_bytes);
    cudaMalloc(&d_query, out_bytes);
    cudaMalloc(&d_key, kv_bytes);
    cudaMalloc(&d_val, kv_bytes);
    cudaMalloc(&d_bt, bt_bytes);
    cudaMalloc(&d_sl, sl_bytes);

    cudaMemset(d_out_v1, 0, out_bytes);
    cudaMemset(d_out_fl, 0, out_bytes);
    cudaMemset(d_query, 0, out_bytes);
    cudaMemset(d_key, 0, kv_bytes);
    cudaMemset(d_val, 0, kv_bytes);

    int h_bt[max_pages];
    int h_sl[1] = {seq_len};
    for (int p = 0; p < max_pages; p++) h_bt[p] = p;
    cudaMemcpy(d_bt, h_bt, bt_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sl, h_sl, sl_bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Run both kernels */
    nxt_paged_attention(
        d_out_v1, d_query, d_key, d_val, d_bt, d_sl,
        num_seqs, num_heads, head_size, num_kv_heads, scale,
        max_pages, block_size, DEFAULT_DTYPE_SIZE,
        kv_block_stride, kv_head_stride, stream);

    nxt_paged_attention_flash(
        d_out_fl, d_query, d_key, d_val, d_bt, d_sl,
        num_seqs, num_heads, head_size, num_kv_heads, scale,
        max_pages, block_size, DEFAULT_DTYPE_SIZE,
        kv_block_stride, kv_head_stride, stream);

    cudaDeviceSynchronize();

    /* Compare outputs (element-wise max difference) */
    __half* h_out_v1 = (__half*)malloc(out_bytes);
    __half* h_out_fl = (__half*)malloc(out_bytes);
    cudaMemcpy(h_out_v1, d_out_v1, out_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_fl, d_out_fl, out_bytes, cudaMemcpyDeviceToHost);

    double max_diff = 0.0;
    double sum_diff = 0.0;
    int num_elems = num_seqs * num_heads * head_size;
    for (int i = 0; i < num_elems; i++) {
        double diff = fabs((double)__half2float(h_out_v1[i]) -
                           (double)__half2float(h_out_fl[i]));
        max_diff = fmax(max_diff, diff);
        sum_diff += diff;
    }
    double mean_diff = sum_diff / num_elems;

    printf("  Elements compared: %d\n", num_elems);
    printf("  Max  absolute difference: %.6e\n", max_diff);
    printf("  Mean absolute difference: %.6e\n", mean_diff);

    /* NOTE: For zero-initialized inputs, both kernels should produce
       identical (zero) output. A mismatch indicates a dispatch error. */
    int ok = (max_diff < 1e-3) ? 0 : 1;

    free(h_out_v1);
    free(h_out_fl);
    cudaFree(d_out_v1);
    cudaFree(d_out_fl);
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_val);
    cudaFree(d_bt);
    cudaFree(d_sl);
    cudaStreamDestroy(stream);

    return ok;
}

#endif  /* CUDART_VERSION */


/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {
    int num_iters       = DEFAULT_NUM_ITERS;
    int batch_size      = DEFAULT_BATCH_SIZE;
    int ctx_len         = DEFAULT_CTX_LEN;
    int num_heads       = DEFAULT_NUM_HEADS;
    int num_kv_heads    = DEFAULT_NUM_KV_HEADS;
    int head_size       = DEFAULT_HEAD_SIZE;

    /* Parse command line arguments */
    for (int i = 1; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--iterations") == 0) {
            num_iters = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--batch") == 0) {
            batch_size = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--ctx") == 0) {
            ctx_len = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--heads") == 0) {
            num_heads = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--kv-heads") == 0) {
            num_kv_heads = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--head-size") == 0) {
            head_size = atoi(argv[i + 1]);
        }
    }

    printf("nxtLLM Attention Kernel Benchmark\n");
    printf("==================================\n");
    printf("  Iterations:    %d\n", num_iters);
    printf("  Batch size:    %d\n", batch_size);
    printf("  Context len:   %d\n", ctx_len);
    printf("  Q heads:       %d\n", num_heads);
    printf("  KV heads:      %d\n", num_kv_heads);
    printf("  Head dim:      %d\n", head_size);
    printf("  GQA group:     %d\n", num_heads / num_kv_heads);

    int failures = 0;

    #define RUN_TEST(t, ...) do {                                \
        printf("\n  %s ... ", #t);                               \
        int rc = test_##t(__VA_ARGS__);                          \
        if (rc == 0) {                                          \
            printf("PASS\n");                                   \
        } else {                                                \
            printf("FAIL\n");                                   \
            failures++;                                         \
        }                                                       \
    } while (0)

#ifdef CUDART_VERSION
    RUN_TEST(flash_adapter_signature);
    RUN_TEST(flash_adapter_null_launch);
    RUN_TEST(correctness_smoke);
    RUN_TEST(benchmark_comparison,
             num_iters, batch_size, ctx_len,
             num_heads, num_kv_heads, head_size);
#else
    printf("\n  CUDA runtime not available. "
           "Build with -DUSE_CUDA=ON -DUSE_FLASHINFER=ON to run benchmarks.\n");
#endif

    #undef RUN_TEST

    printf("\n==================================\n");
    if (failures == 0) {
        printf("All tests PASSED.\n");
    } else {
        printf("%d test(s) FAILED.\n", failures);
    }
    return failures;
}
