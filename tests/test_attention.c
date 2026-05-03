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

    #undef RUN_TEST

    printf("\n======================================\n");
    if (failures == 0) {
        printf("All tests PASSED.\n");
    } else {
        printf("%d test(s) FAILED.\n", failures);
    }
    return failures;
}
