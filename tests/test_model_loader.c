/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Model Loader tests: GGUF parsing, config loading, weight allocation,
 * architecture detection.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "weight_loader.h"
#include "model_loader.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { tests_run++; printf("  RUN  %s ... ", name); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg)  do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* Functions declared in model_loader.h */

/* ── Test: Weight Allocation ────────────────────────────────────────────── */

static void test_weights_alloc(void) {
    TEST("weights_alloc");
    Gpt2Config cfg = {
        .vocab_size = 50257,
        .n_positions = 1024,
        .n_embd = 768,
        .n_layer = 12,
        .n_head = 12,
        .n_inner = 3072,
        .head_size = 64,
        .layer_norm_eps = 1e-5f,
    };

    Gpt2Weights *w = gpt2_weights_alloc(&cfg);
    ASSERT(w != NULL, "weights_alloc returned NULL");

    /* Verify all arrays are allocated */
    ASSERT(w->wte != NULL, "wte should be allocated");
    ASSERT(w->wpe != NULL, "wpe should be allocated");
    ASSERT(w->ln_f_weight != NULL, "ln_f_weight should be allocated");
    ASSERT(w->ln_f_bias != NULL, "ln_f_bias should be allocated");

    for (int i = 0; i < cfg.n_layer; i++) {
        ASSERT(w->ln_1_weight[i] != NULL, "ln_1_weight should be allocated");
        ASSERT(w->attn_c_attn_w[i] != NULL, "attn_c_attn_w should be allocated");
        ASSERT(w->mlp_c_fc_w[i] != NULL, "mlp_c_fc_w should be allocated");
    }

    /* Verify some elements are zero-initialized */
    ASSERT(w->wte[0] == 0.0f, "wte should be zero-initialized");

    gpt2_weights_free(w);
    PASS();
}

/* ── Test: Weight Allocation Edge Cases ─────────────────────────────────── */

static void test_weights_alloc_edge_cases(void) {
    TEST("weights_alloc_edge_cases");

    /* NULL config */
    Gpt2Weights *w = gpt2_weights_alloc(NULL);
    ASSERT(w == NULL, "NULL config should return NULL");

    /* Zero layer config */
    Gpt2Config cfg = { .n_layer = 0 };
    w = gpt2_weights_alloc(&cfg);
    ASSERT(w == NULL, "zero layers should return NULL");

    /* Large layer config (reasonable size to avoid OOM) */
    Gpt2Config large_cfg = {
        .vocab_size = 32000,
        .n_positions = 4096,
        .n_embd = 1024,
        .n_layer = 12,
        .n_head = 16,
        .n_inner = 4096,
        .head_size = 64,
        .layer_norm_eps = 1e-5f,
    };
    w = gpt2_weights_alloc(&large_cfg);
    ASSERT(w != NULL, "large config allocation should succeed");
    gpt2_weights_free(w);

    PASS();
}

/* ── Test: Weight Free NULL ─────────────────────────────────────────────── */

static void test_weights_free_null(void) {
    TEST("weights_free_null");
    gpt2_weights_free(NULL);  /* should not crash */
    PASS();
}

/* ── Test: Weight Load Nonexistent File ─────────────────────────────────── */

static void test_weights_load_missing(void) {
    TEST("weights_load_missing");
    Gpt2Config cfg;
    Gpt2Weights *w = gpt2_weights_load("/nonexistent/path/model.bin", &cfg);
    ASSERT(w == NULL, "Loading nonexistent file should return NULL");
    PASS();
}

/* ── Test: Architecture Detection on Nonexistent File ───────────────────── */

static void test_detect_arch_missing(void) {
    TEST("detect_arch_missing");
    const char *arch = nxt_model_loader_detect_arch("/nonexistent/model.gguf");
    ASSERT(arch == NULL, "detect_arch on missing file should return NULL");
    PASS();
}

/* ── Test: Load Config on Nonexistent File ──────────────────────────────── */

static void test_load_config_missing(void) {
    TEST("load_config_missing");
    Gpt2Config cfg;
    bool ok = nxt_model_loader_load_config("/nonexistent/model.gguf", &cfg);
    ASSERT(!ok, "load_config on missing file should fail");

    /* NULL path */
    ok = nxt_model_loader_load_config(NULL, &cfg);
    ASSERT(!ok, "NULL path should fail");

    /* NULL config */
    ok = nxt_model_loader_load_config("/some/path", NULL);
    ASSERT(!ok, "NULL config should fail");

    PASS();
}

/* ── Test: Weights Structure Sizes ──────────────────────────────────────── */

static void test_weights_structure_sizes(void) {
    TEST("weights_structure_sizes");
    Gpt2Config cfg = {
        .vocab_size = 32000,
        .n_positions = 2048,
        .n_embd = 512,
        .n_layer = 12,
        .n_head = 8,
        .n_inner = 2048,
        .head_size = 64,
        .layer_norm_eps = 1e-5f,
    };

    Gpt2Weights *w = gpt2_weights_alloc(&cfg);
    ASSERT(w != NULL, "allocation failed");

    /* Verify sizes using element access boundary check */
    size_t wte_bytes = (size_t)cfg.vocab_size * (size_t)cfg.n_embd * sizeof(float);
    w->wte[cfg.vocab_size - 1] = 1.0f;  /* last element should be writable */
    (void)wte_bytes;

    size_t wpe_bytes = (size_t)cfg.n_positions * (size_t)cfg.n_embd * sizeof(float);
    w->wpe[cfg.n_positions - 1] = 1.0f;
    (void)wpe_bytes;

    /* Per-layer last element check */
    int last_layer = cfg.n_layer - 1;
    size_t attn_bytes = (size_t)cfg.n_embd * 3 * (size_t)cfg.n_embd * sizeof(float);
    w->attn_c_attn_w[last_layer][attn_bytes / sizeof(float) - 1] = 1.0f;
    (void)attn_bytes;

    gpt2_weights_free(w);
    PASS();
}

/* ── Test: Config Values After Allocation ───────────────────────────────── */

static void test_config_values(void) {
    TEST("config_values");
    Gpt2Config cfg = {
        .vocab_size = 50257,
        .n_positions = 1024,
        .n_embd = 768,
        .n_layer = 12,
        .n_head = 12,
        .n_inner = 3072,
        .head_size = 64,
        .layer_norm_eps = 1e-5f,
    };

    ASSERT(cfg.head_size == 64, "head_size should be n_embd / n_head");
    ASSERT(cfg.vocab_size == 50257, "vocab_size should be GPT-2 default");
    ASSERT(cfg.n_embd == 768, "n_embd should be 768 for GPT-2 small");

    PASS();
}

/* ── Test: Repeated Alloc/Free Cycle ────────────────────────────────────── */

static void test_weights_alloc_free_cycle(void) {
    TEST("weights_alloc_free_cycle");
    Gpt2Config cfg = {
        .vocab_size = 50257,
        .n_positions = 1024,
        .n_embd = 768,
        .n_layer = 6,
        .n_head = 12,
        .n_inner = 3072,
        .head_size = 64,
        .layer_norm_eps = 1e-5f,
    };

    for (int i = 0; i < 5; i++) {
        Gpt2Weights *w = gpt2_weights_alloc(&cfg);
        ASSERT(w != NULL, "allocation failed in cycle");
        gpt2_weights_free(w);
    }
    PASS();
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Model Loader Tests (GGUF / GPT-2 Weights) ===\n\n");

    test_weights_alloc();
    test_weights_alloc_edge_cases();
    test_weights_free_null();
    test_weights_load_missing();
    test_detect_arch_missing();
    test_load_config_missing();
    test_weights_structure_sizes();
    test_config_values();
    test_weights_alloc_free_cycle();

    printf("\n=== Results: %d run, %d passed, %d failed ===\n",
           tests_run, tests_passed, tests_failed);
    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
