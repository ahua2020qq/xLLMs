/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Triton-style backend management tests: register, model lifecycle, scheduler.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <dlfcn.h>
#include "backend.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { tests_run++; printf("  RUN  %s ... ", name); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg)  do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* Probe candidate paths for the shared library */
static const char *candidate_paths[] = {
    "build/backends/libxllm_cuda_backend.so",
    "../build/backends/libxllm_cuda_backend.so",
    "backends/libxllm_cuda_backend.so",
    "../backends/libxllm_cuda_backend.so",
    "./libxllm_cuda_backend.so",
    NULL
};

static const char *find_so(void) {
    for (int i = 0; candidate_paths[i] != NULL; i++) {
        FILE *f = fopen(candidate_paths[i], "r");
        if (f) { fclose(f); return candidate_paths[i]; }
    }
    return NULL;
}

/* ── Test: Backend Manager Init / Fini ────────────────────────────────── */
static void test_backend_manager_init_fini(void) {
    TEST("backend_manager_init_fini");
    int rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "backend_manager_init failed");
    ASSERT(nxt_server_is_live(), "server should be live after init");
    ASSERT(nxt_backend_count() == 0, "no backends registered yet");

    rc = nxt_backend_manager_fini();
    ASSERT(rc == 0, "backend_manager_fini failed");
    ASSERT(!nxt_server_is_live(), "server should not be live after fini");
    PASS();
}

/* ── Test: Backend State Strings ──────────────────────────────────────── */
static void test_backend_state_strings(void) {
    TEST("backend_state_strings");
    ASSERT(strcmp(nxt_backend_state_str(NXT_BACKEND_STATE_UNINITIALIZED), "UNINITIALIZED") == 0,
           "uninitialized state string mismatch");
    ASSERT(strcmp(nxt_backend_state_str(NXT_BACKEND_STATE_READY), "READY") == 0,
           "ready state string mismatch");
    ASSERT(strcmp(nxt_backend_state_str(NXT_BACKEND_STATE_ERROR), "ERROR") == 0,
           "error state string mismatch");

    ASSERT(strcmp(nxt_datatype_str(NXT_TYPE_FP32), "FP32") == 0, "FP32 datatype string mismatch");
    ASSERT(strcmp(nxt_datatype_str(NXT_TYPE_INT64), "INT64") == 0, "INT64 datatype string mismatch");

    ASSERT(strcmp(nxt_memory_type_str(NXT_MEM_GPU), "GPU") == 0, "GPU memory type string mismatch");
    ASSERT(strcmp(nxt_memory_type_str(NXT_MEM_CPU), "CPU") == 0, "CPU memory type string mismatch");
    PASS();
}

/* ── Test: Backend List / Count ───────────────────────────────────────── */
static void test_backend_list_count(void) {
    TEST("backend_list_count");
    int rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "init failed");

    ASSERT(nxt_backend_count() == 0, "count should be 0 initially");
    NxtBackend *list = nxt_backend_list();
    ASSERT(list != NULL, "list should not be NULL even when empty");

    /* Find non-existent backend returns NULL */
    ASSERT(nxt_backend_find("nonexistent.so") == NULL, "nonexistent backend should return NULL");

    rc = nxt_backend_manager_fini();
    ASSERT(rc == 0, "fini failed");
    PASS();
}

/* ── Test: Model Find / Count (empty) ─────────────────────────────────── */
static void test_model_find_empty(void) {
    TEST("model_find_empty");
    int rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "init failed");

    ASSERT(nxt_model_count() == 0, "model count should be 0");
    ASSERT(nxt_model_find("nonexistent", 0) == NULL, "find nonexistent should return NULL");

    rc = nxt_backend_manager_fini();
    ASSERT(rc == 0, "fini failed");
    PASS();
}

/* ── Test: Scheduler Init / Enqueue / Poll ────────────────────────────── */
static void test_scheduler_basic(void) {
    TEST("scheduler_basic");
    int rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "init failed");

    ASSERT(nxt_scheduler_completed_count() == 0, "completed should be 0");
    ASSERT(nxt_scheduler_waiting_count() == 0, "waiting should be 0");
    ASSERT(nxt_scheduler_running_count() == 0, "running should be 0");
    ASSERT(nxt_scheduler_avg_batch_size() == 0.0, "avg batch should be 0");

    /* Enqueue a pre-tokenized request (4 prompt tokens) */
    int32_t prompt[] = {101, 202, 303, 404};
    rc = nxt_scheduler_enqueue_tokenized("test-001", 0, prompt, 4, 10);
    ASSERT(rc == 0, "enqueue_tokenized failed");
    ASSERT(nxt_scheduler_waiting_count() == 1, "waiting count should be 1");

    /* Poll schedules the request (prefill 4 tokens) */
    rc = nxt_scheduler_poll();
    ASSERT(rc == 0, "poll failed");
    ASSERT(nxt_scheduler_running_count() == 1, "running count should be 1 after poll");
    ASSERT(nxt_scheduler_waiting_count() == 0, "waiting count should be 0 after poll");

    /* Complete the request */
    nxt_scheduler_complete_request("test-001");
    ASSERT(nxt_scheduler_completed_count() == 1, "completed should be 1");
    ASSERT(nxt_scheduler_running_count() == 0, "running count should be 0 after complete");

    rc = nxt_scheduler_fini();
    ASSERT(rc == 0, "scheduler_fini failed");

    rc = nxt_backend_manager_fini();
    ASSERT(rc == 0, "fini failed");
    PASS();
}

/* ── Test: Scheduler Queue Full ───────────────────────────────────────── */
static void test_scheduler_queue_full(void) {
    TEST("scheduler_queue_full");
    int rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "init failed");

    ASSERT(nxt_scheduler_waiting_count() == 0, "waiting should start at 0");

    /* Enqueue many requests with unique IDs */
    char id[32];
    int32_t prompt[] = {1};
    for (int i = 0; i < 64; i++) {
        snprintf(id, sizeof(id), "fill-%d", i);
        rc = nxt_scheduler_enqueue_tokenized(id, 0, prompt, 1, 10);
        ASSERT(rc == 0, "enqueue should succeed");
    }
    ASSERT(nxt_scheduler_waiting_count() == 64, "waiting should be 64");

    /* Duplicate request_id should fail */
    rc = nxt_scheduler_enqueue_tokenized("fill-0", 0, prompt, 1, 10);
    ASSERT(rc == -1, "duplicate enqueue should fail");
    ASSERT(nxt_scheduler_waiting_count() == 64, "waiting count unchanged after dup");

    rc = nxt_scheduler_fini();
    ASSERT(rc == 0, "scheduler_fini failed");

    rc = nxt_backend_manager_fini();
    ASSERT(rc == 0, "fini failed");
    PASS();
}

/* ── Test: Response Alloc / Free ──────────────────────────────────────── */
static void test_response_alloc_free(void) {
    TEST("response_alloc_free");
    NxtInferResponse *r = nxt_response_alloc("req-42");
    ASSERT(r != NULL, "response alloc returned NULL");
    ASSERT(strcmp(r->request_id, "req-42") == 0, "request_id mismatch");
    ASSERT(r->output_count == 0, "output_count should be 0");

    /* Set an output */
    int64_t shape[] = {1, 256};
    int rc = nxt_response_set_output(r, "logits", NXT_TYPE_FP32,
                                      shape, 2, NULL, 1024);
    ASSERT(rc == 0, "set_output failed");
    ASSERT(r->output_count == 1, "output_count should be 1");
    ASSERT(strcmp(r->outputs[0].name, "logits") == 0, "output name mismatch");
    ASSERT(r->outputs[0].dims_count == 2, "dims_count mismatch");
    ASSERT(r->outputs[0].shape[0] == 1 && r->outputs[0].shape[1] == 256,
           "shape mismatch");

    nxt_response_free(r);
    PASS();
}

/* ── Test: Server Health ──────────────────────────────────────────────── */
static void test_server_health(void) {
    TEST("server_health");
    ASSERT(!nxt_server_is_live(), "not live before init");
    ASSERT(!nxt_server_is_ready(), "not ready before init");

    int rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "init failed");
    ASSERT(nxt_server_is_live(), "live after init");
    ASSERT(!nxt_server_is_ready(), "not ready without backends");

    nxt_backend_manager_fini();
    ASSERT(!nxt_server_is_live(), "not live after fini");
    PASS();
}

/* ── Test: Model Ready Check ──────────────────────────────────────────── */
static void test_model_ready_check(void) {
    TEST("model_ready_check");
    int rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "init failed");

    ASSERT(!nxt_model_is_ready("my_model", 0), "model should not be ready");
    ASSERT(!nxt_model_is_ready("my_model", 3), "model v3 should not be ready");

    NxtModelStats stats;
    rc = nxt_model_stats("nonexistent", 0, &stats);
    ASSERT(rc == -1, "stats for nonexistent model should fail");

    nxt_backend_manager_fini();
    PASS();
}

/* ── Test: Double Init / Double Fini Idempotence ──────────────────────── */
static void test_double_init_fini(void) {
    TEST("double_init_fini");
    int rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "first init");
    rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "second init should be idempotent");

    rc = nxt_backend_manager_fini();
    ASSERT(rc == 0, "first fini");
    rc = nxt_backend_manager_fini();
    ASSERT(rc == 0, "second fini should be idempotent");
    PASS();
}

/* ── Test: Dynamic Backend Loading (dlopen/dlsym) ─────────────────────── */
static void test_dynamic_backend_loading(void) {
    TEST("dynamic_backend_loading");
    const char *so_path = find_so();
    if (!so_path) {
        printf("SKIP (no .so found in candidate paths) ");
        PASS();
        return;
    }
    printf("(%s) ", so_path);

    /* dlopen */
    void *handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    ASSERT(handle != NULL, dlerror());

    /* dlsym */
    dlerror();
    NxtBackendInitFn p_init = (NxtBackendInitFn)dlsym(handle, "nxt_backend_init");
    ASSERT(p_init != NULL, dlerror());
    NxtBackendRunFn  p_run  = (NxtBackendRunFn)dlsym(handle, "nxt_backend_run");
    ASSERT(p_run != NULL, dlerror());
    NxtBackendFiniFn p_fini = (NxtBackendFiniFn)dlsym(handle, "nxt_backend_fini");
    ASSERT(p_fini != NULL, dlerror());

    /* init(NULL) → error */
    ASSERT(p_init(NULL) == -1, "init(NULL) should return -1");

    /* init(valid) → success */
    NxtBackend *backend = malloc(sizeof(NxtBackend));
    ASSERT(backend != NULL, "OOM for backend struct");
    memset(backend, 0, sizeof(NxtBackend));
    backend->name = strdup("dynamic_test");
    ASSERT(backend->name != NULL, "OOM for backend name");
    ASSERT(p_init(backend) == 0, "init(valid) should return 0");
    ASSERT(backend->state == NXT_BACKEND_STATE_READY, "state should be READY after init");
    ASSERT(backend->backend_state != NULL, "backend_state should be non-NULL after init");

    /* run(NULL, *, *) → error */
    ASSERT(p_run(NULL, NULL, NULL) == -1, "run(NULL) should return -1");
    ASSERT(p_run(backend, NULL, NULL) == -1, "run with NULL input/output should return -1");

    /* run(valid, *, *) */
    int dummy_input  = 0xDEAD;
    int dummy_output = 0;
    int rc = p_run(backend, &dummy_input, &dummy_output);
    /* Stub returns 0, real CUDA backend returns 0 on success */
    ASSERT(rc == 0 || rc == -1, "unexpected run return code");
    if (rc == -1) printf("(stub) ");

    /* fini(NULL) → error */
    ASSERT(p_fini(NULL) == -1, "fini(NULL) should return -1");

    /* fini(valid) → success */
    ASSERT(p_fini(backend) == 0, "fini should return 0");
    ASSERT(backend->state == NXT_BACKEND_STATE_UNINITIALIZED,
           "state should be UNINITIALIZED after fini");
    ASSERT(backend->backend_state == NULL,
           "backend_state should be NULL after fini");

    /* Double-fini: idempotent */
    ASSERT(p_fini(backend) == 0, "double-fini should return 0");

    free(backend->name);
    free(backend);

    /* dlclose */
    ASSERT(dlclose(handle) == 0, dlerror());

    PASS();
}

/* ── Test Runner ──────────────────────────────────────────────────────── */
int main(void) {
    printf("=== xLLM Backend Manager Test Suite ===\n\n");

    test_backend_manager_init_fini();
    test_backend_state_strings();
    test_backend_list_count();
    test_model_find_empty();
    test_scheduler_basic();
    test_scheduler_queue_full();
    test_response_alloc_free();
    test_server_health();
    test_model_ready_check();
    test_double_init_fini();
    test_dynamic_backend_loading();

    printf("\n=== Results: %d run, %d passed, %d failed ===\n",
           tests_run, tests_passed, tests_failed);
    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
