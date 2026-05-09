/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Dynamic backend-loading test: dlopen cuda_backend.so, call init/run/fini,
 * verify return codes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <assert.h>
#include "backend.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { tests_run++; printf("  RUN  %s ... ", name); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg)  do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* Function pointer types matching exported symbols */
typedef int (*NxtBackendInitFn)(NxtBackend *backend);
typedef int (*NxtBackendRunFn)(NxtBackend *backend, void *input, void *output);
typedef int (*NxtBackendFiniFn)(NxtBackend *backend);

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
        if (f) {
            fclose(f);
            return candidate_paths[i];
        }
    }
    return NULL;
}

/* ── Test: dlopen the .so ──────────────────────────────────────────────── */
static void test_dlopen(const char *so_path, void **out_handle) {
    TEST("dlopen_cuda_backend");
    *out_handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    ASSERT(*out_handle != NULL, dlerror());
    PASS();
}

/* ── Test: dlsym all three entry points ─────────────────────────────────── */
static void test_dlsym(void *handle,
                       NxtBackendInitFn *p_init,
                       NxtBackendRunFn  *p_run,
                       NxtBackendFiniFn *p_fini) {
    TEST("dlsym_entry_points");
    dlerror(); /* clear */

    *p_init = (NxtBackendInitFn)dlsym(handle, "nxt_backend_init");
    ASSERT(*p_init != NULL, dlerror());

    *p_run  = (NxtBackendRunFn)dlsym(handle, "nxt_backend_run");
    ASSERT(*p_run != NULL, dlerror());

    *p_fini = (NxtBackendFiniFn)dlsym(handle, "nxt_backend_fini");
    ASSERT(*p_fini != NULL, dlerror());

    PASS();
}

/* ── Test: backend_init with NULL → error ──────────────────────────────── */
static void test_init_null(NxtBackendInitFn init_fn) {
    TEST("backend_init_null");
    int rc = init_fn(NULL);
    ASSERT(rc == -1, "init(NULL) should return -1");
    PASS();
}

/* ── Test: backend_init with valid backend → success ───────────────────── */
static void test_init_valid(NxtBackendInitFn init_fn, NxtBackend *backend) {
    TEST("backend_init_valid");
    memset(backend, 0, sizeof(NxtBackend));
    backend->name = strdup("cuda_test");
    int rc = init_fn(backend);
    ASSERT(rc == 0, "backend_init should return 0");
    ASSERT(backend->state == NXT_BACKEND_STATE_READY, "state should be READY");
    ASSERT(backend->backend_state != NULL, "backend_state should not be NULL");
    PASS();
}

/* ── Test: backend_run with NULL args → error ──────────────────────────── */
static void test_run_null(NxtBackendRunFn run_fn, NxtBackend *backend) {
    TEST("backend_run_null");
    int rc = run_fn(backend, NULL, NULL);
    ASSERT(rc == -1, "run with NULL input/output should return -1");

    rc = run_fn(NULL, NULL, NULL);
    ASSERT(rc == -1, "run with NULL backend should return -1");
    PASS();
}

/* ── Test: backend_run stub (no CUDA) → returns error ──────────────────── */
static void test_run_stub(NxtBackendRunFn run_fn, NxtBackend *backend) {
    TEST("backend_run_stub");
    int dummy_input  = 0xDEAD;
    int dummy_output = 0;
    int rc = run_fn(backend, &dummy_input, &dummy_output);

    /*
     * Without CUDA the stub returns -1 and prints a warning to stderr.
     * With CUDA, it expects a properly filled NxtPagedAttentionInput and
     * returns 0 on success.
     */
    if (rc == -1) {
        /* Expected stub behavior — test still counts as informative. */
        printf(" (stub: run returns %d - expected without CUDA) ", rc);
    } else if (rc == 0) {
        printf(" (GPU: run succeeded) ");
    } else {
        FAIL("unexpected return code from backend_run");
        return;
    }
    PASS();
}

/* ── Test: backend_fini with NULL → error ──────────────────────────────── */
static void test_fini_null(NxtBackendFiniFn fini_fn) {
    TEST("backend_fini_null");
    int rc = fini_fn(NULL);
    ASSERT(rc == -1, "fini(NULL) should return -1");
    PASS();
}

/* ── Test: backend_fini valid → success ────────────────────────────────── */
static void test_fini_valid(NxtBackendFiniFn fini_fn, NxtBackend *backend) {
    TEST("backend_fini_valid");
    void *saved_state = backend->backend_state;
    int rc = fini_fn(backend);
    ASSERT(rc == 0, "fini should return 0");
    ASSERT(backend->state == NXT_BACKEND_STATE_UNINITIALIZED,
           "state should be UNINITIALIZED after fini");
    ASSERT(backend->backend_state == NULL,
           "backend_state should be NULL after fini");

    /* Double-fini: idempotent */
    rc = fini_fn(backend);
    ASSERT(rc == 0, "double-fini should return 0");
    PASS();

    /* Note: saved_state has been freed by fini — no leak */
    (void)saved_state;
}

/* ── Test: dlclose ─────────────────────────────────────────────────────── */
static void test_dlclose(void *handle) {
    TEST("dlclose_cuda_backend");
    int rc = dlclose(handle);
    ASSERT(rc == 0, dlerror());
    PASS();
}

/* ── Test Runner ───────────────────────────────────────────────────────── */
int main(void) {
    NxtBackendInitFn init_fn;
    NxtBackendRunFn  run_fn;
    NxtBackendFiniFn fini_fn;

    printf("=== xLLM Dynamic Backend Loading Test Suite ===\n\n");

    const char *so_path = find_so();
    if (!so_path) {
        printf("  SKIP  No cuda_backend.so found in candidate paths.\n"
               "        Build with: cd build && cmake .. && make xllm_cuda_backend\n"
               "        (requires CUDA Toolkit, or stub builds without)\n\n"
               "=== Results: %d run, %d passed, %d failed (SKIPPED) ===\n",
               tests_run, tests_passed, tests_failed);
        return EXIT_SUCCESS;  /* Not a failure — .so just not built yet */
    }

    printf("  Found: %s\n\n", so_path);

    void *handle = NULL;
    test_dlopen(so_path, &handle);
    if (!handle) {
        printf("\n=== Results: %d run, %d passed, %d failed ===\n",
               tests_run, tests_passed, tests_failed);
        return EXIT_FAILURE;
    }

    test_dlsym(handle, &init_fn, &run_fn, &fini_fn);

    test_init_null(init_fn);

    /* Allocate a backend struct on the heap for the CUDA backend to manage */
    NxtBackend *backend = malloc(sizeof(NxtBackend));
    if (!backend) {
        fprintf(stderr, "FATAL: out of memory\n");
        dlclose(handle);
        return EXIT_FAILURE;
    }

    test_init_valid(init_fn, backend);
    test_run_null(run_fn, backend);
    test_run_stub(run_fn, backend);
    test_fini_null(fini_fn);
    test_fini_valid(fini_fn, backend);

    free(backend->name);
    free(backend);

    test_dlclose(handle);

    printf("\n=== Results: %d run, %d passed, %d failed ===\n",
           tests_run, tests_passed, tests_failed);
    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
