/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Continuous Batching Scheduler tests: enqueue, schedule, preemption, stats.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "backend.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { tests_run++; printf("  RUN  %s ... ", name); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg)  do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ── Test: Scheduler Init / Fini ────────────────────────────────────────── */

static void test_scheduler_init_fini(void) {
    TEST("scheduler_init_fini");
    NxtSchedulerConfig cfg = {
        .policy = NXT_SCHED_DYNAMIC,
        .max_preferred_batch_size = 32,
        .max_queue_delay_ms = 100.0,
        .preserve_ordering = true,
        .priority_levels = 1,
        .max_queue_size = 2048,
    };
    int rc = nxt_scheduler_init(&cfg);
    ASSERT(rc == 0, "scheduler_init failed");
    rc = nxt_scheduler_fini();
    ASSERT(rc == 0, "scheduler_fini failed");
    PASS();
}

/* ── Test: Double Init / Fini ───────────────────────────────────────────── */

static void test_scheduler_double_init(void) {
    TEST("scheduler_double_init");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    cfg.max_preferred_batch_size = 16;

    int rc = nxt_scheduler_init(&cfg);
    ASSERT(rc == 0, "first init failed");
    rc = nxt_scheduler_init(&cfg);
    ASSERT(rc == 0, "second init should succeed (resets state)");
    rc = nxt_scheduler_fini();
    ASSERT(rc == 0, "fini failed");
    PASS();
}

/* ── Test: Enqueue Single Request ───────────────────────────────────────── */

static void test_scheduler_enqueue(void) {
    TEST("scheduler_enqueue");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    cfg.max_preferred_batch_size = 16;
    nxt_scheduler_init(&cfg);

    NxtInferRequest req = {
        .request_id = "req-001",
        .batch_size = 1,
        .priority = 0,
    };
    int rc = nxt_scheduler_enqueue(&req);
    ASSERT(rc == 0, "enqueue failed");

    uint64_t queued = nxt_scheduler_queued_count();
    ASSERT(queued == 1, "queued count should be 1 after enqueue");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Enqueue Multiple Requests ────────────────────────────────────── */

static void test_scheduler_enqueue_multiple(void) {
    TEST("scheduler_enqueue_multiple");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    cfg.max_preferred_batch_size = 16;
    nxt_scheduler_init(&cfg);

    for (int i = 0; i < 10; i++) {
        NxtInferRequest req = {0};
        char id[32];
        snprintf(id, sizeof(id), "req-%03d", i);
        /* req.request_id allocated on stack, scheduler copies it */
        NxtInferRequest req_local = {
            .request_id = id,
            .batch_size = 1,
            .priority = i,
        };
        int rc = nxt_scheduler_enqueue(&req_local);
        ASSERT(rc == 0, "enqueue failed");
    }

    ASSERT(nxt_scheduler_queued_count() == 10, "should have 10 queued");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Enqueue Duplicate Request ────────────────────────────────────── */

static void test_scheduler_enqueue_duplicate(void) {
    TEST("scheduler_enqueue_duplicate");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    nxt_scheduler_init(&cfg);

    NxtInferRequest req = { .request_id = "dup", .batch_size = 1, .priority = 0 };
    int rc = nxt_scheduler_enqueue(&req);
    ASSERT(rc == 0, "first enqueue failed");

    rc = nxt_scheduler_enqueue(&req);
    ASSERT(rc == -1, "duplicate enqueue should fail");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Enqueue NULL Request ─────────────────────────────────────────── */

static void test_scheduler_enqueue_null(void) {
    TEST("scheduler_enqueue_null");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    nxt_scheduler_init(&cfg);

    int rc = nxt_scheduler_enqueue(NULL);
    ASSERT(rc == -1, "NULL enqueue should fail");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Enqueue Tokenized Request ────────────────────────────────────── */

static void test_scheduler_enqueue_tokenized(void) {
    TEST("scheduler_enqueue_tokenized");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    cfg.max_preferred_batch_size = 16;
    nxt_scheduler_init(&cfg);

    int32_t tokens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int rc = nxt_scheduler_enqueue_tokenized("tok-req", 0, tokens, 10, 50);
    ASSERT(rc == 0, "tokenized enqueue failed");

    ASSERT(nxt_scheduler_waiting_count() == 1, "should have 1 waiting");
    ASSERT(nxt_scheduler_running_count() == 0, "should have 0 running");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Schedule Step with Tokenized Requests ────────────────────────── */

static void test_scheduler_schedule_step(void) {
    TEST("scheduler_schedule_step");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    cfg.max_preferred_batch_size = 16;
    cfg.max_queue_size = 128;  /* token_budget */
    nxt_scheduler_init(&cfg);

    /* Enqueue tokenized requests */
    int32_t tokens_small[] = {1, 2, 3};
    nxt_scheduler_enqueue_tokenized("small-req", 0, tokens_small, 3, 20);

    int32_t tokens_large[100];
    for (int i = 0; i < 100; i++) tokens_large[i] = i;
    nxt_scheduler_enqueue_tokenized("large-req", 1, tokens_large, 100, 50);

    /* Schedule step */
    uint32_t count = 0;
    SchedRequest **scheduled = NULL;
    uint32_t *token_budgets = calloc(64, sizeof(uint32_t));

    int rc = nxt_scheduler_schedule_step(&scheduled, &count, token_budgets);
    ASSERT(rc == 0, "schedule_step failed");
    ASSERT(count > 0, "should have scheduled at least one request");

    /* Clean up */
    free(scheduled);
    free(token_budgets);
    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Complete Request ─────────────────────────────────────────────── */

static void test_scheduler_complete_request(void) {
    TEST("scheduler_complete_request");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    cfg.max_preferred_batch_size = 16;
    nxt_scheduler_init(&cfg);

    int32_t tokens[] = {1, 2, 3};
    nxt_scheduler_enqueue_tokenized("complete-me", 0, tokens, 3, 10);

    uint32_t count = 0;
    SchedRequest **scheduled = NULL;
    uint32_t *token_budgets = calloc(64, sizeof(uint32_t));
    nxt_scheduler_schedule_step(&scheduled, &count, token_budgets);
    free(scheduled);
    free(token_budgets);

    nxt_scheduler_complete_request("complete-me");
    ASSERT(nxt_scheduler_completed_count() >= 1, "completed count should be >= 1");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Scheduler Stats ──────────────────────────────────────────────── */

static void test_scheduler_stats(void) {
    TEST("scheduler_stats");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    cfg.max_preferred_batch_size = 32;
    nxt_scheduler_init(&cfg);

    /* Initially all zero */
    ASSERT(nxt_scheduler_completed_count() == 0, "initial completed should be 0");
    ASSERT(nxt_scheduler_queued_count() == 0, "initial queued should be 0");
    ASSERT(nxt_scheduler_running_count() == 0, "initial running should be 0");
    ASSERT(nxt_scheduler_waiting_count() == 0, "initial waiting should be 0");
    ASSERT(nxt_scheduler_preempted_count() == 0, "initial preempted should be 0");

    double avg_batch = nxt_scheduler_avg_batch_size();
    ASSERT(avg_batch == 0.0, "avg batch should be 0 with no steps");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Scheduler Policy Configuration ───────────────────────────────── */

static void test_scheduler_policy_config(void) {
    TEST("scheduler_policy_config");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    nxt_scheduler_init(&cfg);

    ASSERT(nxt_scheduler_get_policy() == NXT_SCHED_DYNAMIC, "default policy should be DYNAMIC");

    nxt_scheduler_set_policy(NXT_SCHED_SEQUENCE);
    ASSERT(nxt_scheduler_get_policy() == NXT_SCHED_SEQUENCE, "policy should be SEQUENCE");

    nxt_scheduler_set_policy(NXT_SCHED_ENSEMBLE);
    ASSERT(nxt_scheduler_get_policy() == NXT_SCHED_ENSEMBLE, "policy should be ENSEMBLE");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Token Budget Configuration ───────────────────────────────────── */

static void test_scheduler_token_budget(void) {
    TEST("scheduler_token_budget");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    nxt_scheduler_init(&cfg);

    nxt_scheduler_set_token_budget(4096);
    ASSERT(nxt_scheduler_get_token_budget() == 4096, "token budget should be 4096");

    nxt_scheduler_set_token_budget(512);
    ASSERT(nxt_scheduler_get_token_budget() == 512, "token budget should be 512");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Poll with Empty Queue ────────────────────────────────────────── */

static void test_scheduler_poll_empty(void) {
    TEST("scheduler_poll_empty");
    NxtSchedulerConfig cfg = {0};
    cfg.policy = NXT_SCHED_DYNAMIC;
    nxt_scheduler_init(&cfg);

    int rc = nxt_scheduler_poll();
    ASSERT(rc == 0 || rc == 1, "poll on empty queue should return 0 or 1");

    nxt_scheduler_fini();
    PASS();
}

/* ── Test: Scheduler Without Init ───────────────────────────────────────── */

static void test_scheduler_without_init(void) {
    TEST("scheduler_without_init");
    /* Ensure scheduler is not initialized */
    nxt_scheduler_fini();  /* safe to call even if not init'd */

    int rc = nxt_scheduler_enqueue(NULL);
    ASSERT(rc == -1, "enqueue without init should fail");

    uint64_t c = nxt_scheduler_completed_count();
    ASSERT(c == 0, "completed count without init should be 0");

    PASS();
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Scheduler Tests (Continuous Batching) ===\n\n");

    test_scheduler_init_fini();
    test_scheduler_double_init();
    test_scheduler_enqueue();
    test_scheduler_enqueue_multiple();
    test_scheduler_enqueue_duplicate();
    test_scheduler_enqueue_null();
    test_scheduler_enqueue_tokenized();
    test_scheduler_schedule_step();
    test_scheduler_complete_request();
    test_scheduler_stats();
    test_scheduler_policy_config();
    test_scheduler_token_budget();
    test_scheduler_poll_empty();
    test_scheduler_without_init();

    printf("\n=== Results: %d run, %d passed, %d failed ===\n",
           tests_run, tests_passed, tests_failed);
    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
