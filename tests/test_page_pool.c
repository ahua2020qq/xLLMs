/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "page_pool.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { tests_run++; printf("  RUN  %s ... ", name); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg)  do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_PTR(p, msg) do { if ((p) == NULL) { FAIL(msg); return; } } while(0)

// ── Test: Pool Initialization ───────────────────────────────────────
static void test_pool_init(void) {
    TEST("pool_init");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  1ULL * 1024 * 1024 * 1024,  // 1 GiB GPU
                  4ULL * 1024 * 1024 * 1024,  // 4 GiB CPU
                  16ULL * 1024 * 1024 * 1024, // 16 GiB SSD
                  64 * 1024,                   // 64 KB pages
                  100000);

    ASSERT(pool.tier_capacity[TIER_GPU] > 0, "GPU capacity zero");
    ASSERT(pool.tier_capacity[TIER_CPU] > 0, "CPU capacity zero");
    ASSERT(pool.tier_capacity[TIER_SSD] > 0, "SSD capacity zero");
    ASSERT(pool.lru_k.capacity > 0, "LRU-K capacity zero");
    ASSERT(pool.free_counts[TIER_GPU][PAGE_TYPE_DATA] > 0, "No free GPU DATA pages");

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Page Alloc / Free Lifecycle ────────────────────────────────
static void test_page_alloc_free(void) {
    TEST("page_alloc_free");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  256ULL * 1024 * 1024,  // 256 MiB GPU
                  512ULL * 1024 * 1024,  // 512 MiB CPU
                  1024ULL * 1024 * 1024, // 1 GiB SSD
                  64 * 1024,              // 64 KB pages
                  10000);

    uint64_t allocs_before = pool.total_allocations;

    NxtPage *p1 = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
    ASSERT_PTR(p1, "first alloc returned NULL");
    ASSERT(p1->ref_count == 1, "ref_count != 1 after alloc");
    ASSERT(p1->type == PAGE_TYPE_DATA, "wrong page type");
    ASSERT(pool.total_allocations == allocs_before + 1, "total_allocations not incremented");

    NxtPage *p2 = nxt_page_alloc(&pool, PAGE_TYPE_INDEX, TIER_CPU);
    ASSERT_PTR(p2, "second alloc returned NULL");
    ASSERT(p2->type == PAGE_TYPE_INDEX, "wrong page type for p2");

    NxtPage *p3 = nxt_page_alloc(&pool, PAGE_TYPE_CONTROL, TIER_SSD);
    ASSERT_PTR(p3, "third alloc returned NULL");
    ASSERT(p3->type == PAGE_TYPE_CONTROL, "wrong page type for p3");

    // Free and verify ref_count decrement triggers free
    nxt_page_ref_dec(&pool, p1);
    nxt_page_ref_dec(&pool, p2);
    nxt_page_ref_dec(&pool, p3);

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Reference Counting ─────────────────────────────────────────
static void test_ref_counting(void) {
    TEST("ref_counting");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  128ULL * 1024 * 1024,
                  256ULL * 1024 * 1024,
                  512ULL * 1024 * 1024,
                  32 * 1024,
                  5000);

    NxtPage *page = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
    ASSERT_PTR(page, "alloc returned NULL");

    // Multiple ref increments
    nxt_page_ref_inc(page);
    nxt_page_ref_inc(page);
    ASSERT(page->ref_count == 3, "ref_count != 3 after 2 incs");

    // Decrement to 2
    nxt_page_ref_dec(&pool, page);
    ASSERT(page->ref_count == 2, "ref_count != 2 after 1 dec");

    // Decrement to 1
    nxt_page_ref_dec(&pool, page);
    ASSERT(page->ref_count == 1, "ref_count != 1 after 2 dec");

    // Final dec to 0 (should trigger free)
    // We verify by ensuring no crash occurs
    nxt_page_ref_dec(&pool, page);

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Admission Control ──────────────────────────────────────────
static void test_admission_control(void) {
    TEST("admission_control");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  10ULL * 1024 * 1024,   // 10 MiB GPU
                  10ULL * 1024 * 1024,   // 10 MiB CPU
                  10ULL * 1024 * 1024,   // 10 MiB SSD
                  4096,
                  500);

    // Request within budget
    NxtReqPageDir small_req = {
        .request_id = 1,
        .estimated_memory = 1024,
        .priority = 0,
        .page_count = 1
    };
    ASSERT(nxt_admission_check(&pool, &small_req) == true, "small request rejected");

    // Request exceeding total capacity
    NxtReqPageDir huge_req = {
        .request_id = 2,
        .estimated_memory = 100ULL * 1024 * 1024 * 1024,  // 100 GiB
        .priority = 0,
        .page_count = 1
    };
    ASSERT(nxt_admission_check(&pool, &huge_req) == false, "huge request should be rejected");

    // NULL safety
    ASSERT(nxt_admission_check(&pool, NULL) == false, "NULL request not rejected");

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: LRU-K Insert and Access ────────────────────────────────────
static void test_lruk_basic(void) {
    TEST("lruk_basic");
    LruKCache cache;
    lruk_init(&cache, 128);

    uint64_t now = 1000;

    // Insert and access
    lruk_insert(&cache, 0, now);
    lruk_access(&cache, 0, now + 100);
    lruk_access(&cache, 0, now + 200);

    ASSERT(lruk_is_mature(&cache.entries[0]), "entry should be mature after 3 accesses");

    uint64_t kth = lruk_kth_timestamp(&cache.entries[0]);
    ASSERT(kth == now, "k-th timestamp should be first access time");

    // Victim selection
    lruk_insert(&cache, 1, now + 50);
    lruk_access(&cache, 1, now + 150);
    lruk_access(&cache, 1, now + 250);

    // Entry 0 has oldest k-th timestamp (1000 vs 1050)
    uint32_t victim = UINT32_MAX;
    bool found = lruk_select_victim(&cache, &victim);
    ASSERT(found, "no victim found");
    ASSERT(victim == 0, "should evict entry 0 (older)");

    lruk_destroy(&cache);
    PASS();
}

// ── Test: LRU-K Remove ───────────────────────────────────────────────
static void test_lruk_remove(void) {
    TEST("lruk_remove");
    LruKCache cache;
    lruk_init(&cache, 64);

    lruk_insert(&cache, 5, 1000);
    lruk_access(&cache, 5, 2000);
    lruk_access(&cache, 5, 3000);
    ASSERT(lruk_is_mature(&cache.entries[5]), "entry should be mature");

    lruk_remove(&cache, 5);
    ASSERT(!lruk_is_mature(&cache.entries[5]), "entry should not be mature after remove");
    ASSERT(lruk_kth_timestamp(&cache.entries[5]) == LRU_TIMESTAMP_INVALID, "kth_ts should be invalid");

    lruk_destroy(&cache);
    PASS();
}

// ── Test Runner ──────────────────────────────────────────────────────
int main(void) {
    printf("=== nxtLLM Test Suite ===\n\n");

    test_pool_init();
    test_page_alloc_free();
    test_ref_counting();
    test_admission_control();
    test_lruk_basic();
    test_lruk_remove();

    printf("\n=== Results: %d run, %d passed, %d failed ===\n",
           tests_run, tests_passed, tests_failed);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
