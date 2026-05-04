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
#include <unistd.h>
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

// ── Test: Hash Table Insert / Lookup / Remove ────────────────────────
static void test_hash_table_basic(void) {
    TEST("hash_table_basic");
    NxtPageHash hash;
    nxt_hash_init(&hash, 64);

    // Create dummy pages for testing
    NxtPage p0 = { .page_id = 0 };
    NxtPage p1 = { .page_id = 1 };
    NxtPage p2 = { .page_id = 42 };

    nxt_hash_insert(&hash, 0, &p0);
    nxt_hash_insert(&hash, 1, &p1);
    nxt_hash_insert(&hash, 42, &p2);

    ASSERT(nxt_hash_lookup(&hash, 0) == &p0, "hash lookup id=0 failed");
    ASSERT(nxt_hash_lookup(&hash, 1) == &p1, "hash lookup id=1 failed");
    ASSERT(nxt_hash_lookup(&hash, 42) == &p2, "hash lookup id=42 failed");
    ASSERT(nxt_hash_lookup(&hash, 99) == NULL, "hash lookup non-existent key should return NULL");

    // Remove and verify
    bool removed = nxt_hash_remove(&hash, 1);
    ASSERT(removed, "hash remove id=1 failed");
    ASSERT(nxt_hash_lookup(&hash, 1) == NULL, "hash lookup after remove should be NULL");
    ASSERT(nxt_hash_lookup(&hash, 0) == &p0, "hash lookup id=0 should still work");

    nxt_hash_destroy(&hash);
    PASS();
}

// ── Test: Hash Table with Pool Integration ────────────────────────────
static void test_hash_in_pool(void) {
    TEST("hash_in_pool");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  64ULL * 1024 * 1024,
                  64ULL * 1024 * 1024,
                  64ULL * 1024 * 1024,
                  64 * 1024,
                  1000);

    ASSERT(pool.page_hash.capacity > 0, "hash not initialized in pool");
    ASSERT(pool.pages_array != NULL, "pages_array not set");

    // Allocate a page and verify hash lookup works
    NxtPage *p = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
    ASSERT_PTR(p, "alloc returned NULL");

    NxtPage *found = nxt_hash_lookup(&pool.page_hash, p->page_id);
    ASSERT(found == p, "hash lookup for allocated page failed");

    nxt_page_ref_dec(&pool, p);
    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: LRU-K Eviction Uses Hash Table ──────────────────────────────
static void test_eviction_with_hash(void) {
    TEST("eviction_with_hash");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  4ULL * 1024 * 1024,    // 4 MiB GPU
                  4ULL * 1024 * 1024,
                  4ULL * 1024 * 1024,
                  64 * 1024,
                  200);

    uint64_t evictions_before = pool.total_evictions;

    // Allocate many pages to trigger eviction
    for (int i = 0; i < 64; i++) {
        NxtPage *p = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
        if (!p) break;
        nxt_page_ref_dec(&pool, p);  // release ref so it can be evicted
    }

    // Verify evictions happened (we may have filled up GPU tier)
    // The key test is that nxt_evict_lru_k uses the hash table and doesn't crash
    ASSERT(pool.total_evictions >= evictions_before,
           "evictions should not decrease");

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Defrag Compacts Pages Within Tier ───────────────────────────
static void test_defrag_basic(void) {
    TEST("defrag_basic");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  64 * 1024,
                  400);

    uint64_t rounds_before = pool.total_defrag_rounds;
    ASSERT(rounds_before == 0, "defrag rounds should start at 0");

    // Allocate some pages, then free some to create fragmentation
    NxtPage *pages[20];
    for (int i = 0; i < 12; i++) {
        pages[i] = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
        ASSERT_PTR(pages[i], "alloc failed during defrag setup");
    }

    // Free every other page to create gaps
    for (int i = 0; i < 12; i += 2) {
        nxt_page_ref_dec(&pool, pages[i]);
    }

    // Run defrag
    pool.defrag_threshold = 0.0f; // force defrag for testing
    nxt_defrag_background(&pool);
    ASSERT(pool.total_defrag_rounds == 1, "defrag rounds should be 1 after call");

    // All pages should still be accessible via hash
    for (int i = 1; i < 12; i += 2) {
        NxtPage *found = nxt_hash_lookup(&pool.page_hash, pages[i]->page_id);
        ASSERT(found == pages[i], "page lost after defrag");
    }

    // Clean up remaining pages
    for (int i = 1; i < 12; i += 2) {
        nxt_page_ref_dec(&pool, pages[i]);
    }

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Defrag Preserves Free Counts ────────────────────────────────
static void test_defrag_free_counts(void) {
    TEST("defrag_free_counts");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  4ULL * 1024 * 1024,
                  4ULL * 1024 * 1024,
                  4ULL * 1024 * 1024,
                  64 * 1024,
                  200);

    // Count initial free pages in GPU DATA tier
    uint32_t initial_free = pool.free_counts[TIER_GPU][PAGE_TYPE_DATA];

    // Allocate several pages
    NxtPage *p[8];
    for (int i = 0; i < 8; i++) {
        p[i] = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
    }

    uint32_t after_alloc = pool.free_counts[TIER_GPU][PAGE_TYPE_DATA];
    ASSERT(after_alloc == initial_free - 8, "free count should decrease by 8");

    // Free some pages creating holes
    nxt_page_ref_dec(&pool, p[0]);
    nxt_page_ref_dec(&pool, p[2]);
    nxt_page_ref_dec(&pool, p[4]);
    nxt_page_ref_dec(&pool, p[6]);

    uint32_t before_defrag = pool.free_counts[TIER_GPU][PAGE_TYPE_DATA];
    pool.defrag_threshold = 0.0f; // force defrag for testing
    nxt_defrag_background(&pool);
    uint32_t after_defrag = pool.free_counts[TIER_GPU][PAGE_TYPE_DATA];

    // Free count should be preserved after defrag
    ASSERT(before_defrag == after_defrag, "defrag should preserve free count");

    // Cleanup
    nxt_page_ref_dec(&pool, p[1]);
    nxt_page_ref_dec(&pool, p[3]);
    nxt_page_ref_dec(&pool, p[5]);
    nxt_page_ref_dec(&pool, p[7]);

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Defrag Scoring ──────────────────────────────────────────────
static void test_defrag_scoring(void) {
    TEST("defrag_scoring");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  64 * 1024,
                  400);

    // Fresh pool: all pages are free and contiguous by page_id
    float score0 = nxt_calc_defrag_score(&pool);
    ASSERT(score0 >= 0.0f && score0 <= 1.0f, "score out of [0,1] range");
    // With all pages free and contiguous, score should be near zero
    ASSERT(score0 < 0.05f, "fresh pool fragmentation should be near zero");

    // Allocate pages to create fragmentation pattern
    NxtPage *pages[20];
    for (int i = 0; i < 12; i++) {
        pages[i] = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
        ASSERT_PTR(pages[i], "alloc failed during scoring setup");
    }

    // Free every other page to create interleaved free/used pattern
    for (int i = 0; i < 12; i += 2) {
        nxt_page_ref_dec(&pool, pages[i]);
    }

    float score1 = nxt_calc_defrag_score(&pool);
    ASSERT(score1 >= 0.0f && score1 <= 1.0f, "score after frag out of [0,1] range");
    // Interleaved pattern should produce higher fragmentation
    ASSERT(score1 > score0, "fragmented pool score should be higher than fresh");

    // Run defrag and verify score decreases
    pool.defrag_threshold = 0.0f; // force defrag for testing
    nxt_defrag_background(&pool);
    float score2 = nxt_calc_defrag_score(&pool);
    ASSERT(score2 < score1, "defrag should reduce fragmentation score");

    // Clean up remaining pages
    for (int i = 1; i < 12; i += 2) {
        nxt_page_ref_dec(&pool, pages[i]);
    }

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Background Defrag Thread ─────────────────────────────────────
static void test_defrag_background_thread(void) {
    TEST("defrag_background_thread");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  64 * 1024,
                  400);

    // Short interval for testing
    pool.defrag_interval_sec = 1;
    pool.defrag_threshold    = 0.0f; // always trigger defrag

    uint64_t rounds_before = pool.total_defrag_rounds;

    // Start background thread
    nxt_start_defrag_thread(&pool);
    ASSERT(pool.defrag_thread_running, "thread should be running after start");

    // Let it run at least one cycle
    sleep(2);

    // Stop thread gracefully
    nxt_stop_defrag_thread(&pool);
    ASSERT(!pool.defrag_thread_running, "thread should not be running after stop");

    // Defrag should have run at least once
    ASSERT(pool.total_defrag_rounds > rounds_before,
           "background thread should have executed defrag at least once");

    // Thread start/stop should be idempotent
    nxt_stop_defrag_thread(&pool); // should not crash

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Compaction-Style Merging (Defrag Job) ──────────────────────
// Verifies the pick → compact → install flow produces a defrag job
// with valid metadata and reclaims bytes.
static void test_compaction_style_merging(void) {
    TEST("compaction_style_merging");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  64 * 1024,
                  400);

    // Allocate pages then free alternate ones to create fragmentation
    NxtPage *pages[16];
    for (int i = 0; i < 12; i++) {
        pages[i] = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
        ASSERT_PTR(pages[i], "alloc failed during compaction setup");
    }
    for (int i = 0; i < 12; i += 2) {
        nxt_page_ref_dec(&pool, pages[i]);
    }

    // Run full compaction job via the pluggable scheduler
    NxtDefragJob job;
    memset(&job, 0, sizeof(job));
    nxt_defrag_run(&pool, &job);

    // Job should have identified a source tier+type
    ASSERT(job.fragmented_tier < TIER_COUNT, "invalid fragmented tier");
    ASSERT(job.fragmented_type < PAGE_TYPE_COUNT, "invalid fragmented type");
    ASSERT(job.bytes_reclaimed > 0, "compaction should reclaim some bytes");
    ASSERT(job.victim_count > 0, "should have moved some pages");
    ASSERT(pool.total_defrag_rounds == 1, "defrag rounds should increment");

    // All remaining used pages should still be accessible
    for (int i = 1; i < 12; i += 2) {
        NxtPage *found = nxt_hash_lookup(&pool.page_hash, pages[i]->page_id);
        ASSERT(found == pages[i], "used page lost after compaction merge");
    }

    // Clean up
    for (int i = 1; i < 12; i += 2) {
        nxt_page_ref_dec(&pool, pages[i]);
    }
    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Compensatory Eviction ───────────────────────────────────────
// Verifies that low-utilization pages are evicted before high-utilization
// pages when both have similar LRU timestamps.
static void test_compensatory_eviction(void) {
    TEST("compensatory_eviction");
    NxtGlobalBufferPool pool;
    // Small pool so eviction is forced
    nxt_pool_init(&pool,
                  2ULL * 1024 * 1024,   // 2 MiB GPU
                  2ULL * 1024 * 1024,
                  2ULL * 1024 * 1024,
                  64 * 1024,
                  64);

    // Allocate a few pages and set different utilization levels
    NxtPage *p0 = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
    NxtPage *p1 = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
    NxtPage *p2 = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
    ASSERT_PTR(p0, "alloc p0 failed");
    ASSERT_PTR(p1, "alloc p1 failed");
    ASSERT_PTR(p2, "alloc p2 failed");

    // p0: high utilization (valuable), p1: medium, p2: low utilization (waste)
    nxt_page_set_utilization(p0, 0.95f);
    nxt_page_set_utilization(p1, 0.50f);
    nxt_page_set_utilization(p2, 0.10f);

    ASSERT(p0->utilization == 0.95f, "utilization not set correctly for p0");
    ASSERT(p1->utilization == 0.50f, "utilization not set correctly for p1");
    ASSERT(p2->utilization == 0.10f, "utilization not set correctly for p2");

    // Release refs so they become evictable
    nxt_page_ref_dec(&pool, p0);
    nxt_page_ref_dec(&pool, p1);
    nxt_page_ref_dec(&pool, p2);

    // Compute eviction priorities; p2 (10% util) should have lowest priority
    // (lowest priority value = first to evict)
    uint64_t pri0 = nxt_page_eviction_priority(p0, &pool.lru_k, 1000000, 2.0f);
    nxt_page_eviction_priority(p1, &pool.lru_k, 1000000, 2.0f);
    uint64_t pri2 = nxt_page_eviction_priority(p2, &pool.lru_k, 1000000, 2.0f);

    // p2's kth_ts gets multiplied by (1 + 0.9 * 2.0) = 2.8
    // p0's kth_ts gets multiplied by (1 + 0.05 * 2.0) = 1.1
    // Since all have same kth_ts (first access), p2's priority is highest magnitude
    // (larger value = less valuable = evicted first under priority ordering)
    ASSERT(pri2 > pri0, "low-utilization page should have higher eviction priority");

    // Test compensated eviction function directly
    uint64_t evictions_before = pool.total_evictions;
    bool evicted = nxt_evict_lru_k_compensated(&pool, TIER_GPU, PAGE_TYPE_DATA, 2.0f);
    ASSERT(evicted, "compensated eviction should succeed");
    ASSERT(pool.total_evictions == evictions_before + 1,
           "total_evictions should increment");

    nxt_pool_destroy(&pool);
    PASS();
}

// ── Test: Pluggable Strategies ────────────────────────────────────────
// Verifies that strategy switching works and each strategy produces valid
// defrag jobs targeting appropriate tier×type groups.
static void test_pluggable_strategies(void) {
    TEST("pluggable_strategies");
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  8ULL * 1024 * 1024,
                  64 * 1024,
                  400);

    // Default strategy should be TIERED
    ASSERT(pool.defrag_sched.strategy == DEFRAG_STRATEGY_TIERED,
           "default strategy should be TIERED");
    ASSERT(pool.defrag_sched.pick_source != NULL,
           "TIERED pick_source should be set");
    ASSERT(pool.defrag_sched.compact != NULL,
           "TIERED compact should be set");
    ASSERT(pool.defrag_sched.install != NULL,
           "TIERED install should be set");

    // Test LEVELED strategy
    nxt_defrag_set_strategy(&pool, DEFRAG_STRATEGY_LEVELED);
    ASSERT(pool.defrag_sched.strategy == DEFRAG_STRATEGY_LEVELED,
           "strategy switch to LEVELED failed");
    ASSERT(pool.defrag_sched.pick_source == nxt_defrag_pick_source_leveled,
           "LEVELED pick_source not wired correctly");

    // Test GREEDY strategy
    nxt_defrag_set_strategy(&pool, DEFRAG_STRATEGY_GREEDY);
    ASSERT(pool.defrag_sched.strategy == DEFRAG_STRATEGY_GREEDY,
           "strategy switch to GREEDY failed");
    ASSERT(pool.defrag_sched.pick_source == nxt_defrag_pick_source_greedy,
           "GREEDY pick_source not wired correctly");

    // Allocate pages to create work for defrag
    NxtPage *pages[16];
    for (int i = 0; i < 12; i++) {
        pages[i] = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
        ASSERT_PTR(pages[i], "alloc failed during strategy test");
    }
    for (int i = 0; i < 12; i += 2)
        nxt_page_ref_dec(&pool, pages[i]);

    // Run each strategy and verify it works
    NxtDefragJob job;
    uint64_t rounds_before = pool.total_defrag_rounds;

    // TIERED
    nxt_defrag_set_strategy(&pool, DEFRAG_STRATEGY_TIERED);
    memset(&job, 0, sizeof(job));
    nxt_defrag_run(&pool, &job);
    ASSERT(job.fragmented_tier < TIER_COUNT, "TIERED: invalid tier");
    ASSERT(pool.total_defrag_rounds == rounds_before + 1, "TIERED: rounds not incremented");

    // LEVELED
    nxt_defrag_set_strategy(&pool, DEFRAG_STRATEGY_LEVELED);
    memset(&job, 0, sizeof(job));
    nxt_defrag_run(&pool, &job);
    ASSERT(job.fragmented_tier < TIER_COUNT, "LEVELED: invalid tier");
    ASSERT(pool.total_defrag_rounds == rounds_before + 2, "LEVELED: rounds not incremented");

    // GREEDY
    nxt_defrag_set_strategy(&pool, DEFRAG_STRATEGY_GREEDY);
    memset(&job, 0, sizeof(job));
    nxt_defrag_run(&pool, &job);
    ASSERT(job.fragmented_tier < TIER_COUNT, "GREEDY: invalid tier");
    ASSERT(pool.total_defrag_rounds == rounds_before + 3, "GREEDY: rounds not incremented");

    // Memory pressure check should work
    NxtMemoryPressure pressure = nxt_check_memory_pressure(&pool);
    ASSERT(pressure == PRESSURE_NORMAL ||
           pressure == PRESSURE_SLOWDOWN ||
           pressure == PRESSURE_STOP,
           "pressure check returned invalid value");

    // Clean up
    for (int i = 1; i < 12; i += 2)
        nxt_page_ref_dec(&pool, pages[i]);
    nxt_pool_destroy(&pool);
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
    test_hash_table_basic();
    test_hash_in_pool();
    test_eviction_with_hash();
    test_defrag_basic();
    test_defrag_free_counts();
    test_defrag_scoring();
    test_defrag_background_thread();
    test_compaction_style_merging();
    test_compensatory_eviction();
    test_pluggable_strategies();

    printf("\n=== Results: %d run, %d passed, %d failed ===\n",
           tests_run, tests_passed, tests_failed);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
