/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "prefix_sharing.h"
#include "page_pool.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { tests_run++; printf("  RUN  %s ... ", name); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg)  do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ── Test: Tree Init / Destroy ───────────────────────────────────────── */
static void test_tree_init_destroy(void) {
    TEST("tree_init_destroy");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    ASSERT(tree.root != NULL, "root is NULL");
    ASSERT(tree.root->lock_ref == 1, "root not locked");
    ASSERT(tree.root->token_count == 0, "root has tokens");
    ASSERT(tree.total_nodes == 1, "expected 1 node (root)");

    nxt_prefix_tree_destroy(&tree);
    ASSERT(tree.root == NULL, "tree not cleared");
    ASSERT(tree.total_nodes == 0, "node count not zero after destroy");

    PASS();
}

/* ── Test: Insert single sequence ────────────────────────────────────── */
static void test_insert_single(void) {
    TEST("insert_single");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    int32_t tokens[] = {1, 2, 3, 4, 5};
    uint32_t pages[] = {100, 101, 102, 103, 104};
    int32_t prefix_len = nxt_prefix_insert(&tree, tokens, 5, pages, 5, 0);
    ASSERT(prefix_len == 0, "expected no shared prefix on first insert");
    ASSERT(tree.total_nodes >= 2, "expected at least 2 nodes (root + inserted)");
    ASSERT(nxt_prefix_total_tokens(&tree) == 5, "expected 5 tokens cached");

    nxt_prefix_tree_destroy(&tree);
    PASS();
}

/* ── Test: Match prefix (full match) ─────────────────────────────────── */
static void test_match_full(void) {
    TEST("match_full");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    int32_t tokens[] = {10, 20, 30, 40, 50};
    uint32_t pages[] = {200, 201, 202, 203, 204};
    nxt_prefix_insert(&tree, tokens, 5, pages, 5, 0);

    NxtMatchResult result = nxt_prefix_match(&tree, tokens, 5);
    ASSERT(result.matched_tokens == 5, "expected 5 matched tokens");
    ASSERT(result.page_count == 5, "expected 5 pages");
    ASSERT(result.page_ids[0] == 200, "first page mismatch");
    ASSERT(result.page_ids[4] == 204, "last page mismatch");

    nxt_prefix_tree_destroy(&tree);
    PASS();
}

/* ── Test: Match partial prefix ──────────────────────────────────────── */
static void test_match_partial(void) {
    TEST("match_partial");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    int32_t tokens1[] = {1, 2, 3};
    uint32_t pages1[] = {10, 20, 30};
    nxt_prefix_insert(&tree, tokens1, 3, pages1, 3, 0);

    /* Query with extra tokens beyond the cached prefix */
    int32_t query[] = {1, 2, 3, 4, 5};
    NxtMatchResult result = nxt_prefix_match(&tree, query, 5);
    ASSERT(result.matched_tokens == 3, "expected 3 matched tokens");
    ASSERT(result.page_count == 3, "expected 3 pages from cached prefix");
    ASSERT(result.page_ids[0] == 10, "wrong first page");

    nxt_prefix_tree_destroy(&tree);
    PASS();
}

/* ── Test: Shared prefix across sequences ────────────────────────────── */
static void test_shared_prefix(void) {
    TEST("shared_prefix");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    /* Insert "Hello world" → tokens [1,2,3,4,5] */
    int32_t seq1[] = {1, 2, 3, 4, 5};
    uint32_t pages1[] = {100, 101, 102, 103, 104};
    int32_t pref1 = nxt_prefix_insert(&tree, seq1, 5, pages1, 5, 0);
    ASSERT(pref1 == 0, "first insert should have zero shared prefix");

    /* Insert "Hello AI" → tokens [1,2,3,6,7], shares prefix [1,2,3] */
    int32_t seq2[] = {1, 2, 3, 6, 7};
    uint32_t pages2[] = {300, 301, 302, 303, 304};
    int32_t pref2 = nxt_prefix_insert(&tree, seq2, 5, pages2, 5, 0);
    ASSERT(pref2 == 3, "expected 3 shared tokens");

    /* The shared prefix should be detected by a fresh match query */
    int32_t query[] = {1, 2, 3, 4, 5};
    NxtMatchResult result = nxt_prefix_match(&tree, query, 5);
    ASSERT(result.matched_tokens == 5, "should match full sequence 1");
    ASSERT(result.page_count == 5, "should return 5 pages");

    nxt_prefix_tree_destroy(&tree);
    PASS();
}

/* ── Test: Empty query ───────────────────────────────────────────────── */
static void test_empty_query(void) {
    TEST("empty_query");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    int32_t tokens[] = {1, 2, 3};
    uint32_t pages[] = {10, 20, 30};
    nxt_prefix_insert(&tree, tokens, 3, pages, 3, 0);

    NxtMatchResult result = nxt_prefix_match(&tree, NULL, 0);
    ASSERT(result.matched_tokens == 0, "empty query should match zero tokens");
    ASSERT(result.last_node == tree.root, "empty query returns root");

    nxt_prefix_tree_destroy(&tree);
    PASS();
}

/* ── Test: Lock / Unlock prevents eviction ───────────────────────────── */
static void test_lock_unlock(void) {
    TEST("lock_unlock");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    int32_t tokens[] = {7, 8, 9};
    uint32_t pages[] = {70, 80, 90};
    nxt_prefix_insert(&tree, tokens, 3, pages, 3, 0);

    /* Match to get the leaf node */
    NxtMatchResult result = nxt_prefix_match(&tree, tokens, 3);
    ASSERT(result.matched_tokens == 3, "should match 3 tokens");

    int32_t evictable_before = nxt_prefix_evictable_size(&tree);
    ASSERT(evictable_before > 0, "leaf should be evictable before lock");

    /* Lock the last node */
    nxt_prefix_lock(&tree, result.last_node);
    int32_t evictable_after_lock = nxt_prefix_evictable_size(&tree);
    ASSERT(evictable_after_lock < evictable_before, "evictable size should decrease after lock");

    /* Evict: the locked node should survive */
    int32_t evicted = nxt_prefix_evict(&tree, 100);
    int32_t total_after = nxt_prefix_total_tokens(&tree);
    ASSERT(total_after > 0, "locked node should not be evicted");

    /* Unlock and evict */
    nxt_prefix_unlock(&tree, result.last_node);
    evicted = nxt_prefix_evict(&tree, 100);
    int32_t total_final = nxt_prefix_total_tokens(&tree);
    ASSERT(total_final == 0, "all nodes should be evicted after unlock");

    nxt_prefix_tree_destroy(&tree);
    PASS();
}

/* ── Test: Eviction ──────────────────────────────────────────────────── */
static void test_eviction(void) {
    TEST("eviction");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    /* Insert several sequences */
    int32_t seq1[] = {1, 2, 3};
    uint32_t pages1[] = {10, 20, 30};
    nxt_prefix_insert(&tree, seq1, 3, pages1, 3, 0);

    int32_t seq2[] = {4, 5, 6, 7};
    uint32_t pages2[] = {40, 50, 60, 70};
    nxt_prefix_insert(&tree, seq2, 4, pages2, 4, 0);

    int32_t seq3[] = {8, 9};
    uint32_t pages3[] = {80, 90};
    nxt_prefix_insert(&tree, seq3, 2, pages3, 2, 0);

    int32_t total_before = nxt_prefix_total_tokens(&tree);
    ASSERT(total_before == 9, "expected 9 cached tokens");

    /* Evict 3 tokens */
    int32_t evicted = nxt_prefix_evict(&tree, 3);
    ASSERT(evicted >= 3, "should evict at least 3 tokens");

    int32_t total_after = nxt_prefix_total_tokens(&tree);
    ASSERT(total_after < total_before, "tokens should decrease after eviction");

    nxt_prefix_tree_destroy(&tree);
    PASS();
}

/* ── Test: Divergent branches ────────────────────────────────────────── */
static void test_divergent_branches(void) {
    TEST("divergent_branches");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    /* Branch 1: [1,2,3,4] */
    int32_t b1[] = {1, 2, 3, 4};
    uint32_t p1[] = {100, 101, 102, 103};
    nxt_prefix_insert(&tree, b1, 4, p1, 4, 0);

    /* Branch 2: [1,2,5,6] — shares [1,2] */
    int32_t b2[] = {1, 2, 5, 6};
    uint32_t p2[] = {200, 201, 202, 203};
    int32_t pref = nxt_prefix_insert(&tree, b2, 4, p2, 4, 0);
    ASSERT(pref == 2, "should share first 2 tokens");

    /* Branch 3: [1,2,3,7,8] — shares [1,2,3] with b1 */
    int32_t b3[] = {1, 2, 3, 7, 8};
    uint32_t p3[] = {300, 301, 302, 303, 304};
    pref = nxt_prefix_insert(&tree, b3, 5, p3, 5, 0);
    ASSERT(pref == 3, "should share first 3 tokens with b1");

    /* Verify each branch can be fully matched */
    NxtMatchResult r1 = nxt_prefix_match(&tree, b1, 4);
    ASSERT(r1.matched_tokens == 4, "b1 should fully match");

    NxtMatchResult r2 = nxt_prefix_match(&tree, b2, 4);
    ASSERT(r2.matched_tokens == 4, "b2 should fully match");

    NxtMatchResult r3 = nxt_prefix_match(&tree, b3, 5);
    ASSERT(r3.matched_tokens == 5, "b3 should fully match");

    /* Verify no cross-contamination: query b1 extended should match only b1 */
    int32_t q[] = {1, 2, 3, 4, 99};
    NxtMatchResult rq = nxt_prefix_match(&tree, q, 5);
    ASSERT(rq.matched_tokens == 4, "should match 4 tokens (b1 prefix)");

    nxt_prefix_tree_destroy(&tree);
    PASS();
}

/* ── Test: Statistics ────────────────────────────────────────────────── */
static void test_statistics(void) {
    TEST("statistics");
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, NULL);

    ASSERT(nxt_prefix_total_tokens(&tree) == 0, "empty tree has zero tokens");
    ASSERT(nxt_prefix_node_count(&tree) == 1, "empty tree has only root");

    int32_t tokens[] = {1, 2, 3};
    uint32_t pages[] = {10, 20, 30};
    nxt_prefix_insert(&tree, tokens, 3, pages, 3, 0);

    ASSERT(nxt_prefix_total_tokens(&tree) == 3, "3 tokens cached");
    ASSERT(nxt_prefix_node_count(&tree) == 2, "root + 1 data node");
    ASSERT(nxt_prefix_evictable_size(&tree) == 3, "3 evictable tokens");

    nxt_prefix_tree_destroy(&tree);
    PASS();
}

/* ── Test Runner ────────────────────────────────────────────────────── */
int main(void) {
    printf("=== xLLM Prefix Sharing Test Suite ===\n\n");

    test_tree_init_destroy();
    test_insert_single();
    test_match_full();
    test_match_partial();
    test_shared_prefix();
    test_empty_query();
    test_lock_unlock();
    test_eviction();
    test_divergent_branches();
    test_statistics();

    printf("\n=== Results: %d run, %d passed, %d failed ===\n",
           tests_run, tests_passed, tests_failed);

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
