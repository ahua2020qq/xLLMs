/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * nxtLLM Prefix Sharing — Radix-Tree KV-Cache Index
 *
 * Implements a radix tree (Patricia trie) over token sequences to enable
 * prefix-aware KV-cache sharing across requests.  Analogue to SGLang's
 * RadixCache, adapted for nxtLLM's three-tier page pool.
 *
 * Each node stores:
 *   - A token-ID fragment (edge label)
 *   - Page indices into NxtGlobalBufferPool (the cached KV data)
 *   - Lock reference count for active-request protection
 *
 * Integration point:
 *   - match_prefix() returns page_ids for the longest cached prefix
 *   - insert() stores new KV-cache pages in the tree
 *   - evict() frees pages back to the pool and prunes the tree
 */

#ifndef PREFIX_SHARING_H_
#define PREFIX_SHARING_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "page_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Radix-tree node ─────────────────────────────────────────────────── */
typedef struct NxtPrefixNode {
    int32_t    *token_ids;        /* edge-label token sequence           */
    int32_t     token_count;      /* length of token_ids                 */

    uint32_t   *page_ids;         /* KV-cache page indices into pool     */
    int32_t     page_count;       /* number of page_ids                  */

    struct NxtPrefixNode *parent; /* parent node (NULL for root)         */

    /* Dynamic child array */
    struct NxtPrefixNode **children;
    int32_t     child_count;
    int32_t     child_capacity;

    int32_t     lock_ref;         /* >0 = protected from eviction         */
    int32_t     hit_count;        /* cumulative prefix-match hits         */
    int32_t     priority;         /* scheduling / priority-aware eviction */
    uint64_t    last_access_time; /* monotonic timestamp (us)             */
    uint64_t    creation_time;    /* monotonic timestamp (us)             */
    bool        evicted;          /* true if pages have been freed        */
} NxtPrefixNode;

/* ── Match result ────────────────────────────────────────────────────── */
#define NXT_PREFIX_MAX_MATCH_PAGES 2048

typedef struct {
    uint32_t       page_ids[NXT_PREFIX_MAX_MATCH_PAGES];
    int32_t        page_count;
    NxtPrefixNode *last_node;     /* terminal node of the matched prefix  */
    int32_t        matched_tokens; /* number of tokens matched            */
} NxtMatchResult;

/* ── Prefix tree ─────────────────────────────────────────────────────── */
typedef struct {
    NxtPrefixNode         *root;
    NxtGlobalBufferPool   *pool;          /* bound page pool (may be NULL) */
    int32_t                total_nodes;
    int32_t                evictable_size; /* tokens in evictable nodes     */
    int32_t                protected_size; /* tokens in locked nodes        */
} NxtPrefixTree;

/* ── Lifecycle ───────────────────────────────────────────────────────── */

/** Initialise a prefix tree, optionally bound to a page pool. */
void nxt_prefix_tree_init(NxtPrefixTree *tree, NxtGlobalBufferPool *pool);

/** Destroy the prefix tree and free all node memory. Does NOT free
 *  pages in the bound pool. */
void nxt_prefix_tree_destroy(NxtPrefixTree *tree);

/* ── Lookup ──────────────────────────────────────────────────────────── */

/**
 * Find the longest cached prefix of `token_ids`.
 *
 * Returns a MatchResult with the concatenated page_ids for the cached
 * prefix.  matched_tokens reports how many input tokens were covered.
 * If no prefix is cached, page_count == 0 and last_node is the root.
 *
 * Internally updates last_access_time and hit_count on visited nodes.
 */
NxtMatchResult nxt_prefix_match(NxtPrefixTree *tree,
                                const int32_t *token_ids, int32_t len);

/* ── Insert ──────────────────────────────────────────────────────────── */

/**
 * Insert a token sequence -> page_ids mapping into the tree.
 *
 * The caller transfers ownership of the page references to the tree:
 * each page_id must have ref_count >= 1 before calling insert.
 *
 * Returns the number of tokens that were already cached (shared prefix
 * length), or -1 on allocation failure.
 */
int32_t nxt_prefix_insert(NxtPrefixTree *tree,
                           const int32_t *token_ids, int32_t len,
                           const uint32_t *page_ids, int32_t page_count,
                           int32_t priority);

/* ── Eviction ────────────────────────────────────────────────────────── */

/**
 * Evict approximately `num_tokens` worth of cached pages from the tree.
 *
 * Preferentially evicts leaves with the oldest last_access_time (LRU).
 * For each evicted node, calls nxt_page_ref_dec() on the bound pool.
 *
 * Returns the actual number of tokens evicted.
 */
int32_t nxt_prefix_evict(NxtPrefixTree *tree, int32_t num_tokens);

/* ── Locking (request-level protection) ──────────────────────────────── */

/**
 * Lock a node and all its ancestors against eviction.
 * Call when a request starts using a cached prefix.
 */
void nxt_prefix_lock(NxtPrefixTree *tree, NxtPrefixNode *node);

/**
 * Unlock a node and all its ancestors.
 * Call when a request releases a cached prefix.
 */
void nxt_prefix_unlock(NxtPrefixTree *tree, NxtPrefixNode *node);

/* ── Statistics ──────────────────────────────────────────────────────── */

/** Return total cached tokens (sum of all node token_counts). */
int32_t nxt_prefix_total_tokens(const NxtPrefixTree *tree);

/** Return total node count. */
int32_t nxt_prefix_node_count(const NxtPrefixTree *tree);

/** Return evictable token count (unlocked). */
int32_t nxt_prefix_evictable_size(const NxtPrefixTree *tree);

#ifdef __cplusplus
}
#endif

#endif /* PREFIX_SHARING_H_ */
