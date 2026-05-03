/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#ifndef PAGE_POOL_H
#define PAGE_POOL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "lru_k.h"

// ── Page type enumeration (three levels) ──────────────────────────────
typedef enum {
    PAGE_TYPE_DATA    = 0,  // stores model weights / KV-cache data
    PAGE_TYPE_INDEX   = 1,  // stores page-table entries / metadata
    PAGE_TYPE_CONTROL = 2,  // stores control info / request metadata
    PAGE_TYPE_COUNT   = 3
} NxtPageType;

// ── Storage tier enumeration ──────────────────────────────────────────
typedef enum {
    TIER_GPU = 0,  // HBM (High Bandwidth Memory)
    TIER_CPU = 1,  // DRAM (host memory)
    TIER_SSD = 2,  // NVMe / disk
    TIER_COUNT = 3
} NxtStorageTier;

// ── Single memory page ────────────────────────────────────────────────
typedef struct NxtPage {
    uint32_t      page_id;
    NxtPageType   type;           // page type (DATA / INDEX / CONTROL)
    NxtStorageTier tier;          // current storage tier
    int32_t       ref_count;      // atomic reference count
    void         *data;           // pointer to the page data
    size_t        size;           // page size in bytes

    // LRU-K timestamps are stored externally in LruKCache (indexed by page_id)

    struct NxtPage *free_next;    // next page in free list (or NULL)
    struct NxtPage *free_prev;    // prev page in free list (or NULL)
} NxtPage;

// ── Per-request page directory ────────────────────────────────────────
#define NXT_MAX_PAGES_PER_REQUEST 256

typedef struct {
    uint64_t  request_id;                     // unique request identifier
    uint32_t  page_ids[NXT_MAX_PAGES_PER_REQUEST];
    uint32_t  page_count;
    uint32_t  priority;                       // scheduling priority (0 = highest)
    uint64_t  arrival_ts;                     // arrival timestamp for admission control
    size_t    estimated_memory;               // estimated total memory needed
} NxtReqPageDir;

// ── Hash table entry (open addressing, linear probing) ─────────────────
#define NXT_HASH_EMPTY 0xFFFFFFFFu

typedef struct {
    uint32_t key;    // page_id
    NxtPage *value;  // pointer to page (NULL = tombstone / free)
} NxtHashEntry;

typedef struct {
    NxtHashEntry *entries;
    size_t        capacity;    // must be power of 2
    size_t        size;        // active entries
} NxtPageHash;

// ── Global buffer pool ────────────────────────────────────────────────
typedef struct {
    // Per-tier, per-type free page lists (doubly-linked)
    NxtPage  *free_heads[TIER_COUNT][PAGE_TYPE_COUNT];
    NxtPage  *free_tails[TIER_COUNT][PAGE_TYPE_COUNT];
    uint32_t  free_counts[TIER_COUNT][PAGE_TYPE_COUNT];

    // LRU-K tracker (shared across all pages)
    LruKCache lru_k;

    // page_id → NxtPage* hash table for fast lookup
    NxtPageHash page_hash;

    // Backing array of all page structs (for cleanup)
    NxtPage  *pages_array;
    size_t    pages_count;       // actual number of initialized pages

    // Memory budget per tier (bytes)
    size_t    tier_capacity[TIER_COUNT];
    size_t    tier_used[TIER_COUNT];

    // Global stats
    uint64_t  total_allocations;
    uint64_t  total_evictions;
    uint64_t  total_defrag_rounds;
} NxtGlobalBufferPool;

// ── Hash table operations ─────────────────────────────────────────────
void nxt_hash_init(NxtPageHash *hash, size_t capacity);
void nxt_hash_destroy(NxtPageHash *hash);
void nxt_hash_insert(NxtPageHash *hash, uint32_t key, NxtPage *value);
NxtPage *nxt_hash_lookup(NxtPageHash *hash, uint32_t key);
bool nxt_hash_remove(NxtPageHash *hash, uint32_t key);

// ── Buffer pool lifecycle ─────────────────────────────────────────────
void nxt_pool_init(NxtGlobalBufferPool *pool,
                   size_t gpu_bytes, size_t cpu_bytes, size_t ssd_bytes,
                   size_t page_size, size_t max_pages);
void nxt_pool_destroy(NxtGlobalBufferPool *pool);

// ── Page allocation / free ────────────────────────────────────────────
NxtPage *nxt_page_alloc(NxtGlobalBufferPool *pool, NxtPageType type, NxtStorageTier preferred_tier);
void      nxt_page_free(NxtGlobalBufferPool *pool, NxtPage *page);

// ── Reference counting ────────────────────────────────────────────────
void nxt_page_ref_inc(NxtPage *page);
void nxt_page_ref_dec(NxtGlobalBufferPool *pool, NxtPage *page);

// ── Admission control ─────────────────────────────────────────────────
bool nxt_admission_check(NxtGlobalBufferPool *pool, const NxtReqPageDir *req);

// ── LRU-K eviction ────────────────────────────────────────────────────
bool nxt_evict_lru_k(NxtGlobalBufferPool *pool, NxtStorageTier tier, NxtPageType type);

// ── Background defragmentation (skeleton) ─────────────────────────────
void nxt_defrag_background(NxtGlobalBufferPool *pool);

#endif // PAGE_POOL_H
