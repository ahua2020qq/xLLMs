/*
 * xLLM — Next-Generation LLM Inference Engine
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
#include <pthread.h>
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

// ── Defrag strategy (pluggable policy) ────────────────────────────────
typedef enum {
    DEFRAG_STRATEGY_TIERED  = 0,  // Universal-style: merge adjacent free blocks
    DEFRAG_STRATEGY_LEVELED = 1,  // Leveled-style: move hot pages toward faster tiers
    DEFRAG_STRATEGY_GREEDY  = 2,  // Greedy: always merge smallest free blocks first
    DEFRAG_STRATEGY_COUNT   = 3
} NxtDefragStrategy;

// ── Memory pressure levels (progressive write stall) ───────────────────
typedef enum {
    PRESSURE_NORMAL    = 0,  // normal operation
    PRESSURE_SLOWDOWN  = 1,  // slow allocations (sleep 1ms)
    PRESSURE_STOP      = 2,  // block allocations until defrag completes
} NxtMemoryPressure;

// ── Forward declaration for function pointer types ───────────────────
typedef struct NxtGlobalBufferPool NxtGlobalBufferPool;

// ── Single defrag job (compaction-style: pick → compact → install) ────
typedef struct {
    uint32_t fragmented_tier;    // tier being compacted
    uint32_t fragmented_type;    // page type being compacted
    uint32_t *victim_pages;      // selected fragmented page ids
    uint32_t  victim_count;
    uint32_t  target_count;      // target contiguous pages created
    uint64_t  bytes_reclaimed;   // bytes recovered after compaction
    uint64_t  start_ts;          // job start timestamp (us)
} NxtDefragJob;

// ── Pluggable defrag scheduler ─────────────────────────────────────────
typedef struct NxtDefragScheduler {
    NxtDefragStrategy strategy;
    double   score_threshold;        // trigger defrag when score exceeds this
    uint32_t max_concurrent_jobs;    // max concurrent defrag tasks
    uint32_t interval_ms;            // scan interval in milliseconds
    void   (*pick_source)(NxtGlobalBufferPool *, NxtDefragJob *);
    void   (*compact)(NxtGlobalBufferPool *, NxtDefragJob *);
    void   (*install)(NxtGlobalBufferPool *, NxtDefragJob *);
} NxtDefragScheduler;

// ── Single memory page ────────────────────────────────────────────────
typedef struct NxtPage {
    uint32_t      page_id;
    NxtPageType   type;           // page type (DATA / INDEX / CONTROL)
    NxtStorageTier tier;          // current storage tier
    int32_t       ref_count;      // atomic reference count
    void         *data;           // pointer to the page data
    size_t        size;           // page size in bytes
    float         utilization;    // ratio of valid data within page [0.0, 1.0]

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
typedef struct NxtGlobalBufferPool {
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

    // Pluggable defrag scheduler
    NxtDefragScheduler defrag_sched;   // active strategy + function pointers

    // Defrag scoring and background thread
    float     defrag_score;           // current fragmentation score [0.0, 1.0]
    float     defrag_threshold;       // trigger defrag when score exceeds this
    uint64_t  last_defrag_time;       // last defrag epoch seconds (time(NULL))
    uint64_t  defrag_interval_sec;    // interval between defrag scans
    bool      defrag_thread_running;  // background thread active flag
    pthread_t defrag_thread;          // background defrag thread handle

    // Global stats
    uint64_t  total_allocations;
    uint64_t  total_evictions;
    uint64_t  total_defrag_rounds;
    uint64_t  total_bytes_reclaimed;  // cumulative bytes recovered by defrag
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

// ── Background defragmentation ─────────────────────────────────────────
void nxt_defrag_background(NxtGlobalBufferPool *pool);

// ── Defrag scoring ────────────────────────────────────────────────────
float nxt_calc_defrag_score(NxtGlobalBufferPool *pool);

// ── Background defrag thread ──────────────────────────────────────────
void *nxt_defrag_thread_func(void *arg);
void  nxt_start_defrag_thread(NxtGlobalBufferPool *pool);
void  nxt_stop_defrag_thread(NxtGlobalBufferPool *pool);

// ── Compaction-style defrag (pick → compact → install) ──────────────────
void nxt_defrag_run(NxtGlobalBufferPool *pool, NxtDefragJob *job);
void nxt_defrag_pick_source_tiered(NxtGlobalBufferPool *pool, NxtDefragJob *job);
void nxt_defrag_pick_source_leveled(NxtGlobalBufferPool *pool, NxtDefragJob *job);
void nxt_defrag_pick_source_greedy(NxtGlobalBufferPool *pool, NxtDefragJob *job);
void nxt_defrag_compact(NxtGlobalBufferPool *pool, NxtDefragJob *job);
void nxt_defrag_install(NxtGlobalBufferPool *pool, NxtDefragJob *job);

// ── Pluggable strategy configuration ─────────────────────────────────────
void nxt_defrag_scheduler_init(NxtDefragScheduler *sched, NxtDefragStrategy strategy);
void nxt_defrag_set_strategy(NxtGlobalBufferPool *pool, NxtDefragStrategy strategy);

// ── Compensatory eviction ────────────────────────────────────────────────
void    nxt_page_set_utilization(NxtPage *page, float util);
uint64_t nxt_page_eviction_priority(NxtPage *page, LruKCache *lru_k,
                                     uint64_t now_ts, float compensation_factor);
bool    nxt_evict_lru_k_compensated(NxtGlobalBufferPool *pool, NxtStorageTier tier,
                                     NxtPageType type, float compensation_factor);

// ── Memory pressure (progressive write stall) ────────────────────────────
NxtMemoryPressure nxt_check_memory_pressure(NxtGlobalBufferPool *pool);

#endif // PAGE_POOL_H
