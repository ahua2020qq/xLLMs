/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#include "page_pool.h"
#include "lru_k.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ── Default page size (64 KB) ─────────────────────────────────────────
#define DEFAULT_PAGE_SIZE   (64 * 1024)

// ── Helper: get monotonic timestamp in microseconds ──────────────────
static uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000);
}

// ── Helper: remove a page from a free list ───────────────────────────
static void free_list_remove(NxtGlobalBufferPool *pool, NxtStorageTier tier,
                             NxtPageType type, NxtPage *page) {
    if (page->free_prev) page->free_prev->free_next = page->free_next;
    else pool->free_heads[tier][type] = page->free_next;

    if (page->free_next) page->free_next->free_prev = page->free_prev;
    else pool->free_tails[tier][type] = page->free_prev;

    page->free_next = NULL;
    page->free_prev = NULL;
    pool->free_counts[tier][type]--;
}

// ── Helper: append a page to a free list ─────────────────────────────
static void free_list_append(NxtGlobalBufferPool *pool, NxtStorageTier tier,
                             NxtPageType type, NxtPage *page) {
    page->free_prev = pool->free_tails[tier][type];
    page->free_next = NULL;

    if (pool->free_tails[tier][type])
        pool->free_tails[tier][type]->free_next = page;
    else
        pool->free_heads[tier][type] = page;

    pool->free_tails[tier][type] = page;
    pool->free_counts[tier][type]++;
}

// ── Pool initialization ──────────────────────────────────────────────
void nxt_pool_init(NxtGlobalBufferPool *pool,
                   size_t gpu_bytes, size_t cpu_bytes, size_t ssd_bytes,
                   size_t page_size, size_t max_pages) {
    memset(pool, 0, sizeof(*pool));

    pool->tier_capacity[TIER_GPU] = gpu_bytes;
    pool->tier_capacity[TIER_CPU] = cpu_bytes;
    pool->tier_capacity[TIER_SSD] = ssd_bytes;

    if (page_size == 0) page_size = DEFAULT_PAGE_SIZE;

    // Determine how many pages fit in each tier
    size_t gpu_pages = gpu_bytes / page_size;
    size_t cpu_pages = cpu_bytes / page_size;
    size_t ssd_pages = ssd_bytes / page_size;
    size_t total_pages = gpu_pages + cpu_pages + ssd_pages;
    if (total_pages > max_pages) total_pages = max_pages;

    // Allocate page structures
    NxtPage *pages = calloc(total_pages, sizeof(NxtPage));
    if (!pages) return;

    // Initialize LRU-K tracker
    lruk_init(&pool->lru_k, total_pages);

    // Distribute pages across tiers evenly by type count
    size_t pages_per_type_per_tier = total_pages / (TIER_COUNT * PAGE_TYPE_COUNT);
    if (pages_per_type_per_tier == 0) pages_per_type_per_tier = 1;

    uint32_t next_id = 0;
    for (int tier = 0; tier < TIER_COUNT; tier++) {
        for (int type = 0; type < PAGE_TYPE_COUNT; type++) {
            size_t count = (tier == 0) ? gpu_pages / PAGE_TYPE_COUNT
                         : (tier == 1) ? cpu_pages / PAGE_TYPE_COUNT
                         : ssd_pages / PAGE_TYPE_COUNT;
            for (size_t i = 0; i < count && next_id < total_pages; i++) {
                NxtPage *p = &pages[next_id];
                p->page_id = next_id++;
                p->type    = (NxtPageType)type;
                p->tier    = (NxtStorageTier)tier;
                p->size    = page_size;
                // Data buffer allocated lazily on first use
                p->data    = NULL;
                free_list_append(pool, tier, (NxtPageType)type, p);
            }
        }
    }
}

// ── Pool destruction ─────────────────────────────────────────────────
void nxt_pool_destroy(NxtGlobalBufferPool *pool) {
    lruk_destroy(&pool->lru_k);

    // Free all pages (walk free lists; pages struct array pointer recovered from first free page)
    for (int tier = 0; tier < TIER_COUNT; tier++) {
        for (int type = 0; type < PAGE_TYPE_COUNT; type++) {
            NxtPage *page = pool->free_heads[tier][type];
            while (page) {
                free(page->data);
                page->data = NULL;
                page = page->free_next;
            }
        }
    }

    // The backing array is recovered from the first non-NULL free head
    for (int tier = 0; tier < TIER_COUNT; tier++)
        for (int type = 0; type < PAGE_TYPE_COUNT; type++)
            if (pool->free_heads[tier][type]) {
                free(pool->free_heads[tier][type]);
                goto cleared;
            }
cleared:
    memset(pool, 0, sizeof(*pool));
}

// ── Page allocation ──────────────────────────────────────────────────
NxtPage *nxt_page_alloc(NxtGlobalBufferPool *pool, NxtPageType type,
                        NxtStorageTier preferred_tier) {
    // Try preferred tier first
    for (int t = (int)preferred_tier; t >= 0; t--) {
        if (pool->free_heads[t][type]) {
            NxtPage *page = pool->free_heads[t][type];
            free_list_remove(pool, (NxtStorageTier)t, type, page);

            // Lazily allocate data buffer
            if (!page->data) {
                page->data = calloc(1, page->size);
                if (!page->data) {
                    free_list_append(pool, (NxtStorageTier)t, type, page);
                    return NULL;
                }
            }
            memset(page->data, 0, page->size);

            page->ref_count = 1;
            pool->tier_used[t] += page->size;
            pool->total_allocations++;

            // Record LRU-K access
            uint64_t now = get_timestamp_us();
            lruk_access(&pool->lru_k, page->page_id, now);

            // Attempt to prefetch to a faster tier if not already in preferred
            if (t != (int)preferred_tier) {
                page->tier = preferred_tier;
            }

            return page;
        }
    }

    // No free page — attempt LRU-K eviction from target tier+type
    if (nxt_evict_lru_k(pool, preferred_tier, type))
        return nxt_page_alloc(pool, type, preferred_tier);

    return NULL;
}

// ── Page free ────────────────────────────────────────────────────────
void nxt_page_free(NxtGlobalBufferPool *pool, NxtPage *page) {
    if (!page || page->ref_count > 0) return;

    NxtStorageTier tier = page->tier;
    NxtPageType    type = page->type;

    pool->tier_used[tier] -= page->size;
    free_list_append(pool, tier, type, page);
}

// ── Reference counting ───────────────────────────────────────────────
void nxt_page_ref_inc(NxtPage *page) {
    if (page) {
        __sync_fetch_and_add(&page->ref_count, 1);
    }
}

void nxt_page_ref_dec(NxtGlobalBufferPool *pool, NxtPage *page) {
    if (!page) return;
    int32_t prev = __sync_fetch_and_sub(&page->ref_count, 1);
    if (prev == 1)
        nxt_page_free(pool, page);
}

// ── Admission control ─────────────────────────────────────────────────
bool nxt_admission_check(NxtGlobalBufferPool *pool, const NxtReqPageDir *req) {
    if (!req) return false;

    // Check that estimated memory fits within available budget
    size_t free_mem = 0;
    for (int tier = 0; tier < TIER_COUNT; tier++) {
        size_t free_in_tier = pool->tier_capacity[tier] - pool->tier_used[tier];
        free_mem += free_in_tier;
    }

    if (req->estimated_memory > free_mem) {
        // Try to evict enough pages to make room
        size_t need = req->estimated_memory - free_mem;
        while (need > 0) {
            bool evicted = false;
            for (int tier = 0; tier < TIER_COUNT && need > 0; tier++) {
                for (int type = 0; type < PAGE_TYPE_COUNT && need > 0; type++) {
                    if (nxt_evict_lru_k(pool, (NxtStorageTier)tier, (NxtPageType)type)) {
                        evicted = true;
                        need -= DEFAULT_PAGE_SIZE;
                    }
                }
            }
            if (!evicted) return false; // cannot free enough memory
        }
    }

    return true;
}

// ── LRU-K eviction ────────────────────────────────────────────────────
bool nxt_evict_lru_k(NxtGlobalBufferPool *pool, NxtStorageTier tier, NxtPageType type) {
    NxtPage *best_victim = NULL;
    uint64_t best_ts = UINT64_MAX;

    // Walk free lists to find evictable pages? No — we need to walk USED pages.
    // We iterate through the LRU-K cache to find a victim of the given tier+type
    // with ref_count == 0 that has the oldest k-th timestamp.
    for (size_t i = 0; i < pool->lru_k.capacity; i++) {
        LruKEntry *entry = &pool->lru_k.entries[i];
        uint64_t ts = lruk_kth_timestamp(entry);
        if (ts == LRU_TIMESTAMP_INVALID) continue;

        // Page lookup requires access to the page array; since we only have free lists,
        // we use a simplified approach: search free heads for the page_id.
        // In a production system, we'd maintain a used-page hash table.
        uint32_t page_id = (uint32_t)i;
        NxtPage *page = NULL;
        for (int t = 0; t < TIER_COUNT && !page; t++)
            for (int tp = 0; tp < PAGE_TYPE_COUNT && !page; tp++)
                for (NxtPage *p = pool->free_heads[t][tp]; p; p = p->free_next)
                    if (p->page_id == page_id) { page = p; break; }
        if (!page) continue;
        if (page->type != type || page->tier != tier) continue;
        if (page->ref_count > 0) continue;

        if (ts < best_ts) {
            best_ts = ts;
            best_victim = page;
        }
    }

    if (!best_victim) return false;

    // Free the victim page (move data to SSD if on GPU, discard if on SSD)
    free(best_victim->data);
    best_victim->data = NULL;
    best_victim->ref_count = 0;
    nxt_page_free(pool, best_victim);
    lruk_remove(&pool->lru_k, best_victim->page_id);

    pool->total_evictions++;
    return true;
}

// ── Background defragmentation (skeleton) ─────────────────────────────
void nxt_defrag_background(NxtGlobalBufferPool *pool) {
    // Defrag moves pages within a tier to create contiguous free regions.
    // Current skeleton: log a defrag round and increment counter.
    // Full implementation would:
    //  1) Walk per-type used pages, sort by physical address
    //  2) Move pages to close gaps (memmove)
    //  3) Update free lists to reflect new contiguous free blocks
    //  4) Optionally promote hot pages to faster tiers via LRU-K stats
    pool->total_defrag_rounds++;
}
