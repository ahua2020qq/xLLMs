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
#include <stdio.h>
#include <unistd.h>

// ── Default page size (64 KB) ─────────────────────────────────────────
#define DEFAULT_PAGE_SIZE   (64 * 1024)

// ── Hash table: open addressing with linear probing ───────────────────
// capacity is rounded up to the next power of 2 internally

static inline size_t next_pow2(size_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    v |= v >> 32;
#endif
    return v + 1;
}

void nxt_hash_init(NxtPageHash *hash, size_t capacity) {
    memset(hash, 0, sizeof(*hash));
    hash->capacity = next_pow2(capacity < 16 ? 16 : capacity);
    hash->entries = calloc(hash->capacity, sizeof(NxtHashEntry));
    if (hash->entries) {
        for (size_t i = 0; i < hash->capacity; i++)
            hash->entries[i].key = NXT_HASH_EMPTY;
    }
}

void nxt_hash_destroy(NxtPageHash *hash) {
    free(hash->entries);
    memset(hash, 0, sizeof(*hash));
}

static inline size_t hash_probe(NxtPageHash *hash, uint32_t key, size_t i) {
    return (key + i) & (hash->capacity - 1);  // capacity is power of 2
}

void nxt_hash_insert(NxtPageHash *hash, uint32_t key, NxtPage *value) {
    if (!hash->entries || key == NXT_HASH_EMPTY) return;
    for (size_t i = 0; i < hash->capacity; i++) {
        size_t idx = hash_probe(hash, key, i);
        if (hash->entries[idx].key == NXT_HASH_EMPTY || hash->entries[idx].key == key) {
            hash->entries[idx].key   = key;
            hash->entries[idx].value = value;
            hash->size++;
            return;
        }
    }
}

NxtPage *nxt_hash_lookup(NxtPageHash *hash, uint32_t key) {
    if (!hash->entries || key == NXT_HASH_EMPTY) return NULL;
    for (size_t i = 0; i < hash->capacity; i++) {
        size_t idx = hash_probe(hash, key, i);
        if (hash->entries[idx].key == NXT_HASH_EMPTY) return NULL;
        if (hash->entries[idx].key == key) return hash->entries[idx].value;
    }
    return NULL;
}

bool nxt_hash_remove(NxtPageHash *hash, uint32_t key) {
    if (!hash->entries || key == NXT_HASH_EMPTY) return false;
    for (size_t i = 0; i < hash->capacity; i++) {
        size_t idx = hash_probe(hash, key, i);
        if (hash->entries[idx].key == NXT_HASH_EMPTY) return false;
        if (hash->entries[idx].key == key) {
            hash->entries[idx].key   = NXT_HASH_EMPTY;
            hash->entries[idx].value = NULL;
            hash->size--;
            return true;
        }
    }
    return false;
}

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

    pool->defrag_threshold    = 0.30f;
    pool->defrag_score        = 0.0f;
    pool->defrag_interval_sec = 60;

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
    pool->pages_array = pages;

    // Initialize LRU-K tracker
    lruk_init(&pool->lru_k, total_pages);

    // Initialize page_id → NxtPage* hash table
    nxt_hash_init(&pool->page_hash, total_pages * 2);

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
                nxt_hash_insert(&pool->page_hash, p->page_id, p);
            }
        }
    }
    pool->pages_count = next_id;
}

// ── Pool destruction ─────────────────────────────────────────────────
void nxt_pool_destroy(NxtGlobalBufferPool *pool) {
    lruk_destroy(&pool->lru_k);
    nxt_hash_destroy(&pool->page_hash);

    // Free data buffers from all pages (walk pages_array)
    if (pool->pages_array) {
        for (size_t i = 0; i < pool->pages_count; i++) {
            free(pool->pages_array[i].data);
            pool->pages_array[i].data = NULL;
        }
        free(pool->pages_array);
        pool->pages_array = NULL;
    }

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

    // Iterate through the LRU-K cache and use hash table for O(1) page lookup
    for (size_t i = 0; i < pool->lru_k.capacity; i++) {
        LruKEntry *entry = &pool->lru_k.entries[i];
        uint64_t ts = lruk_kth_timestamp(entry);
        if (ts == LRU_TIMESTAMP_INVALID) continue;

        uint32_t page_id = (uint32_t)i;
        NxtPage *page = nxt_hash_lookup(&pool->page_hash, page_id);
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

// ── Helper: compare pages by page_id for sorting ──────────────────────
static int page_cmp_by_id(const void *a, const void *b) {
    const NxtPage *pa = *(const NxtPage **)a;
    const NxtPage *pb = *(const NxtPage **)b;
    if (pa->page_id < pb->page_id) return -1;
    if (pa->page_id > pb->page_id) return 1;
    return 0;
}

// ── Defrag scoring ────────────────────────────────────────────────────
// Iterates every tier×type group, measures free-page dispersion, and
// returns a pool-wide fragmentation score in [0.0, 1.0].
//   0.0 = all free pages form a single contiguous block per group
//   1.0 = free pages are maximally interleaved with used pages
float nxt_calc_defrag_score(NxtGlobalBufferPool *pool) {
    if (!pool->pages_array || pool->pages_count == 0) return 0.0f;

    float total_score = 0.0f;
    int   groups_measured = 0;

    for (int tier = 0; tier < TIER_COUNT; tier++) {
        for (int type = 0; type < PAGE_TYPE_COUNT; type++) {

            // Collect pages of this tier×type, sorted by page_id
            NxtPage *group[4096];
            size_t   count = 0;

            for (size_t i = 0; i < pool->pages_count && count < 4096; i++) {
                NxtPage *p = &pool->pages_array[i];
                if ((int)p->tier == tier && (int)p->type == type)
                    group[count++] = p;
            }
            if (count < 2) continue;

            qsort(group, count, sizeof(NxtPage *), page_cmp_by_id);

            // Count contiguous free-page runs
            size_t free_runs  = 0;
            size_t total_free = 0;
            bool   in_run     = false;

            for (size_t i = 0; i < count; i++) {
                if (group[i]->ref_count == 0) {
                    total_free++;
                    if (!in_run) { free_runs++; in_run = true; }
                } else {
                    in_run = false;
                }
            }

            if (total_free > 1) {
                float gscore = (float)(free_runs - 1) / (float)(total_free - 1);
                total_score += gscore;
            }
            groups_measured++;
        }
    }

    if (groups_measured == 0) return 0.0f;
    pool->defrag_score = total_score / (float)groups_measured;
    return pool->defrag_score;
}

// ── Background defragmentation ────────────────────────────────────────
// Within-tier compaction: moves used pages toward low page_ids, creating
// a contiguous free region at the end of each tier×type group.
//
// Algorithm:
//  1) Collect all pages of a given tier+type from pages_array
//  2) Partition into "used" (ref_count > 0) and "free" pages
//  3) Sort used pages by page_id (proxy for physical address)
//  4) For the first N used pages at the lowest page_ids, leave them in place
//  5) Remaining used pages swap data with free pages at lower page_ids
//  6) This compacts used pages to the front, freeing contiguous space at the back

void nxt_defrag_background(NxtGlobalBufferPool *pool) {
    if (!pool->pages_array) return;

    size_t total_pages = pool->pages_count;
    if (total_pages == 0) return;

    // Score-driven: skip defrag when fragmentation is below threshold
    float score = nxt_calc_defrag_score(pool);
    if (score < pool->defrag_threshold) {
        fprintf(stderr, "[nxtLLM] defrag score %.3f below threshold %.3f, skipping\n",
                (double)score, (double)pool->defrag_threshold);
        return;
    }

    // Per tier+type: collect all pages, separate used from free
    for (int tier = 0; tier < TIER_COUNT; tier++) {
        for (int type = 0; type < PAGE_TYPE_COUNT; type++) {

            // Collect all pages belonging to this tier+type
            NxtPage *group_pages[4096];
            size_t group_count = 0;

            for (size_t i = 0; i < total_pages && group_count < 4096; i++) {
                NxtPage *p = &pool->pages_array[i];
                if ((int)p->tier == tier && (int)p->type == type) {
                    group_pages[group_count++] = p;
                }
            }
            if (group_count < 2) continue;

            // Sort by page_id (ascending physical address)
            qsort(group_pages, group_count, sizeof(NxtPage *), page_cmp_by_id);

            // Partition: used pages at front, free pages at back
            // Two-pointer approach within the sorted array:
            // left  = first free slot (should be occupied by a used page)
            // right = last used page (should be moved to fill the free slot)
            size_t left = 0;
            size_t right = group_count;

            while (left < right) {
                // Advance left to first free page
                while (left < right && group_pages[left]->ref_count > 0)
                    left++;
                // Find rightmost used page
                do {
                    right--;
                } while (right > left && group_pages[right]->ref_count == 0);

                if (left >= right) break;

                // Swap data buffers between free page (left) and used page (right)
                NxtPage *free_pg = group_pages[left];
                NxtPage *used_pg = group_pages[right];

                // Swap data pointers
                void *tmp_data = free_pg->data;
                free_pg->data = used_pg->data;
                used_pg->data = tmp_data;

                // Swap ref_counts
                int32_t tmp_ref = free_pg->ref_count;
                free_pg->ref_count = used_pg->ref_count;
                used_pg->ref_count = tmp_ref;

                // Update hash table: the page_id→page mapping stays the same
                // because we're swapping data within existing page structs
                // The free list needs to be rebuilt below
            }

            // Rebuild the free list for this tier+type after compaction
            pool->free_heads[tier][type] = NULL;
            pool->free_tails[tier][type] = NULL;
            pool->free_counts[tier][type] = 0;

            for (size_t i = 0; i < group_count; i++) {
                NxtPage *p = group_pages[i];
                p->free_next = NULL;
                p->free_prev = NULL;

                if (p->ref_count == 0) {
                    // Append to free list
                    if (pool->free_tails[tier][type]) {
                        pool->free_tails[tier][type]->free_next = p;
                        p->free_prev = pool->free_tails[tier][type];
                    } else {
                        pool->free_heads[tier][type] = p;
                    }
                    pool->free_tails[tier][type] = p;
                    pool->free_counts[tier][type]++;
                }
            }
        }
    }

    pool->total_defrag_rounds++;

    fprintf(stderr, "[nxtLLM] defrag round %lu complete: "
            "tier_used=[%zu, %zu, %zu] bytes, score=%.3f\n",
            (unsigned long)pool->total_defrag_rounds,
            pool->tier_used[0], pool->tier_used[1], pool->tier_used[2],
            (double)nxt_calc_defrag_score(pool));
}

// ── Background defrag thread ──────────────────────────────────────────
void *nxt_defrag_thread_func(void *arg) {
    NxtGlobalBufferPool *pool = (NxtGlobalBufferPool *)arg;

    while (pool->defrag_thread_running) {
        sleep((unsigned int)pool->defrag_interval_sec);
        if (!pool->defrag_thread_running) break;

        nxt_defrag_background(pool);
        pool->last_defrag_time = (uint64_t)time(NULL);
    }
    return NULL;
}

void nxt_start_defrag_thread(NxtGlobalBufferPool *pool) {
    if (pool->defrag_thread_running) return;
    pool->defrag_thread_running = true;
    pthread_create(&pool->defrag_thread, NULL, nxt_defrag_thread_func, pool);
}

void nxt_stop_defrag_thread(NxtGlobalBufferPool *pool) {
    if (!pool->defrag_thread_running) return;
    pool->defrag_thread_running = false;
    pthread_join(pool->defrag_thread, NULL);
}
