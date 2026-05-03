/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#include "lru_k.h"
#include <stdlib.h>
#include <string.h>

void lruk_init(LruKCache *cache, size_t capacity) {
    cache->capacity = capacity;
    cache->entries = calloc(capacity, sizeof(LruKEntry));
}

void lruk_destroy(LruKCache *cache) {
    free(cache->entries);
    cache->entries = NULL;
    cache->capacity = 0;
}

void lruk_insert(LruKCache *cache, uint32_t page_id, uint64_t now_ts) {
    if (!cache->entries || page_id >= cache->capacity) return;
    LruKEntry *e = &cache->entries[page_id];
    memset(e->timestamps, 0, sizeof(e->timestamps));
    e->timestamps[0] = now_ts;
}

void lruk_access(LruKCache *cache, uint32_t page_id, uint64_t now_ts) {
    if (!cache->entries || page_id >= cache->capacity) return;
    LruKEntry *e = &cache->entries[page_id];

    // Shift timestamps right, drop oldest, insert newest at [0]
    if (e->timestamps[0] == LRU_TIMESTAMP_INVALID) {
        // First access
        e->timestamps[0] = now_ts;
        return;
    }

    memmove(&e->timestamps[1], &e->timestamps[0],
            (LRU_K - 1) * sizeof(uint64_t));
    e->timestamps[0] = now_ts;
}

void lruk_remove(LruKCache *cache, uint32_t page_id) {
    if (!cache->entries || page_id >= cache->capacity) return;
    memset(&cache->entries[page_id], 0, sizeof(LruKEntry));
}

bool lruk_select_victim(LruKCache *cache, uint32_t *victim_id) {
    if (!cache->entries || !victim_id) return false;

    uint64_t best_ts = UINT64_MAX;
    bool     found   = false;

    for (size_t i = 0; i < cache->capacity; i++) {
        uint64_t ts = lruk_kth_timestamp(&cache->entries[i]);
        if (ts == LRU_TIMESTAMP_INVALID) continue;
        if (ts < best_ts) {
            best_ts = ts;
            *victim_id = (uint32_t)i;
            found = true;
        }
    }
    return found;
}

int lruk_compare(const LruKEntry *a, const LruKEntry *b) {
    uint64_t ta = lruk_kth_timestamp(a);
    uint64_t tb = lruk_kth_timestamp(b);

    if (ta == LRU_TIMESTAMP_INVALID && tb == LRU_TIMESTAMP_INVALID) return 0;
    if (ta == LRU_TIMESTAMP_INVALID) return -1;  // empty entries go first
    if (tb == LRU_TIMESTAMP_INVALID) return 1;
    if (ta < tb) return -1;
    if (ta > tb) return 1;
    return 0;
}

uint64_t lruk_kth_timestamp(const LruKEntry *entry) {
    if (!entry) return LRU_TIMESTAMP_INVALID;

    // Count valid entries
    int count = 0;
    for (int i = 0; i < LRU_K; i++) {
        if (entry->timestamps[i] != LRU_TIMESTAMP_INVALID) count++;
    }
    if (count == 0) return LRU_TIMESTAMP_INVALID;

    // The k-th timestamp is the oldest retained one, at index [count-1]
    return entry->timestamps[count - 1];
}

bool lruk_is_mature(const LruKEntry *entry) {
    if (!entry) return false;
    for (int i = 0; i < LRU_K; i++)
        if (entry->timestamps[i] == LRU_TIMESTAMP_INVALID)
            return false;
    return true;
}
