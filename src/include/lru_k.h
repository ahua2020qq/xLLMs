/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#ifndef LRU_K_H
#define LRU_K_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define LRU_K         3
#define LRU_TIMESTAMP_INVALID 0

typedef struct {
    uint64_t timestamps[LRU_K];   // K=3 most recent access timestamps (ring buffer, newest at index 0)
} LruKEntry;

typedef struct {
    LruKEntry *entries;           // array of entries, index = page_id
    size_t      capacity;         // max entries
} LruKCache;

// Lifecycle
void lruk_init(LruKCache *cache, size_t capacity);
void lruk_destroy(LruKCache *cache);

// Insert a page entry into the LRU-K tracker (first access)
void lruk_insert(LruKCache *cache, uint32_t page_id, uint64_t now_ts);

// Record a new access timestamp for an existing page (shifts ring buffer)
void lruk_access(LruKCache *cache, uint32_t page_id, uint64_t now_ts);

// Remove a page entry from the tracker
void lruk_remove(LruKCache *cache, uint32_t page_id);

// Select the victim page_id with the oldest k-th (K=3rd) timestamp
// Returns true if a victim is found, false if cache is empty
bool lruk_select_victim(LruKCache *cache, uint32_t *victim_id);

// Compare two entries by their k-th (oldest retained) timestamp
// Returns negative if a is older than b (a should be evicted first)
int  lruk_compare(const LruKEntry *a, const LruKEntry *b);

// Get the k-th (oldest) timestamp of an entry
uint64_t lruk_kth_timestamp(const LruKEntry *entry);

// Check if an entry has accumulated K accesses
bool lruk_is_mature(const LruKEntry *entry);

#endif // LRU_K_H
