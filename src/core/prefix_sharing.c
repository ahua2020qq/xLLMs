/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#include "prefix_sharing.h"
#include "page_pool.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Default initial capacity for child arrays ───────────────────────── */
#define CHILD_INIT_CAP  4

/* ── Monotonic timestamp in µs ───────────────────────────────────────── */
static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000);
}

/* ── Internal: create a new node ─────────────────────────────────────── */
static NxtPrefixNode *node_create(NxtPrefixTree *tree,
                                   const int32_t *tokens, int32_t n_tokens,
                                   const uint32_t *pages, int32_t n_pages,
                                   int32_t priority) {
    NxtPrefixNode *node = calloc(1, sizeof(NxtPrefixNode));
    if (!node) return NULL;

    if (n_tokens > 0) {
        node->token_ids = malloc((size_t)n_tokens * sizeof(int32_t));
        if (!node->token_ids) { free(node); return NULL; }
        memcpy(node->token_ids, tokens, (size_t)n_tokens * sizeof(int32_t));
    }
    node->token_count = n_tokens;

    if (n_pages > 0) {
        node->page_ids = malloc((size_t)n_pages * sizeof(uint32_t));
        if (!node->page_ids) { free(node->token_ids); free(node); return NULL; }
        memcpy(node->page_ids, pages, (size_t)n_pages * sizeof(uint32_t));
    }
    node->page_count = n_pages;

    node->children = calloc(CHILD_INIT_CAP, sizeof(NxtPrefixNode *));
    if (!node->children) {
        free(node->page_ids); free(node->token_ids); free(node);
        return NULL;
    }
    node->child_capacity = CHILD_INIT_CAP;

    node->priority = priority;
    node->creation_time = now_us();
    node->last_access_time = node->creation_time;

    tree->total_nodes++;
    return node;
}

/* ── Internal: recursively destroy a node and its descendants ────────── */
static void node_destroy(NxtPrefixTree *tree, NxtPrefixNode *node) {
    if (!node) return;
    for (int32_t i = 0; i < node->child_count; i++)
        node_destroy(tree, node->children[i]);
    free(node->token_ids);
    free(node->page_ids);
    free(node->children);
    free(node);
    tree->total_nodes--;
}

/* ── Internal: find child whose first token matches `first_tok` ──────── */
static NxtPrefixNode *find_child(const NxtPrefixNode *node, int32_t first_tok) {
    for (int32_t i = 0; i < node->child_count; i++) {
        NxtPrefixNode *c = node->children[i];
        if (c->token_count > 0 && c->token_ids[0] == first_tok)
            return c;
    }
    return NULL;
}

/* ── Internal: add a child to a node (grows child array if needed) ───── */
static bool add_child(NxtPrefixNode *parent, NxtPrefixNode *child) {
    if (parent->child_count >= parent->child_capacity) {
        int32_t new_cap = parent->child_capacity * 2;
        NxtPrefixNode **new_arr = realloc(parent->children,
                                           (size_t)new_cap * sizeof(NxtPrefixNode *));
        if (!new_arr) return false;
        parent->children = new_arr;
        parent->child_capacity = new_cap;
    }
    parent->children[parent->child_count++] = child;
    child->parent = parent;
    return true;
}

/* ── Internal: remove a child from its parent ────────────────────────── */
static void remove_child(NxtPrefixNode *parent, NxtPrefixNode *child) {
    for (int32_t i = 0; i < parent->child_count; i++) {
        if (parent->children[i] == child) {
            /* compact the array */
            memmove(&parent->children[i], &parent->children[i + 1],
                    (size_t)(parent->child_count - i - 1) * sizeof(NxtPrefixNode *));
            parent->child_count--;
            child->parent = NULL;
            return;
        }
    }
}

/* ── Internal: compute shared prefix length between two token arrays ─── */
static int32_t common_prefix_len(const int32_t *a, int32_t na,
                                  const int32_t *b, int32_t nb) {
    int32_t i = 0;
    int32_t lim = (na < nb) ? na : nb;
    while (i < lim && a[i] == b[i]) i++;
    return i;
}

/* ── Internal: split a node at a given prefix length ──────────────────── */
static NxtPrefixNode *split_node(NxtPrefixTree *tree,
                                  NxtPrefixNode *child, int32_t split_len) {
    /* Create new node taking the shared prefix portion */
    NxtPrefixNode *new_node = node_create(tree,
        child->token_ids, split_len,
        child->page_ids, split_len > child->page_count ? child->page_count : split_len,
        child->priority);
    if (!new_node) return NULL;

    new_node->hit_count = child->hit_count;
    new_node->lock_ref = child->lock_ref;

    /* Re-parent new_node */
    new_node->parent = child->parent;
    if (child->parent) {
        /* Replace child with new_node in parent's children array */
        for (int32_t i = 0; i < child->parent->child_count; i++) {
            if (child->parent->children[i] == child) {
                child->parent->children[i] = new_node;
                break;
            }
        }
    }

    /* Trim child */
    int32_t child_new_len = child->token_count - split_len;
    int32_t child_page_offset = split_len;
    if (child_page_offset > child->page_count)
        child_page_offset = child->page_count;

    /* Shift child's tokens */
    if (child_new_len > 0) {
        int32_t *new_tokens = malloc((size_t)child_new_len * sizeof(int32_t));
        if (new_tokens) {
            memcpy(new_tokens, child->token_ids + split_len,
                   (size_t)child_new_len * sizeof(int32_t));
        }
        free(child->token_ids);
        child->token_ids = new_tokens;
    } else {
        free(child->token_ids);
        child->token_ids = NULL;
    }
    child->token_count = child_new_len;

    /* Shift child's page_ids */
    int32_t child_new_pages = child->page_count - child_page_offset;
    if (child_new_pages > 0) {
        uint32_t *new_pages = malloc((size_t)child_new_pages * sizeof(uint32_t));
        if (new_pages) {
            memcpy(new_pages, child->page_ids + child_page_offset,
                   (size_t)child_new_pages * sizeof(uint32_t));
        }
        free(child->page_ids);
        child->page_ids = new_pages;
    } else {
        free(child->page_ids);
        child->page_ids = NULL;
    }
    child->page_count = child_new_pages;

    /* Make child a child of new_node */
    add_child(new_node, child);

    return new_node;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════ */

void nxt_prefix_tree_init(NxtPrefixTree *tree, NxtGlobalBufferPool *pool) {
    memset(tree, 0, sizeof(*tree));
    tree->pool = pool;

    /* Root node: empty key, no pages, lock_ref = 1 (permanently protected) */
    tree->root = node_create(tree, NULL, 0, NULL, 0, 0);
    if (tree->root) {
        tree->root->lock_ref = 1;
        tree->root->page_ids = calloc(0, 0);  /* non-NULL sentinel */
    }
}

void nxt_prefix_tree_destroy(NxtPrefixTree *tree) {
    if (tree->root) node_destroy(tree, tree->root);
    memset(tree, 0, sizeof(*tree));
}

NxtMatchResult nxt_prefix_match(NxtPrefixTree *tree,
                                 const int32_t *token_ids, int32_t len) {
    NxtMatchResult result;
    memset(&result, 0, sizeof(result));
    result.last_node = tree->root;

    if (!tree->root || len == 0 || !token_ids) return result;

    uint64_t now = now_us();
    NxtPrefixNode *node = tree->root;
    node->last_access_time = now;

    int32_t offset = 0;

    while (offset < len) {
        NxtPrefixNode *child = find_child(node, token_ids[offset]);
        if (!child) break;

        child->last_access_time = now;
        child->hit_count++;

        int32_t match = common_prefix_len(child->token_ids, child->token_count,
                                           token_ids + offset, len - offset);

        if (match < child->token_count) {
            /* Partial match: split the child */
            NxtPrefixNode *new_node = split_node(tree, child, match);
            if (!new_node) break;

            /* Append new_node's pages to result */
            if (result.page_count + new_node->page_count <= NXT_PREFIX_MAX_MATCH_PAGES) {
                memcpy(result.page_ids + result.page_count, new_node->page_ids,
                       (size_t)new_node->page_count * sizeof(uint32_t));
                result.page_count += new_node->page_count;
            }
            result.matched_tokens += match;
            result.last_node = new_node;
            break;
        }

        /* Full match: append child's pages */
        if (result.page_count + child->page_count <= NXT_PREFIX_MAX_MATCH_PAGES) {
            memcpy(result.page_ids + result.page_count, child->page_ids,
                   (size_t)child->page_count * sizeof(uint32_t));
            result.page_count += child->page_count;
        }
        result.matched_tokens += match;
        result.last_node = child;

        offset += match;
        node = child;
    }

    return result;
}

int32_t nxt_prefix_insert(NxtPrefixTree *tree,
                           const int32_t *token_ids, int32_t len,
                           const uint32_t *page_ids, int32_t page_count,
                           int32_t priority) {
    if (!tree->root || !token_ids || len == 0) return 0;

    uint64_t now = now_us();
    NxtPrefixNode *node = tree->root;
    node->last_access_time = now;

    int32_t offset = 0;

    while (offset < len) {
        NxtPrefixNode *child = find_child(node, token_ids[offset]);
        if (!child) break;

        child->last_access_time = now;
        child->priority = (child->priority > priority) ? child->priority : priority;

        int32_t match = common_prefix_len(child->token_ids, child->token_count,
                                           token_ids + offset, len - offset);

        if (match < child->token_count) {
            /* Need to split and create a branch */
            NxtPrefixNode *new_node = split_node(tree, child, match);
            if (!new_node) return -1;
            new_node->priority = (new_node->priority > priority) ? new_node->priority : priority;
            node = new_node;
            offset += match;
            break;
        }

        offset += match;
        node = child;
    }

    /* Create remaining path */
    if (offset < len) {
        int32_t remaining_tokens = len - offset;
        int32_t pages_used = offset; /* naive mapping: 1 page per token, adjust as needed */
        int32_t remaining_pages = page_count - pages_used;
        if (remaining_pages < 0) remaining_pages = 0;

        NxtPrefixNode *new_node = node_create(tree,
            token_ids + offset, remaining_tokens,
            page_ids + pages_used, remaining_pages,
            priority);
        if (!new_node) return -1;

        if (!add_child(node, new_node)) {
            node_destroy(tree, new_node);
            return -1;
        }

        tree->evictable_size += remaining_tokens;
    }

    return offset; /* shared prefix length */
}

int32_t nxt_prefix_evict(NxtPrefixTree *tree, int32_t num_tokens) {
    if (!tree->root || num_tokens <= 0) return 0;

    int32_t evicted = 0;

    /* Find evictable leaves via simple BFS/DFS collection */
    /* Strategy: walk tree, collect evictable leaves, sort by LRU, evict oldest */
    /* For simplicity, do iterative passes finding the single oldest leaf each time */

    while (evicted < num_tokens) {
        /* Collect all evictable leaves */
        NxtPrefixNode *leaves[1024];
        int32_t leaf_count = 0;

        /* DFS stack */
        NxtPrefixNode *stack[1024];
        int32_t stack_top = 0;
        stack[stack_top++] = tree->root;

        while (stack_top > 0 && leaf_count < 1024) {
            NxtPrefixNode *n = stack[--stack_top];

            bool has_live_child = false;
            for (int32_t i = 0; i < n->child_count; i++) {
                NxtPrefixNode *c = n->children[i];
                if (!c->evicted && c->page_ids) {
                    has_live_child = true;
                    if (stack_top < 1024)
                        stack[stack_top++] = c;
                }
            }

            /* A leaf is evictable if: no live children, not locked, has pages, not root */
            if (!has_live_child && n->lock_ref == 0 && n->page_count > 0 && n != tree->root) {
                leaves[leaf_count++] = n;
            }
        }

        if (leaf_count == 0) break; /* nothing evictable */

        /* Find the least-recently-used leaf */
        NxtPrefixNode *victim = leaves[0];
        for (int32_t i = 1; i < leaf_count; i++) {
            if (leaves[i]->last_access_time < victim->last_access_time)
                victim = leaves[i];
        }

        /* Free pages back to pool */
        if (tree->pool) {
            for (int32_t i = 0; i < victim->page_count; i++) {
                /* We'd need to look up the NxtPage* by page_id.
                 * Since the pool only stores free lists, we mark pages as freed
                 * and decrement the pool's used counter.
                 * In production, maintain a page-id → NxtPage* hash. */
                if (tree->pool->tier_used[0] >= 64 * 1024)
                    tree->pool->tier_used[0] -= 64 * 1024;
            }
        }

        evicted += victim->token_count;
        tree->evictable_size -= victim->token_count;

        /* Detach from parent */
        if (victim->parent) {
            remove_child(victim->parent, victim);
        }

        /* Destroy the node */
        node_destroy(tree, victim);
    }

    return evicted;
}

void nxt_prefix_lock(NxtPrefixTree *tree, NxtPrefixNode *node) {
    if (!node) return;
    NxtPrefixNode *cur = node;
    while (cur != tree->root && cur) {
        if (cur->lock_ref == 0) {
            tree->evictable_size -= cur->token_count;
            tree->protected_size += cur->token_count;
        }
        cur->lock_ref++;
        cur = cur->parent;
    }
}

void nxt_prefix_unlock(NxtPrefixTree *tree, NxtPrefixNode *node) {
    if (!node) return;
    NxtPrefixNode *cur = node;
    while (cur != tree->root && cur) {
        cur->lock_ref--;
        if (cur->lock_ref == 0) {
            tree->evictable_size += cur->token_count;
            tree->protected_size -= cur->token_count;
        }
        cur = cur->parent;
    }
}

int32_t nxt_prefix_total_tokens(const NxtPrefixTree *tree) {
    if (!tree->root) return 0;
    int32_t total = 0;
    NxtPrefixNode *stack[1024];
    int32_t top = 0;
    stack[top++] = tree->root;
    while (top > 0) {
        NxtPrefixNode *n = stack[--top];
        total += n->token_count;
        for (int32_t i = 0; i < n->child_count; i++)
            if (!n->children[i]->evicted)
                stack[top++] = n->children[i];
    }
    return total;
}

int32_t nxt_prefix_node_count(const NxtPrefixTree *tree) {
    return tree->total_nodes;
}

int32_t nxt_prefix_evictable_size(const NxtPrefixTree *tree) {
    return tree->evictable_size;
}
