<!--
  nxtLLM — Next-Generation LLM Inference Engine
  Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
  SPDX-License-Identifier: Apache-2.0
  This header must not be removed. All derivative works must retain this notice.
-->

# nxtLLM V0.4 API 参考文档

## 1. 缓冲池生命周期

### nxt_pool_init

```c
void nxt_pool_init(NxtGlobalBufferPool *pool,
                   size_t gpu_bytes, size_t cpu_bytes, size_t ssd_bytes,
                   size_t page_size, size_t max_pages);
```

初始化全局缓冲池。分配页面结构体数组，按层级和类型初始化空闲链表。

| 参数 | 说明 |
|------|------|
| `pool` | 缓冲池指针 |
| `gpu_bytes` | GPU HBM 容量（字节） |
| `cpu_bytes` | CPU DRAM 容量（字节） |
| `ssd_bytes` | SSD 容量（字节） |
| `page_size` | 页面大小，传 0 使用默认 64 KB |
| `max_pages` | 最大页面数上限 |

### nxt_pool_destroy

```c
void nxt_pool_destroy(NxtGlobalBufferPool *pool);
```

释放所有页面数据缓冲区和 LRU-K 追踪器，清零池结构体。

## 2. 页面分配与释放

### nxt_page_alloc

```c
NxtPage *nxt_page_alloc(NxtGlobalBufferPool *pool,
                        NxtPageType type,
                        NxtStorageTier preferred_tier);
```

从指定类型和首选层级分配一个页面。若无空闲页面则触发 LRU-K 驱逐后重试。

**返回值**：成功返回页面指针（ref_count=1），失败返回 NULL。

**降级策略**：从 `preferred_tier` 向 TIER_GPU 方向逐级查找空闲页面。

### nxt_page_free

```c
void nxt_page_free(NxtGlobalBufferPool *pool, NxtPage *page);
```

将页面归还到对应层级+类型的空闲链表。通常不直接调用，由 `nxt_page_ref_dec` 自动触发。

**前置条件**：`page->ref_count == 0`。

## 3. 引用计数

### nxt_page_ref_inc

```c
void nxt_page_ref_inc(NxtPage *page);
```

原子递增引用计数。线程安全。

### nxt_page_ref_dec

```c
void nxt_page_ref_dec(NxtGlobalBufferPool *pool, NxtPage *page);
```

原子递减引用计数。若递减后 ref_count 归零，自动调用 `nxt_page_free`。线程安全。

## 4. 准入控制

### nxt_admission_check

```c
bool nxt_admission_check(NxtGlobalBufferPool *pool, const NxtReqPageDir *req);
```

检查请求的预估内存是否可被满足。若空间不足，尝试驱逐页面腾出空间。

| 返回值 | 说明 |
|--------|------|
| `true` | 请求被接纳 |
| `false` | 内存不足且无法驱逐足够页面，或 req 为 NULL |

## 5. LRU-K 驱逐

### nxt_evict_lru_k

```c
bool nxt_evict_lru_k(NxtGlobalBufferPool *pool,
                     NxtStorageTier tier,
                     NxtPageType type);
```

从指定层级和类型中选择 LRU-K victim 并驱逐。

| 返回值 | 说明 |
|--------|------|
| `true` | 成功驱逐一个页面 |
| `false` | 无可驱逐的页面（所有页面 ref_count > 0 或无匹配） |

## 6. 后台碎片整理

### nxt_defrag_background

```c
void nxt_defrag_background(NxtGlobalBufferPool *pool);
```

V0.1 骨架实现，仅递增 `total_defrag_rounds` 计数器。

## 7. LRU-K 追踪器 API

### lruk_init / lruk_destroy

```c
void lruk_init(LruKCache *cache, size_t capacity);
void lruk_destroy(LruKCache *cache);
```

初始化/销毁 LRU-K 追踪器。`capacity` 应等于总页面数。

### lruk_insert / lruk_access / lruk_remove

```c
void lruk_insert(LruKCache *cache, uint32_t page_id, uint64_t now_ts);
void lruk_access(LruKCache *cache, uint32_t page_id, uint64_t now_ts);
void lruk_remove(LruKCache *cache, uint32_t page_id);
```

| 函数 | 说明 |
|------|------|
| `lruk_insert` | 首次插入页面，记录初始时间戳 |
| `lruk_access` | 记录新的访问，移位时间戳环形缓冲区 |
| `lruk_remove` | 清除页面追踪记录 |

### lruk_select_victim

```c
bool lruk_select_victim(LruKCache *cache, uint32_t *victim_id);
```

选择 k-th 时间戳最旧的页面作为驱逐候选。

### lruk_kth_timestamp / lruk_is_mature / lruk_compare

```c
uint64_t lruk_kth_timestamp(const LruKEntry *entry);
bool     lruk_is_mature(const LruKEntry *entry);
int      lruk_compare(const LruKEntry *a, const LruKEntry *b);
```

| 函数 | 说明 |
|------|------|
| `lruk_kth_timestamp` | 返回第 K 次访问时间戳（驱逐依据） |
| `lruk_is_mature` | 检查是否已有 K 次访问记录 |
| `lruk_compare` | 比较两个条目的 k-th 时间戳，负值表示 a 更旧 |

## 8. 枚举与常量

### NxtPageType

| 常量 | 值 | 说明 |
|------|-----|------|
| `PAGE_TYPE_DATA` | 0 | 模型权重 / KV-cache |
| `PAGE_TYPE_INDEX` | 1 | 页表 / 索引元数据 |
| `PAGE_TYPE_CONTROL` | 2 | 控制 / 请求元数据 |
| `PAGE_TYPE_COUNT` | 3 | 类型总数 |

### NxtStorageTier

| 常量 | 值 | 说明 |
|------|-----|------|
| `TIER_GPU` | 0 | HBM |
| `TIER_CPU` | 1 | DRAM |
| `TIER_SSD` | 2 | NVMe |
| `TIER_COUNT` | 3 | 层级总数 |

### 其他常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `LRU_K` | 3 | LRU 阶数 |
| `LRU_TIMESTAMP_INVALID` | 0 | 无效时间戳标记 |
| `NXT_MAX_PAGES_PER_REQUEST` | 256 | 每请求最大页面数 |
| `DEFAULT_PAGE_SIZE` | 65536 | 默认页面大小 64 KB |

## 9. 前缀共享基数树 (V0.4)

### nxt_prefix_tree_init / nxt_prefix_tree_destroy

```c
void nxt_prefix_tree_init(NxtPrefixTree *tree, NxtGlobalBufferPool *pool);
void nxt_prefix_tree_destroy(NxtPrefixTree *tree);
```

初始化/销毁前缀树。可选绑定页池（淘汰时自动释放页面）。

### nxt_prefix_match

```c
NxtMatchResult nxt_prefix_match(NxtPrefixTree *tree,
                                const int32_t *token_ids, int32_t len);
```

查找最长缓存前缀，返回匹配的 page_id 列表和终止节点。O(L) 复杂度。

**返回值** `NxtMatchResult`：
- `page_ids[]` — 匹配到的 KV-cache 页面 ID（最多 2048 个）
- `page_count` — 匹配页面数
- `last_node` — 匹配终止节点（用于后续 lock/unlock）
- `matched_tokens` — 匹配的 token 数

### nxt_prefix_insert

```c
int32_t nxt_prefix_insert(NxtPrefixTree *tree,
                           const int32_t *token_ids, int32_t len,
                           const uint32_t *page_ids, int32_t page_count,
                           int32_t priority);
```

插入 token 序列 → page_ids 映射。自动节点分裂实现前缀共享。

**返回值**：已缓存的共享前缀长度，-1 表示分配失败。

### nxt_prefix_evict

```c
int32_t nxt_prefix_evict(NxtPrefixTree *tree, int32_t num_tokens);
```

淘汰约 `num_tokens` 个 token 的缓存页面。LRU 策略，优先淘汰最久未访问的叶子。若绑定了页池，自动释放对应页面。

**返回值**：实际淘汰的 token 数。

### nxt_prefix_lock / nxt_prefix_unlock

```c
void nxt_prefix_lock(NxtPrefixTree *tree, NxtPrefixNode *node);
void nxt_prefix_unlock(NxtPrefixTree *tree, NxtPrefixNode *node);
```

锁定/解锁节点及其祖先，防止被驱逐。请求开始使用缓存前缀时 lock，释放时 unlock。

### nxt_prefix_total_tokens / nxt_prefix_node_count / nxt_prefix_evictable_size

```c
int32_t nxt_prefix_total_tokens(const NxtPrefixTree *tree);
int32_t nxt_prefix_node_count(const NxtPrefixTree *tree);
int32_t nxt_prefix_evictable_size(const NxtPrefixTree *tree);
```

统计查询：总缓存 token 数、节点数、可淘汰 token 数。
