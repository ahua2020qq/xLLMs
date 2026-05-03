<!--
  nxtLLM — Next-Generation LLM Inference Engine
  Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
  SPDX-License-Identifier: Apache-2.0
  This header must not be removed. All derivative works must retain this notice.
-->

# nxtLLM V0.5 页面池与内存管理设计文档

## 1. 概述

页面池（Page Pool）是 nxtLLM 内存管理的核心子系统，负责在 GPU HBM、CPU DRAM、NVMe SSD 三个存储层级之间统一管理固定大小的内存页面。每个页面默认 64 KB，按用途分为三类，按存储位置分为三层。

## 2. 核心数据结构

### 2.1 NxtPage — 单个内存页面

```c
typedef struct NxtPage {
    uint32_t      page_id;      // 全局唯一 ID
    NxtPageType   type;         // DATA / INDEX / CONTROL
    NxtStorageTier tier;        // GPU / CPU / SSD
    int32_t       ref_count;    // 原子引用计数
    void         *data;         // 页面数据指针（延迟分配）
    size_t        size;         // 页面大小（字节）
    struct NxtPage *free_next;  // 空闲链表后继
    struct NxtPage *free_prev;  // 空闲链表前驱
} NxtPage;
```

**设计要点**：
- `data` 采用延迟分配策略，首次使用时才 `calloc`，避免初始化时一次性分配所有内存
- `ref_count` 使用 `__sync_fetch_and_add/sub` 实现原子操作，支持多线程安全引用计数
- 空闲链表指针嵌入结构体内，无需额外分配链表节点

### 2.2 NxtPageType — 页面类型

| 类型 | 值 | 用途 |
|------|-----|------|
| `PAGE_TYPE_DATA` | 0 | 模型权重、KV-cache 数据 |
| `PAGE_TYPE_INDEX` | 1 | 页表项、索引元数据 |
| `PAGE_TYPE_CONTROL` | 2 | 控制信息、请求元数据 |

### 2.3 NxtStorageTier — 存储层级

| 层级 | 值 | 介质 | 特点 |
|------|-----|------|------|
| `TIER_GPU` | 0 | HBM | 延迟最低，容量最小 |
| `TIER_CPU` | 1 | DRAM | 中等延迟，中等容量 |
| `TIER_SSD` | 2 | NVMe | 延迟最高，容量最大 |

### 2.4 NxtReqPageDir — 请求页面目录

```c
#define NXT_MAX_PAGES_PER_REQUEST 256

typedef struct {
    uint64_t  request_id;
    uint32_t  page_ids[NXT_MAX_PAGES_PER_REQUEST];
    uint32_t  page_count;
    uint32_t  priority;           // 0 = 最高优先级
    uint64_t  arrival_ts;         // 到达时间戳
    size_t    estimated_memory;   // 预估内存需求
} NxtReqPageDir;
```

每个推理请求关联一个页面目录，记录其占用的所有页面 ID，用于准入控制和资源回收。

### 2.5 NxtGlobalBufferPool — 全局缓冲池

```c
typedef struct {
    NxtPage  *free_heads[3][3];   // [tier][type] 空闲链表头
    NxtPage  *free_tails[3][3];   // [tier][type] 空闲链表尾
    uint32_t  free_counts[3][3];  // [tier][type] 空闲页面数
    NxtPageHash page_hash;        // V0.5: page_id → NxtPage* 哈希表
    LruKCache lru_k;              // LRU-K 追踪器
    size_t    tier_capacity[3];   // 各层容量（字节）
    size_t    tier_used[3];       // 各层已用量（字节）
    uint64_t  total_allocations;
    uint64_t  total_evictions;
    uint64_t  total_defrag_rounds;
} NxtGlobalBufferPool;
```

**空闲链表组织**：二维数组 `[TIER_COUNT][PAGE_TYPE_COUNT]`，共 9 条双向链表，实现 O(1) 的按层级+类型分配。

## 3. 核心算法

### 3.1 池初始化（nxt_pool_init）

```
输入: gpu_bytes, cpu_bytes, ssd_bytes, page_size, max_pages
  1. 计算每层可容纳的页面数: gpu_pages = gpu_bytes / page_size
  2. 分配 NxtPage 结构体数组
  3. 初始化 LRU-K 追踪器
  4. 按 [tier][type] 均匀分布页面到 9 条空闲链表
  5. 所有页面 data 指针置 NULL（延迟分配）
```

### 3.2 页面分配（nxt_page_alloc）

```
输入: type, preferred_tier
  1. 从 preferred_tier 向 TIER_GPU 方向遍历（降级分配）
  2. 取出空闲链表头部页面
  3. 延迟分配 data 缓冲区（calloc）
  4. 设置 ref_count = 1
  5. 更新 tier_used 和 total_allocations
  6. 记录 LRU-K 访问时间戳
  7. 若分配在非首选层级，标记 tier 为 preferred_tier（预取）
  8. 若所有层级无空闲页面 → 触发 LRU-K 驱逐后重试
```

**降级策略**：优先从快速层级分配，若无空闲则依次向慢速层级回退。

### 3.3 页面释放（nxt_page_free）

```
输入: page
  1. 检查 ref_count == 0（由 ref_dec 触发）
  2. 更新 tier_used
  3. 将页面追加回对应 [tier][type] 空闲链表尾部
```

### 3.4 引用计数

```
nxt_page_ref_inc:  __sync_fetch_and_add(&ref_count, 1)
nxt_page_ref_dec:  __sync_fetch_and_sub(&ref_count, 1)
                   若 prev == 1 → 调用 nxt_page_free
```

**生命周期**：alloc 时 ref_count=1 → 用户可 inc/dec → ref_count 归零时自动释放。

### 3.5 准入控制（nxt_admission_check）

```
输入: req (NxtReqPageDir)
  1. 计算所有层级的空闲内存总量
  2. 若 req->estimated_memory <= free_mem → 通过
  3. 否则循环驱逐页面直到有足够空间
  4. 若无法驱逐出足够空间 → 拒绝
```

### 3.6 碎片整理（nxt_defrag_background）— V0.5 已实现

按 tier×type 分组紧凑整理，消除页面间隙：

1. 按 tier×type 收集页面（最多 4096 页/组）
2. 按 page_id 排序（qsort，O(n log n)）
3. 双指针交换：`left` 找首个空闲页，`right` 找末尾已用页，交换 data 指针和 ref_count
4. 重建空闲链表：重置 free_heads/free_tails，追加所有 ref_count==0 的页面
5. 更新 free_counts

**设计要点**：
- 指针交换（O(1)/次）代替 memmove（O(page_size)/次），高效无数据拷贝
- 按 tier 隔离，不会跨层移动数据
- 需在推理间隙执行（batch 之间），暂停 alloc/free

### 3.7 哈希索引（V0.5 新增）

在 `NxtGlobalBufferPool` 中嵌入 `NxtPageHash`，实现 page_id → NxtPage* 的 O(1) 查找：

```c
typedef struct {
    uint32_t key;    // page_id
    NxtPage *value;  // 页面指针
} NxtHashEntry;

typedef struct {
    NxtHashEntry *entries;
    size_t        capacity;  // 2 的幂次
    size_t        size;
} NxtPageHash;
```

**API**：
- `nxt_hash_init()` — 初始化，容量为 next_pow2(total_pages × 2)
- `nxt_hash_insert()` — 插入，线性探测冲突解决
- `nxt_hash_lookup()` — O(1) 查找
- `nxt_hash_remove()` — 哨兵值标记删除（NXT_HASH_EMPTY = 0xFFFFFFFF）

**集成点**：
- `nxt_pool_init()` — 初始化哈希表，插入所有页面
- `nxt_evict_lru_k()` — 用 `nxt_hash_lookup()` 替代链表遍历
- `nxt_defrag_background()` — 整理后更新哈希表

**性能提升**：
- 驱逐操作从 O(n²) 降至 O(k)（k = 候选页面数）
- 内存开销约 24 bytes/page

## 4. 线程安全模型

| 操作 | 线程安全 | 机制 |
|------|---------|------|
| ref_count 增减 | ✅ | `__sync_fetch_and_add/sub` |
| 空闲链表操作 | ❌ | V0.1 无锁保护，需外部同步 |
| LRU-K 时间戳 | ❌ | V0.1 无锁保护 |

**V0.1 限制**：仅引用计数为原子操作，空闲链表和 LRU-K 追踪器需要外部互斥锁保护。

## 6. 已知限制

| 限制 | 影响 | 修复方向 |
|------|------|---------|
| ~~Victim 选择 O(n) 全表扫描~~ | ~~V0.5 已通过哈希索引解决~~ | — |
| ~~碎片整理仅骨架~~ | ~~V0.5 已实现完整碎片整理~~ | — |
| 空闲链表非线程安全 | 并发分配需外部锁 | 细粒度 per-tier 锁或无锁链表 |

## 5. 内存布局

```
NxtPage 结构体数组（连续内存）
┌────┬────┬────┬────┬────┬─────────────┐
│ P0 │ P1 │ P2 │ P3 │ P4 │ ...         │
└────┴────┴────┴────┴────┴─────────────┘
  │         空闲链表通过 free_next/prev 串联
  │         data 指针指向独立分配的缓冲区
  ▼
┌──────────────┐
│ data buffer  │ ← calloc(page_size, 1)
│  (64 KB)     │
└──────────────┘
```
