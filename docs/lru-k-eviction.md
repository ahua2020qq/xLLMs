# nxtLLM V0.1 LRU-K 驱逐算法设计文档

## 1. 概述

nxtLLM 采用 LRU-K（K=3）驱逐算法替代传统 LRU，通过保留每个页面的最近 K 次访问时间戳，利用第 K 次访问时间（而非最近一次）作为驱逐优先级依据，有效避免偶发访问导致的页面抖动。

## 2. 为什么选 LRU-K 而非 LRU

| 问题 | LRU | LRU-K (K=3) |
|------|-----|-------------|
| 偶发访问一次的页面立刻受保护 | ✅ 受保护 | ❌ 不受保护，需 3 次访问才"成熟" |
| 周期性扫描污染 | 严重 | 显著减轻 |
| 实现复杂度 | 低 | 中等 |
| 空间开销 | 1 个时间戳/页 | 3 个时间戳/页 |

**典型场景**：推理引擎顺序扫描 KV-cache 时，LRU 会将所有扫描过的页面标记为"最近使用"，驱逐真正热点的页面。LRU-K 通过要求多次访问才提升优先级，避免了这个问题。

## 3. 数据结构

### 3.1 LruKEntry — 页面访问历史

```c
#define LRU_K  3
#define LRU_TIMESTAMP_INVALID  0

typedef struct {
    uint64_t timestamps[LRU_K];  // 环形缓冲区，最新在 [0]
} LruKEntry;
```

- `timestamps[0]`：最近一次访问时间
- `timestamps[1]`：倒数第二次访问时间
- `timestamps[2]`：倒数第三次访问时间（第 K 次，即驱逐依据）
- `LRU_TIMESTAMP_INVALID (0)` 表示无效/未使用

### 3.2 LruKCache — 全局追踪器

```c
typedef struct {
    LruKEntry *entries;   // 数组，下标 = page_id
    size_t     capacity;  // 最大条目数
} LruKCache;
```

以 page_id 直接索引，O(1) 定位条目。

## 4. 核心操作

### 4.1 插入（lruk_insert）

```
输入: page_id, now_ts
  1. 清零所有时间戳
  2. timestamps[0] = now_ts
```

页面首次进入追踪器，仅记录一次访问。

### 4.2 访问记录（lruk_access）

```
输入: page_id, now_ts
  1. 若 timestamps[0] 无效 → 直接写入（首次访问）
  2. 否则：
     a. memmove: timestamps[1..K-1] ← timestamps[0..K-2]
     b. timestamps[0] = now_ts
     c. 最旧的时间戳被丢弃
```

**时间戳移位示例**：

```
访问前: [t3, t2, t1]   (t3 最新)
新访问 t4:
访问后: [t4, t3, t2]   (t1 被丢弃)
```

### 4.3 移除（lruk_remove）

```
输入: page_id
  1. memset 清零整个 LruKEntry
```

页面被驱逐后清除追踪记录。

### 4.4 Victim 选择（lruk_select_victim）

```
  1. 遍历所有 entries
  2. 跳过 k-th 时间戳无效的条目
  3. 选择 k-th 时间戳最小的页面（最久未被第 K 次访问）
  4. 返回 victim_id
```

**复杂度**：O(n)，n 为追踪器容量。V0.1 采用全表扫描，未来可优化为堆结构。

### 4.5 第 K 次时间戳（lruk_kth_timestamp）

```
  1. 统计有效时间戳数量 count
  2. 若 count == 0 → 返回 INVALID
  3. 返回 timestamps[count - 1]（最旧的有效时间戳）
```

对于未"成熟"的条目（访问不足 K 次），使用当前已有的最旧时间戳参与比较。

### 4.6 成熟度检查（lruk_is_mature）

```
  所有 K 个时间戳均非 INVALID → mature
  否则 → immature
```

**意义**：未成熟的页面更容易被驱逐（只有 1-2 次访问记录），防止偶发访问页面占据缓存。

## 5. 驱动集成

### 5.1 分配时记录访问

`nxt_page_alloc()` 中调用 `lruk_access(page_id, now)`，每次分配视为一次访问。

### 5.2 驱逐时选择 victim

`nxt_evict_lru_k()` 中：
1. 遍历 LRU-K 追踪器寻找 victim
2. 筛选条件：匹配 tier + type、ref_count == 0
3. 选择 k-th 时间戳最小的页面
4. 释放 victim 数据、从追踪器移除

### 5.3 时间戳来源

使用 `clock_gettime(CLOCK_MONOTONIC)` 获取微秒级单调时间戳，不受系统时间调整影响。

## 6. V0.1 已知限制

| 限制 | 影响 | 未来方向 |
|------|------|---------|
| Victim 选择 O(n) | 大规模页面池性能瓶颈 | 最小堆或分段树 |
| 无"成熟"优先驱逐 | 未成熟条目与成熟条目混合比较 | 优先驱逐未成熟页面 |
| 非线程安全 | 并发访问需外部加锁 | 引入细粒度锁或无锁结构 |
| 页面查找需遍历空闲链表 | 驱逐时定位页面效率低 | 维护 used-page 哈希表 |
