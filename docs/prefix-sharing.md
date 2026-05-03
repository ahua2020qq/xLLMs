<!--
  nxtLLM — Next-Generation LLM Inference Engine
  Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
  SPDX-License-Identifier: Apache-2.0
  This header must not be removed. All derivative works must retain this notice.
-->

# nxtLLM V0.4 前缀共享基数树设计文档

## 1. 概述

前缀共享基数树（Prefix Sharing Radix Tree）是 nxtLLM V0.4 引入的核心模块，灵感来自 [SGLang RadixCache](https://github.com/sgl-project/sglang)。通过基数树（Patricia Trie）索引 token 序列到 KV-cache 页面的映射，实现跨请求的前缀复用，避免重复 prefill 计算。

## 2. 设计动机

LLM 推理服务中，多个请求常共享相同前缀（系统提示词、few-shot 示例等）。传统做法为每个请求独立计算并缓存 KV，导致：

- 重复计算相同前缀的 KV-cache → 浪费算力
- 相同 KV 数据在显存中多次存储 → 浪费内存

**基数树方案**：将 token 序列组织为压缩前缀树，共享前缀的请求复用祖先节点的 KV-cache 页面。

## 3. 数据结构

### 3.1 NxtPrefixNode — 基数树节点

```c
typedef struct NxtPrefixNode {
    int32_t    *token_ids;       // 边标签：token 序列片段
    int32_t     token_count;     // token 数量
    uint32_t   *page_ids;        // KV-cache 页面索引
    int32_t     page_count;      // 页面数量
    struct NxtPrefixNode *parent;
    struct NxtPrefixNode **children;  // 动态子节点数组
    int32_t     child_count;
    int32_t     child_capacity;
    int32_t     lock_ref;        // 引用锁定计数（防驱逐）
    int32_t     hit_count;       // 命中计数
    int32_t     priority;        // 优先级
    uint64_t    last_access_time; // LRU 时间戳
    uint64_t    creation_time;
    bool        evicted;
} NxtPrefixNode;
```

### 3.2 NxtPrefixTree — 前缀树

```c
typedef struct {
    NxtPrefixNode       *root;
    NxtGlobalBufferPool *pool;           // 绑定的页池
    int32_t              total_nodes;
    int32_t              evictable_size;  // 可淘汰 token 数
    int32_t              protected_size;  // 锁定 token 数
} NxtPrefixTree;
```

### 3.3 NxtMatchResult — 匹配结果

```c
typedef struct {
    uint32_t       page_ids[2048];  // 匹配到的页面 ID
    int32_t        page_count;
    NxtPrefixNode *last_node;       // 匹配终止节点
    int32_t        matched_tokens;  // 匹配 token 数
} NxtMatchResult;
```

## 4. 核心算法

### 4.1 前缀匹配 — nxt_prefix_match() → O(L)

```
输入: token_ids[], len
  1. 从 root 开始，沿树向下匹配
  2. 每步查找首 token 匹配的子节点
  3. 计算公共前缀长度 common_prefix_len()
  4. 若部分匹配 → split_node() 分裂节点
  5. 收集沿途节点的 page_ids
  6. 更新 last_access_time 和 hit_count
```

### 4.2 插入 — nxt_prefix_insert() → O(L)

```
输入: token_ids[], page_ids[], priority
  1. 沿树匹配已有前缀
  2. 遇到部分匹配 → split_node() 分裂
  3. 为未匹配的剩余 token 创建新节点
  4. 新节点挂载到匹配路径末端
  5. 返回共享前缀长度
```

### 4.3 节点分裂 — split_node()

```
输入: child 节点, split_len
  1. 创建新节点持有 shared prefix [0..split_len]
  2. 原子节点截断为 [split_len..end]
  3. 新节点替换原子节点在父节点中的位置
  4. 原子节点成为新节点的子节点
```

### 4.4 淘汰 — nxt_prefix_evict()

```
输入: num_tokens
  1. DFS 收集所有可淘汰叶子（lock_ref == 0, 无活跃子节点）
  2. 选择 last_access_time 最小的叶子（LRU）
  3. 释放叶子节点的 page_ids 到页池
  4. 从父节点移除叶子，销毁节点
  5. 重复直到淘汰足够 token 或无可淘汰节点
```

### 4.5 引用锁定 — lock/unlock

```
nxt_prefix_lock: 从 node 向上遍历到 root，lock_ref++
  - 首次锁定时，evictable_size → protected_size

nxt_prefix_unlock: 从 node 向上遍历到 root，lock_ref--
  - 归零时，protected_size → evictable_size
```

## 5. 与三层页池的集成

```
请求 "Hello world" 和 "Hello AI" 共享前缀 "Hello"

基数树:
  root → "Hello" → " world"
                  └→ " AI"

"Hello" 节点的 page_ids 指向 GPU HBM 中的 KV-cache 页面
两个请求复用这些页面，各自只需计算后缀的 KV
```

**淘汰联动**：
- 基数树淘汰叶子节点时，调用页池释放对应页面
- 页池的 LRU-K 驱逐触发时，可通知基数树标记对应节点为 evicted
- 未来支持：淘汰时将 GPU 页迁移到 CPU/SSD，而非直接释放

## 6. 测试覆盖

| 测试 | 覆盖内容 |
|------|---------|
| test_tree_init_destroy | 初始化/销毁生命周期 |
| test_insert_single | 单序列插入 |
| test_match_full | 完整前缀匹配 |
| test_match_partial | 部分前缀匹配 |
| test_shared_prefix | 跨序列共享前缀 |
| test_empty_query | 空查询边界条件 |
| test_lock_unlock | 引用锁定防驱逐 |
| test_eviction | LRU 淘汰 |
| test_divergent_branches | 分叉路径正确性 |
| test_statistics | 统计信息准确性 |

## 7. 已知限制与未来方向

| 限制 | 未来方向 |
|------|---------|
| 淘汰使用简单线性扫描找 LRU 叶子 | 最小堆优化为 O(log N) |
| 无线程安全保护 | 调度器层面串行化或细粒度锁 |
| page_id 到 NxtPage 反查缺失 | 维护 page-id → NxtPage 哈希表 |
| 无 SHA256 链式哈希 | 可选：跨请求去重和完整性校验 |
| 淘汰直接操作 `tier_used[0]` 而非走页池释放 | 通过 page_id 哈希表调用 `nxt_page_ref_dec()` |
| DFS/BFS 使用固定栈/队列（1024） | 超大树的深度可能溢出 |
