# SGLang RadixCache 设计与 nxtLLM 集成分析

> 来源：`sglang/python/sglang/srt/mem_cache/radix_cache.py`
> 分析日期：2026-05-03
> 目标：将 RadixCache 的前缀共享思想融入 nxtLLM 三级页池

---

## 1. 设计思想

SGLang 的 RadixCache 是一个**基数树（Radix Tree / Patricia Trie）**驱动的 KV-cache 管理器。
核心思想：**多个请求共享相同前缀的 KV-cache，避免重复计算和存储**。

### 1.1 基数树作为 KV-cache 索引

- 树的每条路径代表一个 token 序列
- 每个节点存储该前缀对应的 KV-cache 索引列表（`value: Tensor`）
- 共享前缀的请求自动共享祖先节点中缓存的 KV-cache
- 通过 `match_prefix()` 查找最长匹配前缀，避免重复 prefill

### 1.2 关键设计决策

| 维度 | SGLang 选择 | 设计理由 |
|------|------------|---------|
| 数据结构 | Radix Tree (压缩前缀树) | O(L) 匹配，L=序列长度；内存紧凑（合并单分支路径） |
| 节点粒度 | 可配置 page_size | `page_size=1` 逐 token 匹配；`page_size>1` 按页对齐 |
| 淘汰策略 | 可插拔策略模式 | 支持 LRU/LFU/FIFO/SLRU/Priority 等多种策略 |
| 引用保护 | `lock_ref` 计数器 | 防止正在使用的节点被淘汰 |
| 命名空间隔离 | `extra_key` (LoRA ID / cache_salt) | 不同 LoRA 的 KV-cache 物理隔离 |
| Hash 标识 | SHA256 链式哈希 | 位置感知的缓存块标识（支持去重/事件溯源） |

---

## 2. 数据结构

### 2.1 类关系图

```
RadixCache (基数缓存管理器)
 ├── root_node: TreeNode (根节点，哨兵)
 ├── evictable_leaves: set[TreeNode] (可淘汰叶子集合)
 ├── eviction_strategy: EvictionStrategy (策略对象)
 └── page_size: int

TreeNode (基数树节点)
 ├── children: dict[key → TreeNode] (子节点映射)
 ├── parent: TreeNode
 ├── key: RadixKey (该节点对应的 token 序列片段)
 ├── value: Tensor (KV-cache 索引)
 ├── lock_ref: int (引用锁定计数)
 ├── host_ref_counter: int (主机存储引用计数)
 ├── last_access_time: float (LRU 时间戳)
 ├── hit_count: int (命中计数)
 ├── priority: int (优先级)
 ├── hash_value: List[str] (SHA256 链式哈希)
 └── host_value: Tensor (备份到 CPU 的 KV-cache)
```

### 2.2 RadixKey

```python
class RadixKey:
    token_ids: List[int]     # token 序列
    extra_key: Optional[str] # 命名空间标签 (LoRA ID / cache_salt)
    is_bigram: bool          # 是否为 EAGLE bigram 键
```

### 2.3 关键不变量

- **根节点不可淘汰**：`lock_ref = 1` 且不在 `evictable_leaves` 中
- **叶子节点淘汰规则**：只有所有子节点已淘汰（`evicted = True`）且 `lock_ref == 0` 的节点才可淘汰
- **split 操作保持哈希链**：节点分裂时，哈希值按页切分，父子各持一部分

---

## 3. 核心算法复杂度

### 3.1 前缀匹配 `match_prefix()` → O(L)

```
L = 输入 key 的长度（token 数量）

最坏情况：L 次逐 token 比较 + 1 次节点分裂
- 遍历路径：O(L) 步，每步查 children dict (O(1))
- key_match_fn：O(min(len(node.key), len(remaining_key)))
- 分裂操作 _split_node：O(1) 字典操作 + O(len(child.key)) 数据切片

总计 O(L)，实际常数很小
```

### 3.2 插入 `insert()` → O(L)

```
- 沿树遍历匹配前缀：O(L)
- 可能触发节点分裂：O(1) 字典更新 + O(剩余 key 长度) 数据切片
- 创建新节点：O(1) 字典插入
- 更新 evictable_leaves：O(1) 集合操作
```

### 3.3 淘汰 `evict()` → O(E log N)

```
E = 要淘汰的 token 数量
N = 可淘汰叶子数量

- 构建最小堆：O(N)（heapify）
- 每次淘汰：O(log N) pop + O(1) delete_leaf + 最多 O(log N) push（父节点变成叶子）
- 总计 O(N + E log N)
```

### 3.4 锁定/解锁 `inc_lock_ref` / `dec_lock_ref` → O(depth)

```
depth = 节点深度（≤ L）
沿父链向上遍历：O(depth) ≤ O(L)
```

### 3.5 空间复杂度

```
O(T)  where T = 所有缓存 token 总数（去重后）
最坏情况（无前缀共享）：每个 token 一个节点 → O(T)
最佳情况（全共享）：O(1) 个节点
```

---

## 4. 与 nxtLLM 三级页池的结合点

nxtLLM 现有的三级页池 (`NxtGlobalBufferPool`) 提供 GPU/CPU/SSD 三级存储，缺少**前缀感知**能力。
RadixCache 的基数树可以在 nxtLLM 中作为**前缀索引层**，架设在页池之上。

### 4.1 架构融合方案

```
┌─────────────────────────────────────────────────┐
│                  Request Layer                    │
│    [Req 1: "Hello world"]  [Req 2: "Hello AI"]  │
└─────────┬──────────────────────┬────────────────┘
          │                      │
          ▼                      ▼
┌─────────────────────────────────────────────────┐
│            Prefix Sharing (NEW ─ 基数树)         │
│  ┌─────────────────────────────────────────┐    │
│  │  root → "Hel" → "lo" → " " → "world"   │    │
│  │                        └→ "AI"          │    │
│  └─────────────────────────────────────────┘    │
│  功能: match_prefix / insert / evict / lock     │
└─────────┬───────────────────────────────────────┘
          │ 映射 token_ids → page_ids
          ▼
┌─────────────────────────────────────────────────┐
│            NxtGlobalBufferPool (三级页池)         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ GPU/HBM  │  │ CPU/DRAM │  │ SSD/NVMe │      │
│  │  (热页)  │  │  (温页)  │  │  (冷页)  │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│  功能: page_alloc / page_free / evict_lru_k     │
└─────────────────────────────────────────────────┘
```

### 4.2 具体结合点

| 结合点 | SGLang 原实现 | nxtLLM 适配 |
|--------|-------------|------------|
| 节点值 `value` | `Tensor[int64]` KV 索引 | `uint32_t[]` page_ids |
| 淘汰回池 | `token_to_kv_pool_allocator.free()` | `nxt_page_ref_dec()` → `nxt_page_free()` |
| 锁定保护 | `lock_ref` 计数 | 复用 `NxtPage.ref_count` 机制 |
| 页对齐 | `page_size` 参数 | 复用 `DEFAULT_PAGE_SIZE` (64KB) |
| 三级淘汰 | 无 | 淘汰节点对应的 GPU 页迁移到 CPU/SSD，而非直接释放 |
| 哈希标识 | SHA256 链式哈希 | 可选：用于跨请求去重和断言 |

### 4.3 新增模块：prefix_sharing

```
nxtLLM/include/prefix_sharing.h    ─ 基数树 API 声明
nxtLLM/src/core/prefix_sharing.c   ─ 基数树实现
nxtLLM/tests/test_prefix_sharing.c ─ 单元测试
```

核心接口：

```c
// 前缀树生命周期
void nxt_prefix_tree_init(NxtPrefixTree *tree, NxtGlobalBufferPool *pool);
void nxt_prefix_tree_destroy(NxtPrefixTree *tree);

// 前缀匹配：返回最长匹配的 page_id 列表
NxtMatchResult nxt_prefix_match(NxtPrefixTree *tree, const int32_t *token_ids, int32_t len);

// 插入新序列
int32_t nxt_prefix_insert(NxtPrefixTree *tree, const int32_t *token_ids, int32_t len,
                          const uint32_t *page_ids, int32_t priority);

// 淘汰
int32_t nxt_prefix_evict(NxtPrefixTree *tree, int32_t num_tokens);

// 引用锁定（防淘汰）
void nxt_prefix_lock(NxtPrefixTree *tree, NxtPrefixNode *node);
void nxt_prefix_unlock(NxtPrefixTree *tree, NxtPrefixNode *node);
```

---

## 5. 实现注意事项

1. **线程安全**：基数树操作需在调度器层面串行化，或在关键路径加锁。SGLang 原实现依赖 Python GIL。
2. **内存一致性**：节点中引用的 `page_ids` 必须在页池中保持有效（ref_count > 0）。淘汰节点前必须释放所有 page 引用。
3. **page_size 对齐**：从 nxtLLM 的 `DEFAULT_PAGE_SIZE` (64KB) 计算 tokens_per_page，确保插入和匹配按页边界对齐。
4. **淘汰策略**：第一版使用 LRU（按 `last_access_time`），后续可扩展为 LRU-K（复用已有 `LruKCache` 基础设施）。
5. **多级缓存感知**：淘汰时优先将 GPU 页迁移到 CPU/SSD，而非直接丢弃。这是 SGLang 原实现不具备的能力（SGLang 直接 free）。
