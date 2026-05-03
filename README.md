# nxtLLM V0.4 — Next-Generation LLM Inference Engine

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Conduct-Contributor%20Covenant%202.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Security](https://img.shields.io/badge/security-report%20vulnerability-orange.svg)](SECURITY.md)

nxtLLM is a high-performance LLM inference engine with a multi-tier memory management system.
It implements LRU-K page eviction, tier-aware page allocation (GPU HBM → CPU DRAM → NVMe SSD),
prefix-sharing radix tree for KV-cache reuse, and admission control for concurrent inference requests.

> **本项目由 [山野小娃](https://github.com/ahua2020qq)、[DeepSeek](https://github.com/deepseek-ai) 和 [豆包](https://www.doubao.com) 共同设计。**
>
> 感谢以下开源项目及其社区为 LLM 推理领域做出的卓越贡献：
>
> - **[vLLM](https://github.com/vllm-project/vllm)** — PagedAttention 与连续批处理启发
> - **[Llama.cpp](https://github.com/ggerganov/llama.cpp)** — 轻量化 CPU 推理与 GGUF 量化
> - **[SGLang](https://github.com/sgl-project/sglang)** — RadixCache 前缀共享设计（已集成）
> - **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — 通用注意力内核（未来参考）
>
> nxtLLM 的成长离不开这些优秀项目的智慧。衷心希望大模型推理能够更快进入 v1.0+ 时代 —— v0.20.X 的漫长过渡，着实让人有些尴尬。

## Project Goals

- **Efficient memory utilization**: Three-tier buffer pool spanning GPU, CPU, and SSD storage
- **Intelligent page eviction**: LRU-K (K=3) algorithm prevents thrashing of recently accessed pages
- **Prefix-aware KV-cache sharing**: Radix tree indexes token sequences for cross-request cache reuse
- **Request-aware scheduling**: Per-request page directories with admission control
- **Background defragmentation**: Coalesces free pages to reduce fragmentation
- **Modular architecture**: Clean separation of page pool, LRU-K, prefix tree, and memory manager layers
- **GPU-accelerated operators**: Paged attention, activation kernels (SiLU/GELU), and KV-cache management

## Architecture (5-Layer Design)

```
+═══════════════════════════════════════════════════════════════+
║                    LAYER 5: INFERENCE ENGINE                  ║
║   Model runner, tokenizer, KV-cache manager, sampler          ║
║   (inference_runner.c, transformer_block.c, tokenizer.c)      ║
+═══════════════════════════════════════════════════════════════+
                              │
                              ▼
+═══════════════════════════════════════════════════════════════+
║                 LAYER 4: REQUEST SCHEDULER                    ║
║   Admission control, request queue, priority management       ║
║   (prefix_sharing.c — radix tree KV-cache index)              ║
+═══════════════════════════════════════════════════════════════+
                              │
                              ▼
+═══════════════════════════════════════════════════════════════+
║                 LAYER 3: PAGE ALLOCATOR                       ║
║   Page allocation, free list management, defragmentation      ║
║   (memory_manager.c)                                          ║
+═══════════════════════════════════════════════════════════════+
                              │
                              ▼
+═══════════════════════════════════════════════════════════════+
║                 LAYER 2: EVICTION MANAGER                     ║
║   LRU-K (K=3) eviction, timestamp tracking, victim selection  ║
║   (lru_k.h — integrated into memory_manager.c)                ║
+═══════════════════════════════════════════════════════════════+
                              │
                              ▼
+═══════════════════════════════════════════════════════════════+
║                 LAYER 1: TIERED BUFFER POOL                   ║
║   GPU HBM (fast) → CPU DRAM (medium) → NVMe SSD (slow)       ║
║   (page_pool.h — NxtGlobalBufferPool)                         ║
+═══════════════════════════════════════════════════════════════+
```

## What's New

### v0.4 — Prefix Sharing Radix Tree + FlashInfer Adapter

SGLang RadixCache 启发的前缀共享基数树 + FlashInfer 风格注意力适配器：

| 功能 | 文件 | 说明 |
|------|------|------|
| **Radix Tree API** | `include/prefix_sharing.h` | 前缀匹配、插入、淘汰、锁定接口 |
| **Radix Tree 实现** | `src/core/prefix_sharing.c` | 基数树节点分裂/合并、LRU 淘汰、与三级页池集成 |
| **FlashInfer Adapter** | `operators/flashinfer_adapter.cu` | Batch decode 内核，cp.async 流水线、GQA-aware 线程布局 |
| **Bench Test** | `tests/test_attention_bench.c` | v1 vs FlashInfer-style A/B 性能对比 |
| **设计分析** | `docs/design/` | SGLang RadixCache + FlashInfer 架构分析 |

核心能力：
- `nxt_prefix_match()` — O(L) 最长前缀匹配，返回缓存的 page_id 列表
- `nxt_prefix_insert()` — O(L) 插入，自动节点分裂实现前缀共享
- `nxt_prefix_evict()` — LRU 叶节点淘汰，与页池联动释放内存
- `nxt_prefix_lock/unlock()` — 请求级引用保护，防止活跃前缀被驱逐
- `nxt_paged_attention_flash()` — FlashInfer 风格 batch decode（SM80+，需 `-DUSE_FLASHINFER=ON`）
- **TensorRT-LLM 适配层** — Stub 模式编译（无需 TRT 依赖），渐进式集成量化/融合/Graph/Batching（`-DUSE_TENSORRT=ON`）

### v0.3 — GPT-2 Inference Engine + CUDA Optional

- **TransformerBlock**：LayerNorm → QKV → Paged Attention → FFN → GELU 完整前向传播
- **BPE Tokenizer**：256 词表，支持 encode/decode
- **Decode Loop**：自回归解码，集成 KV-cache 和 LM Head 投影
- **run_gpt2 示例**：自包含的 GPT-2 推理 demo
- **CUDA 可选**：`-DUSE_CUDA=ON` 启用，默认纯 CPU 构建

### v0.2 — GPU Operators

| Operator | File | Description |
|---|---|---|
| **Paged Attention** | `operators/page_attention.cu` | Multi-head attention over paged KV-cache blocks. Supports GQA/MQA. |
| **SiLU & Mul** | `operators/activation_kernels.cu` | Fused SiLU-gate + element-wise multiply. |
| **GELU & Mul** | `operators/activation_kernels.cu` | Fused GELU-gate + multiply (none/tanh approximations). |
| **Element-wise Activations** | `operators/activation_kernels.cu` | Standalone SiLU and GELU kernels. |
| **KV-Cache Config** | `operators/cache.py` | Python dataclass for cache dtype, prefix caching, multi-tier blocks. |

All GPU kernels are exposed through a unified C API in `include/operator_api.h`.

## Core Data Structures

### NxtPage
A single memory page (default 64 KB) belonging to a type (DATA / INDEX / CONTROL)
and residing on a storage tier (GPU / CPU / SSD). Pages are tracked in doubly-linked
free lists and inherit LRU-K timestamps from the global `LruKCache`.

### NxtReqPageDir
Per-request page directory maps an inference request to its allocated pages.
Up to 256 pages per request; includes scheduling priority, estimated memory,
and arrival timestamp for admission control.

### NxtGlobalBufferPool
The global buffer pool maintains per-tier, per-type free page lists, an LRU-K
tracker, memory budget caps, and global allocation/eviction/defrag statistics.

### LruKCache (K=3)
Tracks the 3 most recent access timestamps for each page. Victim selection uses
the oldest (k-th) timestamp, which captures long-term access patterns and avoids
evicting pages with only a single recent access burst.

### NxtPrefixTree (Radix Tree)
Token-sequence radix tree indexing KV-cache pages for prefix sharing across requests.
Each node stores a token fragment, associated page_ids, and a lock_ref counter for
request-level eviction protection.

## Quick Start

### Prerequisites

- CMake >= 3.16
- C11/C++17-compliant compiler (GCC >= 9, Clang >= 10)
- *(Optional)* CUDA Toolkit >= 11.8, NVIDIA driver >= 525 (SM 75+)

### Build & Run

```sh
git clone <repo-url> && cd nxtLLM

# CPU-only build (default)
cmake -B build -S .
cmake --build build

# With CUDA support
cmake -B build -S . -DUSE_CUDA=ON
cmake --build build

# With CUDA + FlashInfer adapter (SM80+)
cmake -B build -S . -DUSE_CUDA=ON -DUSE_FLASHINFER=ON
cmake --build build

# With CUDA + TensorRT adapter (stub mode, no TRT dependency)
cmake -B build -S . -DUSE_CUDA=ON -DUSE_TENSORRT=ON
cmake --build build

# Run tests
cd build && ctest --output-on-failure

# Run GPT-2 inference demo (random weights)
./run_gpt2
```

### Basic Usage

```c
#include "page_pool.h"
#include "prefix_sharing.h"

int main() {
    /* Initialize tiered buffer pool */
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  /* GPU */ 4ULL << 30,    // 4 GiB
                  /* CPU */ 16ULL << 30,   // 16 GiB
                  /* SSD */ 64ULL << 30,   // 64 GiB
                  /* page_size */ 64 << 10,
                  /* max_pages */ 100000);

    /* Initialize prefix-sharing radix tree */
    NxtPrefixTree tree;
    nxt_prefix_tree_init(&tree, &pool);

    /* Insert a cached prefix */
    int32_t tokens[] = {1, 2, 3, 4, 5};
    uint32_t pages[] = {100, 101, 102, 103, 104};
    nxt_prefix_insert(&tree, tokens, 5, pages, 5, 0);

    /* Match longest cached prefix for a new request */
    NxtMatchResult match = nxt_prefix_match(&tree, tokens, 5);
    // match.page_ids — reusable KV-cache pages
    // match.matched_tokens — number of tokens to skip in prefill

    nxt_prefix_tree_destroy(&tree);
    nxt_pool_destroy(&pool);
    return 0;
}
```

## Directory Layout

```
nxtLLM/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── include/
│   ├── operator_api.h          # GPU operator C API (v0.2)
│   ├── prefix_sharing.h       # Radix tree API (v0.4)
│   └── tensorrt_adapter.h    # TensorRT-LLM stub adapter (v0.4)
├── operators/
│   ├── page_attention.cu       # Paged attention kernel (v0.2)
│   ├── activation_kernels.cu   # SiLU/GELU activation kernels (v0.2)
│   ├── flashinfer_adapter.cu   # FlashInfer-style adapter (v0.4)
│   └── cache.py                # KV-cache configuration (v0.2)
├── src/
│   ├── include/
│   │   ├── page_pool.h
│   │   ├── lru_k.h
│   │   ├── transformer_block.h # (v0.3)
│   │   ├── tokenizer.h         # (v0.3)
│   │   ├── decoder.h           # (v0.3)
│   │   └── weight_loader.h     # (v0.3)
│   ├── core/
│   │   ├── memory_manager.c
│   │   ├── lru_k.c
│   │   └── prefix_sharing.c    # Radix tree implementation (v0.4)
│   └── inference/
│       ├── transformer_block.c  # (v0.3)
│       ├── tokenizer.c          # (v0.3)
│       ├── decode_loop.c        # (v0.3)
│       └── weight_loader.c      # (v0.3)
│   └── adapter/
│       └── tensorrt_adapter.c  # TensorRT stub (v0.4)
├── examples/
│   └── run_gpt2.c              # GPT-2 inference demo (v0.3)
├── tests/
│   ├── test_page_pool.c
│   ├── test_attention.c        # (v0.2)
│   ├── test_prefix_sharing.c  # (v0.4)
│   ├── test_attention_bench.c # FlashInfer bench (v0.4)
│   └── test_tensorrt_integration.c # TensorRT stub tests (v0.4)
├── scripts/
│   └── convert_gpt2_weights.py # (v0.3)
└── docs/
    ├── architecture.md
    ├── memory-manager.md
    ├── lru-k-eviction.md
    ├── api-reference.md
    └── design/
        ├── sglang_radixcache_analysis.md  # (v0.4)
        ├── flashinfer_analysis.md         # (v0.4)
        └── tensorrt_llm_analysis.md      # (v0.4)
```

## Roadmap

### v0.1 ✅
- [x] Three-tier buffer pool (GPU/CPU/SSD)
- [x] LRU-K (K=3) page eviction
- [x] Page allocation and reference counting
- [x] Basic test suite

### v0.2 ✅
- [x] Paged attention CUDA kernel
- [x] Activation kernels (SiLU, GELU)
- [x] KV-cache configuration module
- [x] Unified operator C API

### v0.3 ✅
- [x] GPT-2 inference engine (TransformerBlock, tokenizer, decode loop)
- [x] CUDA optional build (`-DUSE_CUDA=ON`)
- [x] run_gpt2 example with random weights

### v0.4 ✅ (current)
- [x] Prefix-sharing radix tree (SGLang RadixCache inspired)
- [x] Cross-request KV-cache reuse via `nxt_prefix_match/insert`
- [x] LRU eviction with request-level lock protection
- [x] FlashInfer-style batch decode adapter (`-DUSE_FLASHINFER=ON`)
- [x] TensorRT-LLM stub adapter with 5-phase integration plan (`-DUSE_TENSORRT=ON`)
- [x] Attention benchmark tool (v1 vs FlashInfer A/B comparison)
- [x] 10 TensorRT integration tests (stub mode, 10/10 pass)

### v0.5 (planned)
- [ ] GPTQ / AWQ quantization kernels
- [ ] paged_kv_t abstraction + multi-stage async pipeline
- [ ] CUDA graph capture for decode phase
- [ ] DecodePlan scheduler for partition-KV

## Project Analysis

### Code Statistics

| Category | Files | Lines | Key Content |
|----------|-------|-------|-------------|
| C source (.c) | 14 | ~3700 | Core logic, inference, adapters, tests |
| C headers (.h) | 9 | ~800 | Data structures, API declarations |
| CUDA (.cu) | 3 | ~1100 | GPU kernels (v1 + FlashInfer) |
| Python (.py) | 2 | ~280 | KV-cache config, weight conversion |
| Docs (.md) | 13 | ~1800 | Design docs, README, community files |
| Other | 10 | ~1300 | CMake, .gitignore, templates |
| **Total** | **51** | **~8900** | |

### Technical Highlights

**Three-Tier Storage** — 9 doubly-linked free lists `[3 tiers × 3 types]` with O(1) allocation. Atomic reference counting via `__sync_fetch_and_add/sub`. Lazy data buffer allocation on first use.

**LRU-K (K=3) Eviction** — Ring buffer tracks 3 most recent access timestamps per page. k-th timestamp used as eviction criterion. Immature entries (<K accesses) are easier to evict.

**Prefix-Sharing Radix Tree** — Patricia Trie indexes token sequences with automatic node splitting. `lock_ref` chain protects active requests' KV-cache. Integrated with page pool for LRU eviction.

**FlashInfer-Style Adapter** — `cp.async` multi-stage pipeline hides memory latency. Vectorized loads `vec_size = 16/sizeof(T)` maximize bandwidth. GQA-aware thread layout with templated `GROUP_SIZE`. Same signature as v1 kernel for A/B comparison.

**TensorRT-LLM Adapter** — Stub-mode compilation without TensorRT dependency. 5-phase integration plan: stub → quantization → layer fusion → CUDA Graph → in-flight batching. 10 integration tests pass in stub mode.

### Test Coverage

| Suite | Cases | Scope |
|-------|-------|-------|
| test_page_pool | 6 | Pool init, alloc/free, refcount, admission, LRU-K |
| test_prefix_sharing | 10 | Tree lifecycle, insert, match, split, lock, evict, diverge, stats |
| test_attention | 9 | Operator signatures, arg layout, smoke test |
| test_attention_bench | 4+ | FlashInfer null launch, correctness, benchmark |
| test_tensorrt_integration | 10 | TRT stub lifecycle, builder, engine, inference, quant, graph, batch |
| **Total** | **39+** | |

### Known Limitations

| Issue | Severity | Impact |
|-------|----------|--------|
| Defragmentation skeleton only | Low | Fragmentation over long runs |
| Victim selection O(n) scan | Medium | Bottleneck at >10K pages |
| No page_id → NxtPage hash | Medium | Slow eviction and prefix reclaim |
| Free lists not thread-safe | Medium | External lock needed for concurrency |
| weight_loader hardcoded n_layer=12 | Medium | Memory leak for non-12-layer models |
| No integration / numerical accuracy tests | Medium | End-to-end correctness unverified |

### Competitive Positioning

| Dimension | nxtLLM | vLLM | SGLang | Llama.cpp |
|-----------|--------|------|--------|-----------|
| Three-tier storage | ✅ GPU/CPU/SSD | GPU/CPU | GPU/CPU | CPU/Disk |
| Prefix sharing | ✅ Radix tree | ❌ | ✅ RadixCache | ❌ |
| LRU-K eviction | ✅ K=3 | LRU | Pluggable | LRU |
| GPU operators | ✅ Optional | ✅ Full | ✅ Full | ❌ CPU only |
| Inference engine | 🔶 GPT-2 demo | ✅ Multi-model | ✅ Multi-model | ✅ Multi-model |
| Quantization | ❌ | ✅ GPTQ/AWQ | ✅ | ✅ GGUF |
| Production ready | ❌ | ✅ | ✅ | ✅ |

nxtLLM is currently an **architecture research project** with unique memory management design (three-tier storage + LRU-K + prefix sharing). It requires a full scheduler, multi-model support, quantization, and performance optimization before production deployment.

### v0.5 Priorities

1. page_id → NxtPage hash table (resolve O(n) bottleneck)
2. Defragmentation implementation
3. `paged_kv_t` abstraction + multi-stage async pipeline (from FlashInfer)
4. DecodePlan scheduler for large batch handling
5. Real model weight loading + end-to-end inference validation

## Contributors

Thanks to the following contributors who made this project possible:

| Contributor | Role |
|-------------|------|
| **山野小娃** ([@ahua2020qq](https://github.com/ahua2020qq)) | Project lead, architecture design, core implementation |
| **DeepSeek** ([@deepseek-ai](https://github.com/deepseek-ai)) | Co-design, architectural guidance |
| **豆包** ([Doubao](https://www.doubao.com)) | Co-design, inference optimization insights |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for code style (C11, C++17, CUDA17),
build instructions, testing guidelines, and the PR checklist.

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).
For security issues, see [SECURITY.md](SECURITY.md).

## License

Apache License 2.0
