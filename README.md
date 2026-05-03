# nxtLLM V0.1 — Next-Generation LLM Inference Engine

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Conduct-Contributor%20Covenant%202.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Security](https://img.shields.io/badge/security-report%20vulnerability-orange.svg)](SECURITY.md)

nxtLLM is a high-performance LLM inference engine with a multi-tier memory management system.
It implements LRU-K page eviction, tier-aware page allocation (GPU HBM → CPU DRAM → NVMe SSD),
and admission control for concurrent inference requests.

> **本项目由 [山野小娃](https://github.com/ahua2020qq)、[DeepSeek](https://github.com/deepseek-ai) 和 [豆包](https://www.doubao.com) 共同设计。**
>
> 感谢 [vLLM](https://github.com/vllm-project/vllm) 和 [Llama.cpp](https://github.com/ggerganov/llama.cpp) 为开源 LLM 推理社区做出的卓越贡献，nxtLLM 未来将参考这些成熟项目中已验证的设计理念。
>
> 衷心希望大模型推理能够更快进入 v1.0+ 时代 —— v0.20.X 的漫长过渡，着实让人有些尴尬。

## Project Goals

- **Efficient memory utilization**: Three-tier buffer pool spanning GPU, CPU, and SSD storage
- **Intelligent page eviction**: LRU-K (K=3) algorithm prevents thrashing of recently accessed pages
- **Request-aware scheduling**: Per-request page directories with admission control
- **Background defragmentation**: Coalesces free pages to reduce fragmentation
- **Modular architecture**: Clean separation of page pool, LRU-K, and memory manager layers

## Architecture (5-Layer Design)

```
+═══════════════════════════════════════════════════════════════+
║                    LAYER 5: INFERENCE ENGINE                  ║
║   Model runner, tokenizer, KV-cache manager, sampler          ║
║   (future: inference_runner.c, kv_cache.c, sampler.c)         ║
+═══════════════════════════════════════════════════════════════+
                              │
                              ▼
+═══════════════════════════════════════════════════════════════+
║                 LAYER 4: REQUEST SCHEDULER                    ║
║   Admission control, request queue, priority management       ║
║   (future: scheduler.c, admission.c)                          ║
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

## Quick Start

### Prerequisites

- CMake >= 3.16
- C11-compliant compiler (GCC >= 9, Clang >= 10)

### Build & Run

```sh
git clone <repo-url> && cd nxtLLM

# Configure and build
cmake -B build -S .
cmake --build build

# Run tests
cd build && ctest --output-on-failure
```

### Basic Usage

```c
#include "page_pool.h"

int main() {
    NxtGlobalBufferPool pool;
    nxt_pool_init(&pool,
                  /* GPU */ 4ULL * 1024 * 1024 * 1024,  // 4 GiB
                  /* CPU */ 16ULL * 1024 * 1024 * 1024, // 16 GiB
                  /* SSD */ 64ULL * 1024 * 1024 * 1024, // 64 GiB
                  /* page_size */ 64 * 1024,
                  /* max_pages */ 100000);

    NxtPage *page = nxt_page_alloc(&pool, PAGE_TYPE_DATA, TIER_GPU);
    // ... use page->data ...
    nxt_page_ref_dec(&pool, page);

    nxt_pool_destroy(&pool);
    return 0;
}
```

## Directory Layout

```
nxtLLM/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── include/
│   │   ├── page_pool.h
│   │   └── lru_k.h
│   └── core/
│       └── memory_manager.c
├── tests/
│   └── test_page_pool.c
└── docs/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for code style (C11),
build instructions, testing guidelines, and the PR checklist.

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).
For security issues, see [SECURITY.md](SECURITY.md).

## License

Apache License 2.0
