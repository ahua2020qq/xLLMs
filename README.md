# nxtLLM — Next-Generation LLM Inference Engine

nxtLLM is a high-performance LLM inference engine with a multi-tier memory management system.
It implements LRU-K page eviction, tier-aware page allocation (GPU HBM → CPU DRAM → NVMe SSD),
and admission control for concurrent inference requests.

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

## License

MIT
