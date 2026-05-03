# nxtLLM V0.2 — Next-Generation LLM Inference Engine

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
- **GPU-accelerated operators**: Paged attention, activation kernels (SiLU/GELU), and KV-cache management

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

## What's New in v0.2

### GPU Operators (`operators/`)

nxtLLM v0.2 introduces a dedicated operator library with CUDA-accelerated kernels:

| Operator | File | Description |
|---|---|---|
| **Paged Attention** | `operators/page_attention.cu` | Multi-head attention over paged KV-cache blocks. Supports GQA/MQA with configurable block sizes and head dimensions. Adapted from the vLLM paged_attention_v1 kernel. |
| **SiLU & Mul** | `operators/activation_kernels.cu` | Fused SiLU-gate + element-wise multiply (silu_and_mul / mul_and_silu). |
| **GELU & Mul** | `operators/activation_kernels.cu` | Fused GELU-gate + multiply with "none" and "tanh" approximations. |
| **Element-wise Activations** | `operators/activation_kernels.cu` | Standalone SiLU and GELU element-wise kernels. |
| **KV-Cache Config** | `operators/cache.py` | Python configuration dataclass for block size, memory utilization, cache dtype, and prefix caching. |

All GPU kernels are exposed through a unified C API defined in `include/operator_api.h`.

### Key Changes

- **New** `nxtllm_operators` static library with CUDA compilation
- **New** `include/operator_api.h` — unified C API for all GPU operators
- **New** `operators/cache.py` — KV-cache configuration module
- **New** `tests/test_attention.c` — operator API signature and smoke tests
- CMakeLists.txt updated: C++17/CUDA17, CUDA Toolkit required, SM 75/80/86/89/90 targets
- `gptq_kernels.cu` deferred to v0.3 (quantization pipeline requires additional dependencies)

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
- C11/C++17-compliant compiler (GCC >= 9, Clang >= 10)
- CUDA Toolkit >= 11.8
- NVIDIA driver >= 525 (SM 75+)

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
#include "operator_api.h"

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
├── include/
│   └── operator_api.h          # GPU operator C API (v0.2)
├── operators/
│   ├── page_attention.cu       # Paged attention kernel (v0.2)
│   ├── activation_kernels.cu   # SiLU/GELU activation kernels (v0.2)
│   └── cache.py                # KV-cache configuration (v0.2)
├── src/
│   ├── include/
│   │   ├── page_pool.h
│   │   └── lru_k.h
│   └── core/
│       └── memory_manager.c
├── tests/
│   ├── test_page_pool.c
│   └── test_attention.c        # Operator API tests (v0.2)
└── docs/
```

## Roadmap

### v0.1 (current)
- [x] Three-tier buffer pool (GPU/CPU/SSD)
- [x] LRU-K (K=3) page eviction
- [x] Page allocation and reference counting
- [x] Basic test suite

### v0.2
- [x] Paged attention CUDA kernel
- [x] Activation kernels (SiLU, GELU)
- [x] KV-cache configuration module
- [x] Unified operator C API

### v0.3 (planned)
- [ ] GPTQ / AWQ quantization kernels
- [ ] FlashAttention integration
- [ ] CUDA graph capture for decode phase
- [ ] Benchmarking harness

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for code style (C11, C++17, CUDA17),
build instructions, testing guidelines, and the PR checklist.

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).
For security issues, see [SECURITY.md](SECURITY.md).

## License

Apache License 2.0
