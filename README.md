# xLLM V0.5 — Next-Generation LLM Inference Engine

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Conduct-Contributor%20Covenant%202.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Security](https://img.shields.io/badge/security-report%20vulnerability-orange.svg)](SECURITY.md)

xLLM is a high-performance LLM inference engine with multi-tier memory management,
Triton-style multi-backend architecture, continuous batching scheduler, and GGUF model loading.

> **本项目由 [山野小娃](https://github.com/ahua2020qq)、[DeepSeek](https://github.com/deepseek-ai)、[豆包](https://www.doubao.com) 和 [GLM](https://bigmodel.cn) 共同设计。**
>
> 感谢以下开源项目及其社区为 LLM 推理领域做出的卓越贡献：
>
> - **[vLLM](https://github.com/vllm-project/vllm)** — PagedAttention 与连续批处理启发
> - **[Llama.cpp](https://github.com/ggerganov/llama.cpp)** — GGUF 模型格式与轻量化推理
> - **[SGLang](https://github.com/sgl-project/sglang)** — RadixCache 前缀共享设计（已集成）
> - **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — cp.async 流水线注意力内核（V2 已集成）
> - **[Triton Inference Server](https://github.com/triton-inference-server/server)** — 多后端插件架构（已集成）

## Project Goals

- **Efficient memory utilization**: Three-tier buffer pool spanning GPU, CPU, and SSD storage
- **Intelligent page eviction**: LRU-K (K=3) with O(1) hash-indexed page lookup
- **Prefix-aware KV-cache sharing**: Radix tree indexes token sequences for cross-request cache reuse
- **Multi-backend plugin architecture**: Triton-style dlopen-based backend loading
- **Continuous batching**: iteration-level scheduling with prefill/decode mixed batches
- **GGUF model loading**: Full GGUF v3 parser with Q4_K/Q6_K quantization support
- **GPU-accelerated operators**: Paged attention V1/V2 with cp.async pipeline (SM80+)

## Architecture

```
╔═══════════════════════════════════════════════════════════════╗
║                  GGUF Model Loader                            ║
║   model_loader.c — v3 parser, tensor lookup, arch detection  ║
╠═══════════════════════════════════════════════════════════════╣
║              Continuous Batching Scheduler                     ║
║   scheduler.c — prefill/decode mixed batch, priority queue    ║
╠═══════════════════════════════════════════════════════════════╣
║              Backend Manager + Plugin                          ║
║   backend_manager.c — dlopen backends, model version control  ║
║   backends/cuda_backend.c — CUDA backend (dlopen'd at runtime)║
╠═══════════════════════════════════════════════════════════════╣
║              Inference Engine                                  ║
║   transformer_block.c / tokenizer.c / decode_loop.c            ║
╠═══════════════════════════════════════════════════════════════╣
║   Prefix Sharing (radix tree) + Page Allocator + LRU-K + Pool ║
║   prefix_sharing.c  memory_manager.c  lru_k.c  page_pool.h    ║
╠═══════════════════════════════════════════════════════════════╣
║              Three-Tier Buffer Pool                            ║
║   GPU HBM (fast) → CPU DRAM (medium) → NVMe SSD (slow)       ║
╚═══════════════════════════════════════════════════════════════╝
```

## What's New

### v0.5 (current)

**Epic 1: Page Hash Index** — O(1) open-addressing hash table maps page_id → NxtPage*. Eliminates O(n²) eviction bottleneck.

**Epic 2: Background Defragmentation** — Per-tier compaction with pointer-swapping. No data copying, automatic free list rebuild.

**Epic 3: Paged Attention V2** — cp.async 3-stage pipeline, vectorized loads, GQA-aware thread layout (GROUP_SIZE 1/2/4/8/16). SM80+ with runtime fallback to V1.

**Epic 4: Multi-Backend Architecture** — Triton-style dlopen-based backend loading. Model version control (LATEST/ALL/SPECIFIC). Health checks and state management.

**Epic 5: Continuous Batching Scheduler** — iteration-level scheduling, prefill/decode mixed batches, priority queue.

**Epic 6: GGUF v3 Model Loader** — Full binary parser, Q4_K/Q6_K dequantization, architecture auto-detection. Successfully loads Llama-3-8B on V100.

### v0.4

- Prefix-sharing radix tree (SGLang RadixCache inspired)
- FlashInfer-style batch decode adapter
- TensorRT-LLM stub adapter

### v0.3

- GPT-2 inference engine (TransformerBlock, BPE Tokenizer, Decode Loop)
- CUDA optional build

### v0.2

- Paged attention CUDA kernel
- SiLU/GELU activation kernels

### v0.1

- Three-tier buffer pool (GPU/CPU/SSD)
- LRU-K (K=3) page eviction

## Quick Start

### Prerequisites

- CMake >= 3.16
- C11/C++17-compliant compiler (GCC >= 9, Clang >= 10)
- *(Optional)* CUDA Toolkit >= 11.8 (SM 75+)

### Build & Run

```sh
git clone https://github.com/ahua2020qq/xLLMs.git && cd xLLMs

# CPU-only build (default)
cmake -B build -S .
cmake --build build

# With CUDA support
cmake -B build -S . -DBUILD_CUDA_BACKEND=ON
cmake --build build

# With CUDA + FlashInfer adapter (SM80+)
cmake -B build -S . -DBUILD_CUDA_BACKEND=ON -DUSE_FLASHINFER=ON
cmake --build build

# Run tests
cd build && ctest --output-on-failure

# Load a GGUF model
./load_model /path/to/model.gguf

# GPT-2 inference demo (random weights)
./run_gpt2
```

## Core Data Structures

### NxtPage
A single memory page (default 64 KB) belonging to a type (DATA / INDEX / CONTROL)
and residing on a storage tier (GPU / CPU / SSD). Tracked in doubly-linked free lists
with O(1) hash-indexed lookup and LRU-K timestamps.

### NxtGlobalBufferPool
Per-tier, per-type free page lists, LRU-K tracker, page hash index, memory budget caps,
and global allocation/eviction/defrag statistics.

### NxtPrefixTree (Radix Tree)
Token-sequence radix tree for KV-cache prefix sharing across requests.
Automatic node splitting with lock_ref eviction protection.

### NxtBackendAPI
Function table for plugin backends: initialize / model_load / infer / model_unload / finalize.
Loaded at runtime via dlopen.

### GGUF Context
Full GGUF v3 parser: magic validation, KV pair parsing, tensor metadata extraction,
data section alignment, Q4_K/Q6_K dequantization.

## Directory Layout

```
xLLMs/
├── CMakeLists.txt
├── LICENSE                    # Apache 2.0
├── README.md
├── cmake/
│   └── xllm_config.h.in      # CMake config template
├── include/
│   ├── operator_api.h         # GPU operator C API
│   ├── prefix_sharing.h       # Radix tree API
│   ├── backend.h              # Backend plugin API
│   └── tensorrt_adapter.h     # TensorRT stub adapter
├── operators/
│   ├── page_attention.cu      # Paged attention V1
│   ├── page_attention_v2.cu   # Paged attention V2 (cp.async pipeline)
│   ├── activation_kernels.cu  # SiLU/GELU kernels
│   ├── flashinfer_adapter.cu  # FlashInfer-style adapter
│   └── cache.py               # KV-cache configuration
├── backends/
│   ├── cuda_backend.c         # CUDA backend plugin
│   └── cuda_backend.h
├── src/
│   ├── include/               # Internal headers
│   ├── core/
│   │   ├── memory_manager.c   # Page allocator + hash index + defrag
│   │   ├── lru_k.c            # LRU-K (K=3) eviction
│   │   ├── prefix_sharing.c   # Radix tree implementation
│   │   ├── backend_manager.c  # Multi-backend manager
│   │   ├── scheduler.c        # Continuous batching scheduler
│   │   └── model_loader.c     # GGUF v3 model loader
│   ├── inference/
│   │   ├── transformer_block.c
│   │   ├── tokenizer.c
│   │   └── decode_loop.c
│   └── adapter/
│       └── tensorrt_adapter.c
├── examples/
│   ├── run_gpt2.c
│   ├── load_model.c
│   └── text_gen.c
├── tests/
│   ├── test_page_pool.c
│   ├── test_prefix_sharing.c
│   ├── test_attention.c
│   ├── test_backend.c
│   ├── test_model_version.c
│   ├── test_model_loader.c
│   ├── test_scheduler.c
│   └── test_tensorrt_integration.c
└── scripts/
    └── convert_gpt2_weights.py
```

## Roadmap

### v0.1 ✅
- [x] Three-tier buffer pool (GPU/CPU/SSD)
- [x] LRU-K (K=3) page eviction
- [x] Page allocation and reference counting

### v0.2 ✅
- [x] Paged attention CUDA kernel
- [x] Activation kernels (SiLU, GELU)
- [x] Unified operator C API

### v0.3 ✅
- [x] GPT-2 inference engine
- [x] CUDA optional build

### v0.4 ✅
- [x] Prefix-sharing radix tree
- [x] FlashInfer-style batch decode adapter
- [x] TensorRT-LLM stub adapter

### v0.5 ✅ (current)
- [x] Page hash index — O(1) eviction lookup
- [x] Background defragmentation — per-tier compaction
- [x] Paged Attention V2 — cp.async pipeline (SM80+)
- [x] Multi-backend plugin architecture (Triton-style)
- [x] Continuous batching scheduler
- [x] GGUF v3 model loader with Q4_K/Q6_K dequantization
- [x] Model version control (LATEST/ALL/SPECIFIC)

### v0.6 (planned)
- [ ] HTTP/gRPC serving API
- [ ] Multi-model support (Llama, Mistral, Qwen)
- [ ] GPTQ/AWQ quantization
- [ ] Performance benchmarks and tuning

## Technical Highlights

**Three-Tier Storage** — 9 doubly-linked free lists `[3 tiers × 3 types]` with O(1) allocation. Atomic reference counting. Lazy data buffer allocation.

**LRU-K (K=3) + O(1) Hash Index** — Ring buffer tracks 3 access timestamps. Open-addressing hash table for O(1) page lookup. Eviction from O(n²) to O(k).

**Background Defragmentation** — Per-tier compaction with pointer-swapping (no memmove). Automatic free list rebuild.

**Prefix-Sharing Radix Tree** — Patricia Trie with automatic node splitting. `lock_ref` chain protects active KV-cache. LRU eviction integration.

**Paged Attention V2** — 3-stage cp.async pipeline. Vectorized loads. GQA-aware (GROUP_SIZE 1/2/4/8/16). 8 head sizes (32–256). SM80+ with V1 fallback.

**Multi-Backend Plugin** — Triton-style dlopen loading. Model version control (LATEST/ALL/SPECIFIC). Health checks. CUDA backend plugin.

**Continuous Batching** — iteration-level scheduling. Prefill/decode mixed batches. Priority queue.

**GGUF v3 Loader** — Full binary parser. Q4_K/Q6_K dequantization. Architecture auto-detection (llama, qwen2, ...). Successfully loads Llama-3-8B on V100.

## Competitive Positioning

| Dimension | xLLM v0.5 | vLLM | SGLang | Triton | llama.cpp |
|-----------|-----------|------|--------|--------|-----------|
| Three-tier storage | ✅ GPU/CPU/SSD | GPU/CPU | GPU/CPU | GPU/CPU | CPU/Disk |
| Prefix sharing | ✅ Radix tree | ❌ | ✅ RadixCache | ❌ | ❌ |
| LRU-K eviction | ✅ K=3 + O(1) hash | LRU | Pluggable | — | LRU |
| Defragmentation | ✅ Per-tier compaction | ❌ | ❌ | ❌ | ❌ |
| GPU operators | ✅ V1 + V2 pipeline | ✅ Full | ✅ Full | ✅ Full | ❌ |
| Multi-backend plugin | ✅ dlopen | ❌ | ❌ | ✅ | ❌ |
| Model version control | ✅ | ❌ | ❌ | ✅ | ❌ |
| Continuous batching | ✅ | ✅ | ✅ | ✅ | ❌ |
| GGUF loading | ✅ v3 + Q4_K/Q6_K | ❌ | ❌ | ❌ | ✅ native |
| Multi-model support | 🔶 GGUF ready | ✅ | ✅ | ✅ | ✅ |
| Quantization | 🔶 GGUF Q4/Q6 | ✅ GPTQ/AWQ | ✅ | ✅ | ✅ GGUF full |
| HTTP/gRPC API | ❌ | ✅ | ✅ | ✅ | ❌ |

## Contributors

Thanks to the following contributors who made this project possible:

| Contributor | Role |
|-------------|------|
| **山野小娃** ([@ahua2020qq](https://github.com/ahua2020qq)) | Project lead, architecture design, core implementation |
| **DeepSeek** ([@deepseek-ai](https://github.com/deepseek-ai)) | Co-design, architectural guidance |
| **豆包** ([Doubao](https://www.doubao.com)) | Co-design, inference optimization insights |
| **GLM** ([智谱AI](https://bigmodel.cn)) | Co-design, model architecture insights |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for code style (C11, C++17, CUDA17),
build instructions, testing guidelines, and the PR checklist.

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).
For security issues, see [SECURITY.md](SECURITY.md).

## License

Apache License 2.0
