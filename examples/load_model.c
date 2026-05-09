/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Example: GGUF Model Loader
 *
 * Demonstrates loading a GGUF model file, parsing metadata,
 * listing tensor info, and loading weight data.
 *
 * Usage: ./load_model <model.gguf>
 *
 * Build: cmake -B build && cmake --build build --target load_model
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

#include "model_loader.h"
#include "weight_loader.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        fprintf(stderr, "  Loads a GGUF model file and prints summary information.\n");
        return 1;
    }

    const char *model_path = argv[1];
    printf("Loading GGUF model: %s\n", model_path);

    /* ── Pass 1: Parse header and metadata ────────────────────────────── */
    GgufContext *ctx = gguf_init_from_file(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to parse GGUF file: %s\n", model_path);
        return 1;
    }

    /* Print full summary */
    gguf_print_summary(ctx);
    printf("\n");

    /* ── Architecture detection ────────────────────────────────────────── */
    ModelArch arch = gguf_detect_architecture(ctx);
    printf("Detected architecture: %s (%d)\n", model_arch_name(arch), arch);

    /* ── Load model configuration ──────────────────────────────────────── */
    Gpt2Config cfg;
    if (gguf_load_model_config(ctx, &cfg)) {
        printf("\nModel configuration loaded:\n");
        printf("  vocab_size:     %d\n", cfg.vocab_size);
        printf("  context_length: %d\n", cfg.n_positions);
        printf("  n_embd:         %d\n", cfg.n_embd);
        printf("  n_layer:        %d\n", cfg.n_layer);
        printf("  n_head:         %d\n", cfg.n_head);
        printf("  n_inner:        %d\n", cfg.n_inner);
        printf("  head_size:      %d\n", cfg.head_size);
        printf("  norm_eps:       %.6e\n", cfg.layer_norm_eps);
    }

    /* ── List KV metadata ──────────────────────────────────────────────── */
    printf("\nKV Metadata (%" PRId64 " pairs):\n", gguf_get_n_kv(ctx));
    int64_t n_kv = gguf_get_n_kv(ctx);
    int64_t show_kv = n_kv > 20 ? 20 : n_kv;
    for (int64_t i = 0; i < show_kv; i++) {
        const char *key = gguf_get_key(ctx, i);
        GgufType type = gguf_get_kv_type(ctx, i);
        printf("  [%3" PRId64 "] %-45s (type=%s)\n", i, key, gguf_type_name(type));
    }
    if (n_kv > 20) {
        printf("  ... and %" PRId64 " more KV pairs\n", n_kv - 20);
    }

    /* ── List all tensors ──────────────────────────────────────────────── */
    printf("\nTensor listing (%" PRId64 " tensors):\n", gguf_get_n_tensors(ctx));
    int64_t n_tensors = gguf_get_n_tensors(ctx);
    size_t total_bytes = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        const GgufTensorInfo *ti = gguf_get_tensor_info(ctx, i);
        if (!ti) continue;
        printf("  [%4" PRId64 "] %-50s shape=[%5" PRId64 ",%5" PRId64 ",%5" PRId64 ",%5" PRId64 "] "
               "type=%s(%d) offset=%9" PRIu64 " nbytes=%10zu\n",
               i, ti->name,
               ti->ne[0], ti->ne[1], ti->ne[2], ti->ne[3],
               ggml_type_name(ti->ggml_type), ti->ggml_type,
               ti->offset, ti->nbytes);
        total_bytes += ti->nbytes;
    }
    printf("  Total: %" PRId64 " tensors, %zu bytes (%.2f MiB)\n",
           n_tensors, total_bytes, (double)total_bytes / (1024.0 * 1024.0));

    /* ── Search for specific tensors ───────────────────────────────────── */
    printf("\nTensor lookup examples:\n");
    const char *test_tensors[] = {
        "token_embd.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        NULL
    };
    for (int i = 0; test_tensors[i]; i++) {
        int64_t id = gguf_find_tensor(ctx, test_tensors[i]);
        if (id >= 0) {
            const GgufTensorInfo *ti = gguf_get_tensor_info(ctx, id);
            printf("  Found '%s' at index %" PRId64 ": %zu bytes\n",
                   test_tensors[i], id, ti->nbytes);
        } else {
            printf("  NOT FOUND: '%s'\n", test_tensors[i]);
        }
    }

    /* ── Verify alignment ──────────────────────────────────────────────── */
    printf("\nAlignment check:\n");
    printf("  Alignment: %zu bytes\n", gguf_get_alignment(ctx));
    printf("  Data offset: %" PRIu64 "\n", gguf_get_data_offset(ctx));
    printf("  Data size: %zu bytes\n", gguf_get_data_size(ctx));

    /* ── Cleanup ───────────────────────────────────────────────────────── */
    gguf_free(ctx);
    printf("\nDone.\n");
    return 0;
}
