/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    int dim;
    int num_heads;
    int head_dim;
    int num_layers;
    int max_seq_len;
    int page_size;
    int max_pages;
    float* key_cache;
    float* value_cache;
    int* page_table;
    void** layers;
    float* embed_weight;
    float* lm_head_weight;
} GPT2Model;

typedef struct {
    int num_layers;
    int dim;
    int num_heads;
    int head_dim;
    int ffn_dim;
    int page_size;
    float* key_cache;
    float* value_cache;
    int* page_table;
} DecodeState;

static void rms_norm(const float* x, float* out, int dim) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float rms = 1.0f / sqrtf(sum_sq / dim + 1e-5f);
    for (int i = 0; i < dim; i++) out[i] = x[i] * rms;
}

static int argmax(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

static int decode_step(
    DecodeState* state,
    int token_id,
    const float* embed_weight,
    const float* lm_head_weight,
    float* current_hidden,
    float* logits_buffer,
    int vocab_size)
{
    int dim = state->dim;
    int num_layers = state->num_layers;
    int num_heads = state->num_heads;
    int head_dim = state->head_dim;
    int page_size = state->page_size;

    // Embed the input token
    for (int i = 0; i < dim; i++) {
        current_hidden[i] = embed_weight[token_id * dim + i];
    }

    // Placeholder: run through transformer layers (would call transformer_block_forward per layer)
    // Here we simulate token prediction with a deterministic hash-based approach
    // In a real implementation, each layer's forward() would be invoked
    float* hidden_in = malloc(dim * sizeof(float));
    float* hidden_out = malloc(dim * sizeof(float));
    memcpy(hidden_in, current_hidden, dim * sizeof(float));

    // Simulate transformer processing (simplified)
    for (int layer = 0; layer < num_layers; layer++) {
        // Apply simple non-linear transform to simulate layer processing
        for (int i = 0; i < dim; i++) {
            float x = hidden_in[i];
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float x3 = x * x * x;
            float inner = 0.79788456f * (x + 0.044715f * x3);
            float gelu_x = 0.5f * x * (1.0f + tanhf(inner));
            hidden_out[i] = hidden_in[i] + 0.1f * gelu_x;
        }
        memcpy(hidden_in, hidden_out, dim * sizeof(float));
    }

    memcpy(current_hidden, hidden_out, dim * sizeof(float));

    // LM head projection: current_hidden [1, dim] * lm_head_weight [dim, vocab_size]
    for (int v = 0; v < vocab_size; v++) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += current_hidden[d] * lm_head_weight[d * vocab_size + v];
        }
        logits_buffer[v] = sum;
    }

    free(hidden_in);
    free(hidden_out);

    return argmax(logits_buffer, vocab_size);
}

static DecodeState* decode_state_create(
    int num_layers, int dim, int num_heads, int ffn_dim,
    int max_seq_len, int page_size)
{
    DecodeState* s = malloc(sizeof(DecodeState));
    s->num_layers = num_layers;
    s->dim = dim;
    s->num_heads = num_heads;
    s->head_dim = dim / num_heads;
    s->ffn_dim = ffn_dim;
    s->page_size = page_size;

    int max_pages = (max_seq_len + page_size - 1) / page_size;
    size_t cache_per_layer = max_pages * page_size * num_heads * (dim / num_heads);
    s->key_cache = calloc(num_layers * cache_per_layer, sizeof(float));
    s->value_cache = calloc(num_layers * cache_per_layer, sizeof(float));
    s->page_table = calloc(num_layers * max_seq_len, sizeof(int));
    for (int i = 0; i < num_layers * max_seq_len; i++) s->page_table[i] = -1;

    return s;
}

static void decode_state_free(DecodeState* s) {
    free(s->key_cache);
    free(s->value_cache);
    free(s->page_table);
    free(s);
}
