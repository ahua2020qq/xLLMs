/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#include "operator_api.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int dim;
    int num_heads;
    int head_dim;
    int ffn_dim;
    float* ln1_gamma;
    float* ln1_beta;
    float* ln2_gamma;
    float* ln2_beta;
    float* qkv_weight;
    float* attn_proj_weight;
    float* ffn_w1;
    float* ffn_w2;
} TransformerBlock;

static TransformerBlock* transformer_block_create(int dim, int num_heads, int ffn_dim) {
    TransformerBlock* b = malloc(sizeof(TransformerBlock));
    b->dim = dim;
    b->num_heads = num_heads;
    b->head_dim = dim / num_heads;
    b->ffn_dim = ffn_dim;

    b->ln1_gamma = calloc(dim, sizeof(float));
    b->ln1_beta  = calloc(dim, sizeof(float));
    b->ln2_gamma = calloc(dim, sizeof(float));
    b->ln2_beta  = calloc(dim, sizeof(float));
    b->qkv_weight = calloc(dim * 3 * dim, sizeof(float));
    b->attn_proj_weight = calloc(dim * dim, sizeof(float));
    b->ffn_w1 = calloc(dim * ffn_dim, sizeof(float));
    b->ffn_w2 = calloc(ffn_dim * dim, sizeof(float));

    for (int i = 0; i < dim; i++) {
        b->ln1_gamma[i] = 1.0f;
        b->ln2_gamma[i] = 1.0f;
    }
    for (int i = 0; i < dim * 3 * dim; i++) {
        b->qkv_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for (int i = 0; i < dim * dim; i++) {
        b->attn_proj_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for (int i = 0; i < dim * ffn_dim; i++) {
        b->ffn_w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for (int i = 0; i < ffn_dim * dim; i++) {
        b->ffn_w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }

    return b;
}

static void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

static void transformer_block_forward(
    TransformerBlock* b,
    const float* hidden_states,
    float* key_cache,
    float* value_cache,
    const int* page_table,
    int seq_len,
    int page_size,
    float* output)
{
    int dim = b->dim;
    int num_heads = b->num_heads;
    int head_dim = b->head_dim;
    float eps = 1e-5f;

    float* ln1_out = malloc(dim * sizeof(float));
    float* attn_input = malloc(dim * sizeof(float));
    float* qkv = malloc(3 * dim * sizeof(float));
    float* attn_out = malloc(dim * sizeof(float));
    float* residual1 = malloc(dim * sizeof(float));
    float* ln2_out = malloc(dim * sizeof(float));
    float* ffn_h = malloc(b->ffn_dim * sizeof(float));
    float* ffn_out = malloc(dim * sizeof(float));

    // LayerNorm 1
    layer_norm(hidden_states, b->ln1_gamma, b->ln1_beta, dim, eps, ln1_out);

    // QKV projection: ln1_out [dim] * qkv_weight [dim, 3*dim] -> qkv [3*dim]
    matmul(ln1_out, b->qkv_weight, qkv, 1, dim, 3 * dim);

    // Extract query (last position only)
    float* query = malloc(num_heads * head_dim * sizeof(float));
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            int src_idx = h * head_dim + d;
            query[h * head_dim + d] = qkv[src_idx];
        }
    }

    // Paged attention
    paged_attention(query, key_cache, value_cache, page_table,
                    seq_len, num_heads, head_dim, page_size, attn_input);

    // Attention output projection: attn_input [dim] * attn_proj_weight [dim, dim] -> attn_out [dim]
    matmul(attn_input, b->attn_proj_weight, attn_out, 1, dim, dim);

    // Residual 1
    for (int i = 0; i < dim; i++) {
        residual1[i] = hidden_states[i] + attn_out[i];
    }

    // LayerNorm 2
    layer_norm(residual1, b->ln2_gamma, b->ln2_beta, dim, eps, ln2_out);

    // FFN: ln2_out [dim] * ffn_w1 [dim, ffn_dim] -> ffn_h [ffn_dim]
    matmul(ln2_out, b->ffn_w1, ffn_h, 1, dim, b->ffn_dim);

    // GELU activation
    gelu(ffn_h, b->ffn_dim, ffn_h);

    // FFN output: ffn_h [ffn_dim] * ffn_w2 [ffn_dim, dim] -> ffn_out [dim]
    matmul(ffn_h, b->ffn_w2, ffn_out, 1, b->ffn_dim, dim);

    // Residual 2 -> final output
    for (int i = 0; i < dim; i++) {
        output[i] = residual1[i] + ffn_out[i];
    }

    free(ln1_out);
    free(attn_input);
    free(qkv);
    free(attn_out);
    free(residual1);
    free(ln2_out);
    free(ffn_h);
    free(ffn_out);
    free(query);
}

static void transformer_block_free(TransformerBlock* b) {
    free(b->ln1_gamma);
    free(b->ln1_beta);
    free(b->ln2_gamma);
    free(b->ln2_beta);
    free(b->qkv_weight);
    free(b->attn_proj_weight);
    free(b->ffn_w1);
    free(b->ffn_w2);
    free(b);
}
