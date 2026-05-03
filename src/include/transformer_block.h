/*
 * nxtLLM Transformer Block — GPT-2 Forward Pass
 *
 * Implements a single GPT-2 transformer block:
 *   x = x + attention(ln_1(x))
 *   x = x + mlp(ln_2(x))
 *
 * Uses nxtLLM operator APIs for paged attention and activation functions.
 */

#ifndef NXTLLM_TRANSFORMER_BLOCK_H_
#define NXTLLM_TRANSFORMER_BLOCK_H_

#include <stdint.h>
#include "weight_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── KV-Cache block descriptor ──────────────────────────────────────── */
typedef struct {
    float *k;            /* key cache buffer */
    float *v;            /* value cache buffer */
    int   *block_table;  /* physical block mapping */
    int    num_blocks;   /* allocated blocks */
    int    block_size;   /* tokens per block */
    int    kv_block_stride;
    int    kv_head_stride;
} Gpt2KVCache;

/* ── Transformer block execution context ────────────────────────────── */
typedef struct {
    Gpt2Config     config;
    Gpt2Weights   *weights;
    Gpt2KVCache   *kv_cache;

    /* Scratch buffers for intermediate tensors */
    float *hidden_buf;     /* [n_embd] hidden state */
    float *attn_buf;       /* [n_embd] attention output */
    float *mlp_buf;        /* [n_inner] MLP hidden */
    float *ln1_buf;        /* [n_embd] after ln_1 */
    float *ln2_buf;        /* [n_embd] after ln_2 */
    float *qkv_buf;        /* [3 * n_embd] QKV concatenated */

    void  *stream;         /* opaque CUDA stream (NULL = CPU fallback) */
} TransformerCtx;

/* ── Lifecycle ──────────────────────────────────────────────────────── */

/** Initialise transformer execution context. */
void transformer_init(TransformerCtx *ctx, const Gpt2Config *cfg,
                      Gpt2Weights *weights, Gpt2KVCache *kv_cache);

/** Release scratch buffers. Does NOT free weights or KV-cache. */
void transformer_destroy(TransformerCtx *ctx);

/* ── Forward pass ───────────────────────────────────────────────────── */

/**
 * Execute one transformer block forward pass.
 *
 * @param ctx        Transformer context
 * @param block_idx  Which transformer layer (0..n_layer-1)
 * @param hidden     [n_embd] input hidden state, overwritten with output
 * @param seq_len    Current sequence length (for KV-cache offset)
 * @param seq_idx    Sequence index (always 0 for single-batch)
 */
void transformer_block_forward(TransformerCtx *ctx, int block_idx,
                               float *hidden, int seq_len, int seq_idx);

/**
 * Execute the full GPT-2 forward pass through all layers.
 *
 * @param ctx        Transformer context
 * @param hidden     [n_embd] input embedding, overwritten with final hidden state
 * @param seq_len    Current sequence length
 */
void transformer_forward(TransformerCtx *ctx, float *hidden, int seq_len);

/* ── Internal helpers (exposed for testing) ─────────────────────────── */

/** LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta */
void gpt2_layer_norm(const float *x, float *y, const float *gamma,
                     const float *beta, int d, float eps);

/** Linear layer: y = xW + b  (x: [1, in_dim], W: [in_dim, out_dim]) */
void gpt2_linear(const float *x, const float *w, const float *b,
                 float *y, int in_dim, int out_dim);

/** GELU activation (tanh approximation, element-wise). */
void gpt2_gelu(float *x, int n);

#ifdef __cplusplus
}
#endif

#endif /* NXTLLM_TRANSFORMER_BLOCK_H_ */
