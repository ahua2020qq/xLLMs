/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * nxtLLM Autoregressive Decoder — GPT-2 Generation Loop
 *
 * Implements token-by-token autoregressive generation with KV-cache
 * management.  Each iteration produces one token, updates the cache,
 * and feeds the new token back as input for the next step.
 */

#ifndef NXTLLM_DECODER_H_
#define NXTLLM_DECODER_H_

#include <stdint.h>
#include "weight_loader.h"
#include "transformer_block.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Generation parameters ──────────────────────────────────────────── */
typedef struct {
    int32_t max_new_tokens;    /* max tokens to generate (default 50) */
    float   temperature;        /* sampling temperature (default 1.0) */
    int32_t top_k;              /* top-k sampling (0 = disabled) */
    float   top_p;              /* nucleus sampling (1.0 = disabled) */
    int32_t seed;               /* random seed (0 = random) */
} Gpt2GenParams;

/* ── Decoder state ──────────────────────────────────────────────────── */
typedef struct {
    Gpt2Config       config;
    Gpt2Weights     *weights;
    Gpt2Tokenizer   *tokenizer;
    Gpt2KVCache     *kv_cache;
    TransformerCtx   tctx;

    int32_t *input_ids;        /* running token sequence */
    int32_t  seq_len;          /* current length */
    int32_t  seq_capacity;     /* buffer capacity */
} Gpt2Decoder;

/* ── API ────────────────────────────────────────────────────────────── */

/** Initialise decoder with model weights, tokenizer, and KV-cache. */
void gpt2_decoder_init(Gpt2Decoder *dec, const Gpt2Config *cfg,
                       Gpt2Weights *weights, Gpt2Tokenizer *tok,
                       Gpt2KVCache *kv_cache);

/** Release decoder resources (not weights/tokenizer/kv_cache). */
void gpt2_decoder_destroy(Gpt2Decoder *dec);

/**
 * Generate text autoregressively.
 *
 * @param dec       Initialised decoder
 * @param prompt    Input text prompt
 * @param params    Generation parameters
 * @param out       Output buffer for generated text
 * @param max_len   Maximum output length (including prompt)
 * @return          Number of generated tokens, or -1 on error
 */
int32_t gpt2_generate(Gpt2Decoder *dec, const char *prompt,
                      const Gpt2GenParams *params,
                      char *out, int32_t max_len);

/**
 * Single decode step: run logits for the next token given current sequence.
 *
 * @param dec       Decoder state (seq_len is the current sequence length)
 * @param logits    [vocab_size] output logits
 */
void gpt2_decode_step(Gpt2Decoder *dec, float *logits);

/**
 * Sample from logits using temperature, top-k, top-p.
 * Returns the selected token ID.
 */
int32_t gpt2_sample(const float *logits, int32_t vocab_size,
                    const Gpt2GenParams *params);

/** Softmax: converts logits to probabilities in-place. */
void gpt2_softmax(float *x, int32_t n);

/** KV-Cache management: allocate cache blocks for a sequence. */
Gpt2KVCache *gpt2_kv_cache_alloc(const Gpt2Config *cfg, int32_t max_seq_len);

/** Free KV-cache. */
void gpt2_kv_cache_free(Gpt2KVCache *cache);

#ifdef __cplusplus
}
#endif

#endif /* NXTLLM_DECODER_H_ */
