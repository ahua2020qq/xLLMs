/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * nxtLLM Tokenizer — GPT-2 BPE Tokenizer
 *
 * Loads GPT-2 vocabulary and merge rules from external files exported
 * by scripts/convert_gpt2_weights.py.  Supports encode (text → token ids)
 * and decode (token ids → text).
 */

#ifndef NXTLLM_TOKENIZER_H_
#define NXTLLM_TOKENIZER_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPT2_VOCAB_SIZE   50257
#define GPT2_MAX_TOKEN_LEN 64
#define GPT2_MAX_ENCODED   2048

/* ── Single vocabulary entry ────────────────────────────────────────── */
typedef struct {
    char   token_str[GPT2_MAX_TOKEN_LEN];
    int32_t id;
} TokenizerEntry;

/* ── BPE merge rule ─────────────────────────────────────────────────── */
typedef struct {
    char pair[2][GPT2_MAX_TOKEN_LEN];
    int32_t priority;
} BpeMergeRule;

/* ── Tokenizer state ────────────────────────────────────────────────── */
typedef struct {
    TokenizerEntry *vocab;       /* id → token string */
    int32_t         vocab_size;

    BpeMergeRule   *merges;      /* BPE merge rules sorted by priority */
    int32_t         num_merges;

    /* Internal lookup tables */
    /* ... reserved for future speed-ups ... */
} Gpt2Tokenizer;

/* ── API ────────────────────────────────────────────────────────────── */

/** Load tokenizer from vocab.json and merges.txt files.
 *  These files are exported by scripts/convert_gpt2_weights.py. */
Gpt2Tokenizer *gpt2_tokenizer_load(const char *vocab_path,
                                   const char *merges_path);

/** Free tokenizer resources. */
void gpt2_tokenizer_free(Gpt2Tokenizer *tok);

/** Encode a UTF-8 string into token IDs.  Returns number of tokens. */
int32_t gpt2_tokenizer_encode(Gpt2Tokenizer *tok, const char *text,
                              int32_t *token_ids, int32_t max_tokens);

/** Decode token IDs back to a string.  Returns length written. */
int32_t gpt2_tokenizer_decode(Gpt2Tokenizer *tok, const int32_t *token_ids,
                              int32_t num_tokens, char *out, int32_t max_len);

/** Create a minimal built-in tokenizer for demo/testing (no external files). */
Gpt2Tokenizer *gpt2_tokenizer_create_minimal(void);

#ifdef __cplusplus
}
#endif

#endif /* NXTLLM_TOKENIZER_H_ */
