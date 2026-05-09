/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * xLLM Tokenizer — BPE Tokenizer (Llama 3 / GPT-2 style)
 *
 * Loads vocabulary and BPE merge rules from a GGUF model file.
 * Supports encode (text → token ids) and decode (token ids → text).
 */

#ifndef XLLM_TOKENIZER_H_
#define XLLM_TOKENIZER_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XLLM_MAX_TOKEN_LEN   128
#define XLLM_MAX_ENCODED     2048

typedef struct {
    char   *str;         /* token string (owned) */
    int32_t id;
} TokenEntry;

typedef struct {
    char   *left;        /* left part of merge (owned) */
    char   *right;       /* right part of merge (owned) */
    int32_t priority;    /* lower = higher priority */
} BpeMerge;

typedef struct {
    TokenEntry *vocab;       /* id → token string, size=vocab_size */
    int32_t     vocab_size;

    BpeMerge   *merges;      /* BPE merge rules */
    int32_t     num_merges;

    int32_t     bos_token_id;
    int32_t     eos_token_id;
    int32_t     pad_token_id;

    int32_t     byte_token_id[256]; /* byte value → token id (GPT-2 mapping) */
} XllmTokenizer;

/** Load tokenizer directly from a GGUF model file. */
XllmTokenizer *xllm_tokenizer_load_from_gguf(const char *gguf_path);

/** Free tokenizer resources. */
void xllm_tokenizer_free(XllmTokenizer *tok);

/** Encode a UTF-8 string into token IDs. Returns number of tokens written. */
int32_t xllm_tokenizer_encode(XllmTokenizer *tok, const char *text,
                               int32_t *token_ids, int32_t max_tokens);

/** Decode token IDs back to a string. Returns length written (excluding null). */
int32_t xllm_tokenizer_decode(XllmTokenizer *tok, const int32_t *token_ids,
                               int32_t num_tokens, char *out, int32_t max_len);

#ifdef __cplusplus
}
#endif

#endif /* XLLM_TOKENIZER_H_ */
