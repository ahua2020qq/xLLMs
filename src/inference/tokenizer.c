/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * BPE Tokenizer — loads vocabulary and merge rules from GGUF metadata.
 */

#include "tokenizer.h"
#include "model_loader.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <inttypes.h>

/* ── GPT-2 byte ↔ Unicode mapping ──────────────────────────────────── */

static int32_t gpt2_byte_to_codepoint(uint8_t b) {
    /* GPT-2 bytes_to_unicode: direct-mapped ranges: 33-126, 161-172, 174-255 */
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 /* && b <= 255 always for uint8 */))
        return (int32_t)b;
    int32_t n = 0;
    for (int i = 0; i < 256; i++) {
        if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))
            continue;
        if (i == (int)b) return 256 + n;
        n++;
    }
    return 0;
}

static int codepoint_to_utf8(int32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; out[1] = '\0'; return 1; }
    else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0'; return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        out[3] = '\0'; return 3;
    }
    return 0;
}

/* ── Vocab string → id lookup ───────────────────────────────────────── */

static int32_t vocab_find(XllmTokenizer *tok, const char *str) {
    for (int32_t i = 0; i < tok->vocab_size; i++) {
        if (strcmp(tok->vocab[i].str, str) == 0) return i;
    }
    return -1;
}

/* ── Load from GGUF ──────────────────────────────────────────────────── */

XllmTokenizer *xllm_tokenizer_load_from_gguf(const char *gguf_path) {
    if (!gguf_path) return NULL;

    uint64_t token_count = 0;
    char **tokens = gguf_read_arr_str(gguf_path, "tokenizer.ggml.tokens", &token_count);
    if (!tokens || token_count == 0) {
        fprintf(stderr, "[tokenizer] Failed to read tokenizer.ggml.tokens\n");
        return NULL;
    }
    fprintf(stderr, "[tokenizer] Loaded %" PRIu64 " tokens from GGUF\n", token_count);

    uint64_t type_count = 0;
    int32_t *token_types = gguf_read_arr_int(gguf_path, "tokenizer.ggml.token_type", &type_count);

    uint64_t merge_count = 0;
    char **merge_strs = gguf_read_arr_str(gguf_path, "tokenizer.ggml.merges", &merge_count);
    fprintf(stderr, "[tokenizer] Loaded %" PRIu64 " merges from GGUF\n", merge_count);

    XllmTokenizer *tok = (XllmTokenizer *)calloc(1, sizeof(XllmTokenizer));
    tok->vocab_size = (int32_t)token_count;
    tok->vocab = (TokenEntry *)calloc((size_t)token_count, sizeof(TokenEntry));

    for (uint64_t i = 0; i < token_count; i++) {
        tok->vocab[i].id = (int32_t)i;
        tok->vocab[i].str = strdup(tokens[i] ? tokens[i] : "");
    }

    /* Build byte→token mapping for GPT-2 tokenizer */
    for (int b = 0; b < 256; b++) {
        char utf8[8];
        int32_t cp = gpt2_byte_to_codepoint((uint8_t)b);
        codepoint_to_utf8(cp, utf8);
        int32_t id = vocab_find(tok, utf8);
        tok->byte_token_id[b] = (id >= 0) ? id : -1;
        if (id < 0)
            fprintf(stderr, "[tokenizer] WARNING: byte 0x%02X -> U+%04X '%s' not in vocab\n",
                    b, cp, utf8);
    }

    /* Parse merges: each string is "left right" */
    if (merge_strs && merge_count > 0) {
        tok->num_merges = (int32_t)merge_count;
        tok->merges = (BpeMerge *)calloc((size_t)merge_count, sizeof(BpeMerge));

        for (uint64_t i = 0; i < merge_count; i++) {
            if (!merge_strs[i]) continue;
            char *space = strchr(merge_strs[i], ' ');
            if (!space) continue;
            *space = '\0';
            tok->merges[i].left  = strdup(merge_strs[i]);
            tok->merges[i].right = strdup(space + 1);
            tok->merges[i].priority = (int32_t)i;
        }
    }

    /* Special tokens */
    GgufContext *ctx = gguf_init_from_file(gguf_path);
    if (ctx) {
        tok->bos_token_id = (int32_t)gguf_get_val_int(ctx, "tokenizer.ggml.bos_token_id", 1);
        tok->eos_token_id = (int32_t)gguf_get_val_int(ctx, "tokenizer.ggml.eos_token_id", 2);
        tok->pad_token_id = (int32_t)gguf_get_val_int(ctx, "tokenizer.ggml.pad_token_id", -1);
        gguf_free(ctx);
    } else {
        tok->bos_token_id = 1;
        tok->eos_token_id = 128001;
    }

    (void)type_count;
    for (uint64_t i = 0; i < token_count; i++) free(tokens[i]);
    free(tokens);
    free(token_types);
    for (uint64_t i = 0; i < merge_count; i++) free(merge_strs[i]);
    free(merge_strs);

    fprintf(stderr, "[tokenizer] Ready: vocab=%d merges=%d bos=%d eos=%d\n",
            tok->vocab_size, tok->num_merges, tok->bos_token_id, tok->eos_token_id);
    return tok;
}

void xllm_tokenizer_free(XllmTokenizer *tok) {
    if (!tok) return;
    for (int32_t i = 0; i < tok->vocab_size; i++) free(tok->vocab[i].str);
    free(tok->vocab);
    for (int32_t i = 0; i < tok->num_merges; i++) {
        free(tok->merges[i].left);
        free(tok->merges[i].right);
    }
    free(tok->merges);
    free(tok);
}

/* ── Encode: byte-level BPE for GPT-2 / Qwen2 tokenizer ─────────────── */

int32_t xllm_tokenizer_encode(XllmTokenizer *tok, const char *text,
                               int32_t *token_ids, int32_t max_tokens) {
    if (!tok || !text || !token_ids || max_tokens < 1) return 0;

    int32_t n_out = 0;
    int32_t text_len = (int32_t)strlen(text);

    if (tok->bos_token_id >= 0 && n_out < max_tokens)
        token_ids[n_out++] = tok->bos_token_id;

    /* Convert text to sequence of byte tokens */
    int32_t byte_tokens[4096];
    int32_t n_byte = 0;
    for (int i = 0; i < text_len && n_byte < (int32_t)(sizeof(byte_tokens)/sizeof(byte_tokens[0])); i++) {
        uint8_t b = (uint8_t)text[i];
        int32_t id = tok->byte_token_id[b];
        if (id < 0) {
            fprintf(stderr, "[tokenizer] ERROR: byte 0x%02X has no token\n", b);
            return n_out;
        }
        byte_tokens[n_byte++] = id;
    }

    if (n_byte == 0) return n_out;

    /* Apply greedy BPE merges */
    for (int iter = 0; iter < 500; iter++) {
        if (n_byte <= 1) break;

        int     best_idx  = -1;
        int32_t best_prio = INT32_MAX;
        char    best_merged[512];

        for (int p = 0; p < n_byte - 1; p++) {
            const char *left  = tok->vocab[byte_tokens[p]].str;
            const char *right = tok->vocab[byte_tokens[p + 1]].str;

            /* Find this pair in the merge list (linear scan by priority) */
            for (int32_t m = 0; m < tok->num_merges; m++) {
                if (strcmp(tok->merges[m].left, left) != 0) continue;
                if (strcmp(tok->merges[m].right, right) != 0) continue;
                if (tok->merges[m].priority < best_prio) {
                    best_prio = tok->merges[m].priority;
                    best_idx  = p;
                    int len = snprintf(best_merged, sizeof(best_merged), "%s%s", left, right);
                    if (len >= (int)sizeof(best_merged)) best_idx = -1;
                }
                break;
            }
        }

        if (best_idx < 0) break;

        int32_t merged_id = vocab_find(tok, best_merged);
        if (merged_id < 0) break;

        byte_tokens[best_idx] = merged_id;
        for (int p = best_idx + 1; p < n_byte - 1; p++)
            byte_tokens[p] = byte_tokens[p + 1];
        n_byte--;
    }

    /* Write output */
    for (int i = 0; i < n_byte && n_out < max_tokens; i++)
        token_ids[n_out++] = byte_tokens[i];

    return n_out;
}

/* ── Decode: reverse GPT-2 byte encoding ─────────────────────────────── */

static int32_t gpt2_codepoint_to_byte(int32_t cp) {
    /* Reverse of gpt2_byte_to_codepoint */
    /* Direct-mapped: 33-126, 161-172, 174-255 */
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 /* && cp <= 255 */))
        return cp;
    /* Extra bytes mapped to 256+ */
    if (cp >= 256 && cp < 256 + 68) {
        /* Count the nth unmapped byte value */
        int32_t target = cp - 256;
        int32_t n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
                continue;
            if (n == target) return b;
            n++;
        }
    }
    return -1;
}

/* Decode a single UTF-8 character to its code point, return bytes consumed */
static int32_t utf8_decode_one(const char *s, int32_t *cp) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) {
        *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        *cp = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        return 4;
    }
    *cp = 0xFFFD; return 1;
}

int32_t xllm_tokenizer_decode(XllmTokenizer *tok, const int32_t *token_ids,
                               int32_t num_tokens, char *out, int32_t max_len) {
    if (!tok || !token_ids || !out || max_len < 1) return 0;

    /* Accumulate raw bytes by reversing GPT-2 byte encoding */
    uint8_t byte_buf[4096];
    int32_t n_bytes = 0;

    for (int32_t i = 0; i < num_tokens; i++) {
        int32_t id = token_ids[i];
        if (id < 0 || id >= tok->vocab_size) continue;

        /* Skip BOS */
        if (id == tok->bos_token_id && i == 0) continue;

        const char *s = tok->vocab[id].str;
        if (!s || !*s) continue;

        /* Decode each Unicode character in the token string back to a byte */
        int32_t pos = 0;
        while (s[pos] && n_bytes < (int32_t)sizeof(byte_buf)) {
            int32_t cp;
            int32_t consumed = utf8_decode_one(s + pos, &cp);
            if (consumed <= 0) break;
            pos += consumed;

            int32_t b = gpt2_codepoint_to_byte(cp);
            if (b >= 0)
                byte_buf[n_bytes++] = (uint8_t)b;
        }
    }

    /* Now decode the byte sequence as UTF-8 */
    int32_t pos = 0;
    int32_t bpos = 0;
    out[0] = '\0';

    while (bpos < n_bytes && pos < max_len - 1) {
        int32_t cp;
        int32_t consumed = utf8_decode_one((const char *)(byte_buf + bpos), &cp);
        if (consumed <= 0) break;

        /* Skip the leading Ġ (U+0120) which GPT-2 uses as space prefix */
        if (cp == 0x0120) {
            if (pos > 0 && out[pos - 1] != ' ') {
                out[pos++] = ' ';
                out[pos] = '\0';
            }
            bpos += consumed;
            continue;
        }

        /* Re-encode the code point to UTF-8 into output */
        char tmp[8];
        int32_t n = codepoint_to_utf8(cp, tmp);
        if (n > 0 && pos + n < max_len) {
            memcpy(out + pos, tmp, (size_t)n);
            pos += n;
            out[pos] = '\0';
        }
        bpos += consumed;
    }

    out[pos] = '\0';
    return pos;
}
