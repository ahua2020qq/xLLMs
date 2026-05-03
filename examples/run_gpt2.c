/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 64
#define NUM_HEADS 4
#define NUM_LAYERS 2
#define FFN_DIM 256
#define MAX_SEQ_LEN 16
#define PAGE_SIZE 4
#define VOCAB_SIZE 256

/* operator implementations */
static void layer_norm(const float* input, const float* gamma,
                       const float* beta, int dim, float eps, float* output) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < dim; i++) mean += input[i];
    mean /= dim;
    for (int i = 0; i < dim; i++) {
        float d = input[i] - mean;
        var += d * d;
    }
    var /= dim;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++)
        output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
}

static void gelu(const float* input, int dim, float* output) {
    for (int i = 0; i < dim; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

static void paged_attention(const float* query, const float* key_cache,
                             const float* value_cache, const int* page_table,
                             int seq_len, int num_heads, int head_dim,
                             int page_size, float* output) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int page_entries = page_size * num_heads * head_dim;

    for (int h = 0; h < num_heads; h++) {
        float* out_h = output + h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));

        float sum_weights = 0.0f;
        float* scores = malloc(seq_len * sizeof(float));

        for (int t = 0; t < seq_len; t++) {
            int page_idx = page_table[t];
            int slot = t % page_size;
            const float* k = key_cache + page_idx * page_entries
                           + slot * num_heads * head_dim + h * head_dim;
            const float* q_h = query + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += q_h[d] * k[d];
            scores[t] = expf(dot * scale);
            sum_weights += scores[t];
        }

        for (int t = 0; t < seq_len; t++) {
            int page_idx = page_table[t];
            int slot = t % page_size;
            const float* v = value_cache + page_idx * page_entries
                           + slot * num_heads * head_dim + h * head_dim;
            float w = scores[t] / sum_weights;
            for (int d = 0; d < head_dim; d++) out_h[d] += w * v[d];
        }

        free(scores);
    }
}

/* BPE tokenizer (inline for single-file build) */
typedef struct { int id; char* token; } BPEToken;
typedef struct { BPEToken* vocab; int vocab_size; int eos_token_id; int pad_token_id; } BPETokenizer;

static const char* vocab_table[VOCAB_SIZE] = {
    [0] = "<PAD>", [1] = "<EOS>", [2] = "<UNK>", [32] = " ",
    [33] = "!", [44] = ",", [46] = ".", [63] = "?",
    [65]="A",[66]="B",[67]="C",[68]="D",[69]="E",[70]="F",[71]="G",[72]="H",
    [73]="I",[74]="J",[75]="K",[76]="L",[77]="M",[78]="N",[79]="O",[80]="P",
    [81]="Q",[82]="R",[83]="S",[84]="T",[85]="U",[86]="V",[87]="W",[88]="X",
    [89]="Y",[90]="Z",
    [97]="a",[98]="b",[99]="c",[100]="d",[101]="e",[102]="f",[103]="g",[104]="h",
    [105]="i",[106]="j",[107]="k",[108]="l",[109]="m",[110]="n",[111]="o",[112]="p",
    [113]="q",[114]="r",[115]="s",[116]="t",[117]="u",[118]="v",[119]="w",[120]="x",
    [121]="y",[122]="z",
    [128]="Hello",[129]="world",[130]="GPT",
    [131]="the",[132]=" is",[133]=" a",
};

static BPETokenizer* bpe_create(void) {
    BPETokenizer* t = calloc(1, sizeof(BPETokenizer));
    t->vocab_size = VOCAB_SIZE;
    t->eos_token_id = 1;
    t->pad_token_id = 0;
    t->vocab = calloc(VOCAB_SIZE, sizeof(BPEToken));
    for (int i = 0; i < VOCAB_SIZE; i++) {
        t->vocab[i].id = i;
        if (vocab_table[i]) t->vocab[i].token = strdup(vocab_table[i]);
    }
    return t;
}

static int bpe_encode(BPETokenizer* t, const char* text, int* ids, int max_len) {
    int count = 0, pos = 0, text_len = strlen(text);
    while (pos < text_len && count < max_len) {
        int best_len = 0, best_id = 2;
        for (int v = 0; v < t->vocab_size; v++) {
            if (!t->vocab[v].token) continue;
            int tl = strlen(t->vocab[v].token);
            if (tl > best_len && tl <= text_len - pos &&
                strncmp(text + pos, t->vocab[v].token, tl) == 0) {
                best_len = tl; best_id = v;
            }
        }
        ids[count++] = best_id;
        pos += best_len > 0 ? best_len : 1;
    }
    if (count < max_len) ids[count++] = t->eos_token_id;
    return count;
}

static char* bpe_decode(BPETokenizer* t, const int* ids, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id == t->eos_token_id) break;
        if (id == t->pad_token_id) continue;
        if (id > 0 && id < t->vocab_size && t->vocab[id].token)
            total += strlen(t->vocab[id].token);
        else total += (id < 128 && id >= 0) ? 1 : 4;
    }
    char* out = calloc(total + 1, 1);
    int pos = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id == t->pad_token_id) continue;
        if (id == t->eos_token_id) break;
        if (id > 0 && id < t->vocab_size && t->vocab[id].token) {
            int l = strlen(t->vocab[id].token);
            memcpy(out + pos, t->vocab[id].token, l);
            pos += l;
        } else if (id < 128 && id >= 0) {
            out[pos++] = (char)id;
        } else {
            memcpy(out + pos, "<UNK>", 5); pos += 5;
        }
    }
    return out;
}

static void bpe_free(BPETokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i].token);
    free(t->vocab); free(t);
}

static int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

int main(void) {
    printf("=== GPT-2 Inference Demo (issue #1) ===\n\n");

    /* Create tokenizer */
    BPETokenizer* tok = bpe_create();
    printf("[1] BPE tokenizer created, vocab_size=%d\n", tok->vocab_size);

    /* Encode input */
    const char* input_text = "Hello";
    int token_ids[16];
    int n_tokens = bpe_encode(tok, input_text, token_ids, 16);
    printf("[2] Input: \"%s\" -> token IDs: ", input_text);
    for (int i = 0; i < n_tokens; i++) printf("%d ", token_ids[i]);
    printf("\n");

    /* Allocate weights */
    srand(42);
    float* embed_w = malloc(VOCAB_SIZE * DIM * sizeof(float));
    float* lm_head_w = malloc(DIM * VOCAB_SIZE * sizeof(float));
    for (int i = 0; i < VOCAB_SIZE * DIM; i++)
        embed_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < DIM * VOCAB_SIZE; i++)
        lm_head_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;

    /* KV cache */
    int max_pages = (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
    size_t cache_bytes = NUM_LAYERS * max_pages * PAGE_SIZE * NUM_HEADS * (DIM/NUM_HEADS) * sizeof(float);
    float* key_cache = calloc(1, cache_bytes);
    float* value_cache = calloc(1, cache_bytes);
    int* page_table = malloc(NUM_LAYERS * MAX_SEQ_LEN * sizeof(int));
    for (int i = 0; i < NUM_LAYERS * MAX_SEQ_LEN; i++) page_table[i] = i % max_pages;

    printf("[3] Model: dim=%d, num_heads=%d, num_layers=%d, ffn_dim=%d\n",
           DIM, NUM_HEADS, NUM_LAYERS, FFN_DIM);
    printf("    KV cache: %zu bytes per cache\n", cache_bytes);

    /* Autoregressive decode */
    int n_generate = 3;
    int current_token = token_ids[0];
    float* hidden = calloc(DIM, sizeof(float));
    float* logits = calloc(VOCAB_SIZE, sizeof(float));
    float* hidden_in = calloc(DIM, sizeof(float));
    float* hidden_out = calloc(DIM, sizeof(float));

    float* ln1_g = calloc(DIM, sizeof(float));
    float* ln1_b = calloc(DIM, sizeof(float));
    float* ln2_g = calloc(DIM, sizeof(float));
    float* ln2_b = calloc(DIM, sizeof(float));
    float* qkv_w = calloc(DIM * 3 * DIM, sizeof(float));
    float* proj_w = calloc(DIM * DIM, sizeof(float));
    float* ffn_w1 = calloc(DIM * FFN_DIM, sizeof(float));
    float* ffn_w2 = calloc(FFN_DIM * DIM, sizeof(float));
    for (int i = 0; i < DIM; i++) ln1_g[i] = ln2_g[i] = 1.0f;
    for (int i = 0; i < DIM*3*DIM; i++) qkv_w[i] = ((float)rand()/RAND_MAX-0.5f)*0.02f;
    for (int i = 0; i < DIM*DIM; i++) proj_w[i] = ((float)rand()/RAND_MAX-0.5f)*0.02f;
    for (int i = 0; i < DIM*FFN_DIM; i++) ffn_w1[i] = ((float)rand()/RAND_MAX-0.5f)*0.02f;
    for (int i = 0; i < FFN_DIM*DIM; i++) ffn_w2[i] = ((float)rand()/RAND_MAX-0.5f)*0.02f;

    printf("\n[4] Starting autoregressive decode (%d steps)...\n", n_generate);
    printf("    Prompt: \"%s\"\n", input_text);

    for (int step = 0; step < n_generate; step++) {
        // Embed
        for (int i = 0; i < DIM; i++) hidden[i] = embed_w[current_token * DIM + i];
        memcpy(hidden_in, hidden, DIM * sizeof(float));

        // Simulate transformer layers
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            // LN1 -> QKV -> Attention -> Proj -> Residual
            float* ln1 = calloc(DIM, sizeof(float));
            layer_norm(hidden_in, ln1_g, ln1_b, DIM, 1e-5f, ln1);

            float* qkv = calloc(3*DIM, sizeof(float));
            for (int i = 0; i < 3*DIM; i++) {
                float s = 0; for (int j = 0; j < DIM; j++) s += ln1[j] * qkv_w[j*(3*DIM)+i];
                qkv[i] = s;
            }

            float* q = calloc(NUM_HEADS*(DIM/NUM_HEADS), sizeof(float));
            for (int h = 0; h < NUM_HEADS; h++)
                for (int d = 0; d < DIM/NUM_HEADS; d++)
                    q[h*(DIM/NUM_HEADS)+d] = qkv[h*(DIM/NUM_HEADS)+d];

            int seq_now = step + 1;
            float* attn_out = calloc(DIM, sizeof(float));
            paged_attention(q, key_cache, value_cache, page_table,
                           seq_now, NUM_HEADS, DIM/NUM_HEADS, PAGE_SIZE, attn_out);

            float* proj = calloc(DIM, sizeof(float));
            for (int i = 0; i < DIM; i++) {
                float s = 0; for (int j = 0; j < DIM; j++) s += attn_out[j]*proj_w[j*DIM+i];
                proj[i] = s;
            }
            for (int i = 0; i < DIM; i++) hidden_in[i] += proj[i];

            // LN2 -> FFN -> GELU -> Proj -> Residual
            float* ln2 = calloc(DIM, sizeof(float));
            layer_norm(hidden_in, ln2_g, ln2_b, DIM, 1e-5f, ln2);

            float* ffn_h = calloc(FFN_DIM, sizeof(float));
            for (int i = 0; i < FFN_DIM; i++) {
                float s = 0; for (int j = 0; j < DIM; j++) s += ln2[j]*ffn_w1[j*FFN_DIM+i];
                ffn_h[i] = s;
            }
            gelu(ffn_h, FFN_DIM, ffn_h);

            float* ffn_o = calloc(DIM, sizeof(float));
            for (int i = 0; i < DIM; i++) {
                float s = 0; for (int j = 0; j < FFN_DIM; j++) s += ffn_h[j]*ffn_w2[j*DIM+i];
                ffn_o[i] = s;
            }
            for (int i = 0; i < DIM; i++) hidden_out[i] = hidden_in[i] + ffn_o[i];
            memcpy(hidden_in, hidden_out, DIM * sizeof(float));

            free(ln1); free(qkv); free(q); free(attn_out); free(proj);
            free(ln2); free(ffn_h); free(ffn_o);
        }

        memcpy(hidden, hidden_out, DIM * sizeof(float));

        // LM head
        for (int v = 0; v < VOCAB_SIZE; v++) {
            float s = 0;
            for (int d = 0; d < DIM; d++) s += hidden[d] * lm_head_w[d*VOCAB_SIZE+v];
            logits[v] = s;
        }

        int next = argmax(logits, VOCAB_SIZE);
        char* decoded = bpe_decode(tok, &next, 1);
        printf("    Step %d: token_id=%d -> \"%s\"\n", step + 1, next, decoded);

        current_token = next;
        free(decoded);
    }

    printf("\n[5] Decode complete.\n");
    printf("=== DONE ===\n");

    /* Cleanup */
    free(embed_w); free(lm_head_w); free(key_cache); free(value_cache);
    free(page_table); free(hidden); free(logits); free(hidden_in); free(hidden_out);
    free(ln1_g); free(ln1_b); free(ln2_g); free(ln2_b);
    free(qkv_w); free(proj_w); free(ffn_w1); free(ffn_w2);
    bpe_free(tok);

    return 0;
}
