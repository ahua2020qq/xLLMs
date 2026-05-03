/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

typedef struct {
    int id;
    char* token;
} BPEToken;

typedef struct {
    BPEToken* vocab;
    int vocab_size;
    int eos_token_id;
    int pad_token_id;
} BPETokenizer;

#define VOCAB_SIZE 256

static const char* default_vocab[VOCAB_SIZE] = {
    [0] = "<PAD>", [1] = "<EOS>", [2] = "<UNK>",
    [32] = " ", [33] = "!", [34] = "\"", [35] = "#", [36] = "$",
    [37] = "%", [38] = "&", [39] = "'", [40] = "(", [41] = ")",
    [42] = "*", [43] = "+", [44] = ",", [45] = "-", [46] = ".",
    [47] = "/",
    [48] = "0", [49] = "1", [50] = "2", [51] = "3", [52] = "4",
    [53] = "5", [54] = "6", [55] = "7", [56] = "8", [57] = "9",
    [58] = ":", [59] = ";", [60] = "<", [61] = "=", [62] = ">",
    [63] = "?", [64] = "@",
    [65] = "A", [66] = "B", [67] = "C", [68] = "D", [69] = "E",
    [70] = "F", [71] = "G", [72] = "H", [73] = "I", [74] = "J",
    [75] = "K", [76] = "L", [77] = "M", [78] = "N", [79] = "O",
    [80] = "P", [81] = "Q", [82] = "R", [83] = "S", [84] = "T",
    [85] = "U", [86] = "V", [87] = "W", [88] = "X", [89] = "Y",
    [90] = "Z",
    [91] = "[", [92] = "\\", [93] = "]", [94] = "^", [95] = "_",
    [96] = "`",
    [97] = "a", [98] = "b", [99] = "c", [100] = "d", [101] = "e",
    [102] = "f", [103] = "g", [104] = "h", [105] = "i", [106] = "j",
    [107] = "k", [108] = "l", [109] = "m", [110] = "n", [111] = "o",
    [112] = "p", [113] = "q", [114] = "r", [115] = "s", [116] = "t",
    [117] = "u", [118] = "v", [119] = "w", [120] = "x", [121] = "y",
    [122] = "z",
    [123] = "{", [124] = "|", [125] = "}", [126] = "~",
    [128] = "Hello", [129] = "world", [130] = "GPT",
    [131] = "the", [132] = " is", [133] = " a",
};

static BPETokenizer* bpe_tokenizer_create(void) {
    BPETokenizer* t = malloc(sizeof(BPETokenizer));
    t->vocab_size = VOCAB_SIZE;
    t->eos_token_id = 1;
    t->pad_token_id = 0;
    t->vocab = calloc(VOCAB_SIZE, sizeof(BPEToken));

    for (int i = 0; i < VOCAB_SIZE; i++) {
        t->vocab[i].id = i;
        if (default_vocab[i]) {
            t->vocab[i].token = strdup(default_vocab[i]);
        } else {
            t->vocab[i].token = NULL;
        }
    }
    return t;
}

static int bpe_tokenizer_encode(BPETokenizer* t, const char* text, int* token_ids, int max_len) {
    int count = 0;
    int text_len = strlen(text);
    int pos = 0;

    while (pos < text_len && count < max_len) {
        int best_len = 0;
        int best_id = 2; // <UNK>

        for (int v = 0; v < t->vocab_size; v++) {
            if (t->vocab[v].token == NULL) continue;
            int tok_len = strlen(t->vocab[v].token);
            if (tok_len > best_len && tok_len <= text_len - pos) {
                if (strncmp(text + pos, t->vocab[v].token, tok_len) == 0) {
                    best_len = tok_len;
                    best_id = v;
                }
            }
        }

        token_ids[count++] = best_id;
        pos += (best_len > 0) ? best_len : 1;
    }

    if (count < max_len) {
        token_ids[count++] = t->eos_token_id;
    }

    return count;
}

static char* bpe_tokenizer_decode(BPETokenizer* t, const int* token_ids, int num_tokens) {
    int total_len = 0;
    for (int i = 0; i < num_tokens; i++) {
        int id = token_ids[i];
        if (id > 0 && id < t->vocab_size && t->vocab[id].token) {
            total_len += strlen(t->vocab[id].token);
        } else if (id < 128 && id >= 0) {
            total_len += 1;
        } else {
            total_len += 4; // "<UNK>"
        }
    }

    char* result = calloc(total_len + 1, 1);
    int pos = 0;
    for (int i = 0; i < num_tokens; i++) {
        int id = token_ids[i];
        if (id == t->pad_token_id) continue;
        if (id == t->eos_token_id) break;

        const char* tok = NULL;
        if (id > 0 && id < t->vocab_size && t->vocab[id].token) {
            tok = t->vocab[id].token;
        } else if (id < 128 && id >= 0) {
            result[pos++] = (char)id;
            continue;
        } else {
            tok = "<UNK>";
        }
        int l = strlen(tok);
        memcpy(result + pos, tok, l);
        pos += l;
    }
    result[pos] = '\0';
    return result;
}

static void bpe_tokenizer_free(BPETokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i].token);
    }
    free(t->vocab);
    free(t);
}
