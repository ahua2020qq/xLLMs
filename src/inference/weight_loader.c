/*
 * nxtLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * nxtLLM Weight Loader Implementation
 *
 * Reads the flat binary format produced by scripts/convert_gpt2_weights.py.
 * File layout:
 *   [config_json_len  : uint32 LE]
 *   [config_json       : char[config_json_len]]
 *   [tensor_count     : uint32 LE]
 *   For each tensor:
 *     [name_len        : uint32 LE]
 *     [name             : char[name_len]]
 *     [data_len        : uint32 LE]   (bytes, = nelem * sizeof(float))
 *     [data             : float32[data_len / 4]]
 */

#include "weight_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPT2_MAX_TENSOR_NAME 256

/* ── JSON mini-parser (only enough for our config) ─────────────────── */

static int json_get_int(const char *json, const char *key, int default_val) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, key);
    if (!pos) {
        /* try with quotes */
        pos = strstr(json, search);
    }
    if (!pos) return default_val;
    pos = strchr(pos, ':');
    if (!pos) return default_val;
    pos++;  /* skip ':' */
    while (*pos == ' ' || *pos == '\t') pos++;
    return (int)strtol(pos, NULL, 10);
}

static float json_get_float(const char *json, const char *key, float default_val) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return default_val;
    pos = strchr(pos, ':');
    if (!pos) return default_val;
    pos++;
    while (*pos == ' ' || *pos == '\t') pos++;
    return (float)strtod(pos, NULL);
}

static int parse_config(const char *json, Gpt2Config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->vocab_size     = json_get_int(json, "vocab_size", 50257);
    cfg->n_positions    = json_get_int(json, "n_positions", 1024);
    cfg->n_embd         = json_get_int(json, "n_embd", 768);
    cfg->n_layer        = json_get_int(json, "n_layer", 12);
    cfg->n_head         = json_get_int(json, "n_head", 12);
    cfg->n_inner        = json_get_int(json, "n_inner", 4 * cfg->n_embd);
    cfg->head_size      = cfg->n_embd / cfg->n_head;
    cfg->layer_norm_eps = json_get_float(json, "layer_norm_eps", 1e-5f);
    return 0;
}

/* ── Helpers ────────────────────────────────────────────────────────── */

static float *alloc_tensor(size_t nelem) {
    if (nelem == 0) return NULL;
    return (float *)calloc(nelem, sizeof(float));
}

/* ── Determine tensor shape from name ───────────────────────────────── */
/* GPT-2 weight naming convention:
 *   transformer.wte.weight           → [vocab_size, n_embd]
 *   transformer.wpe.weight           → [n_positions, n_embd]
 *   transformer.h.{l}.ln_1.weight    → [n_embd]
 *   transformer.h.{l}.ln_1.bias      → [n_embd]
 *   transformer.h.{l}.attn.c_attn.weight → [n_embd, 3*n_embd]
 *   transformer.h.{l}.attn.c_attn.bias   → [3*n_embd]
 *   transformer.h.{l}.attn.c_proj.weight → [n_embd, n_embd]
 *   transformer.h.{l}.attn.c_proj.bias   → [n_embd]
 *   transformer.h.{l}.ln_2.weight    → [n_embd]
 *   transformer.h.{l}.ln_2.bias      → [n_embd]
 *   transformer.h.{l}.mlp.c_fc.weight → [n_embd, n_inner]
 *   transformer.h.{l}.mlp.c_fc.bias  → [n_inner]
 *   transformer.h.{l}.mlp.c_proj.weight → [n_inner, n_embd]
 *   transformer.h.{l}.mlp.c_proj.bias → [n_embd]
 *   transformer.ln_f.weight          → [n_embd]
 *   transformer.ln_f.bias            → [n_embd]
 */

static int parse_layer_idx(const char *name) {
    /* Find ".h.<N>." in the name */
    const char *p = strstr(name, ".h.");
    if (!p) return -1;
    p += 3;  /* skip ".h." */
    return (int)strtol(p, NULL, 10);
}

static int assign_weight(Gpt2Weights *w, const Gpt2Config *cfg,
                         const char *name, const float *data) {
    int layer = parse_layer_idx(name);

    if (strstr(name, "transformer.wte.weight")) {
        memcpy(w->wte, data, (size_t)cfg->vocab_size * cfg->n_embd * sizeof(float));
        return 0;
    }
    if (strstr(name, "transformer.wpe.weight")) {
        memcpy(w->wpe, data, (size_t)cfg->n_positions * cfg->n_embd * sizeof(float));
        return 0;
    }
    if (strstr(name, "transformer.ln_f.weight")) {
        memcpy(w->ln_f_weight, data, cfg->n_embd * sizeof(float));
        return 0;
    }
    if (strstr(name, "transformer.ln_f.bias")) {
        memcpy(w->ln_f_bias, data, cfg->n_embd * sizeof(float));
        return 0;
    }

    if (layer < 0 || layer >= cfg->n_layer) return -1;

    if (strstr(name, ".ln_1.weight")) {
        memcpy(w->ln_1_weight[layer], data, cfg->n_embd * sizeof(float));
    } else if (strstr(name, ".ln_1.bias")) {
        memcpy(w->ln_1_bias[layer], data, cfg->n_embd * sizeof(float));
    } else if (strstr(name, ".attn.c_attn.weight")) {
        memcpy(w->attn_c_attn_w[layer], data,
               (size_t)cfg->n_embd * 3 * cfg->n_embd * sizeof(float));
    } else if (strstr(name, ".attn.c_attn.bias")) {
        memcpy(w->attn_c_attn_b[layer], data, 3 * cfg->n_embd * sizeof(float));
    } else if (strstr(name, ".attn.c_proj.weight")) {
        memcpy(w->attn_c_proj_w[layer], data,
               (size_t)cfg->n_embd * cfg->n_embd * sizeof(float));
    } else if (strstr(name, ".attn.c_proj.bias")) {
        memcpy(w->attn_c_proj_b[layer], data, cfg->n_embd * sizeof(float));
    } else if (strstr(name, ".ln_2.weight")) {
        memcpy(w->ln_2_weight[layer], data, cfg->n_embd * sizeof(float));
    } else if (strstr(name, ".ln_2.bias")) {
        memcpy(w->ln_2_bias[layer], data, cfg->n_embd * sizeof(float));
    } else if (strstr(name, ".mlp.c_fc.weight")) {
        memcpy(w->mlp_c_fc_w[layer], data,
               (size_t)cfg->n_embd * cfg->n_inner * sizeof(float));
    } else if (strstr(name, ".mlp.c_fc.bias")) {
        memcpy(w->mlp_c_fc_b[layer], data, cfg->n_inner * sizeof(float));
    } else if (strstr(name, ".mlp.c_proj.weight")) {
        memcpy(w->mlp_c_proj_w[layer], data,
               (size_t)cfg->n_inner * cfg->n_embd * sizeof(float));
    } else if (strstr(name, ".mlp.c_proj.bias")) {
        memcpy(w->mlp_c_proj_b[layer], data, cfg->n_embd * sizeof(float));
    }
    return 0;
}

/* ── Public API ─────────────────────────────────────────────────────── */

Gpt2Weights *gpt2_weights_alloc(const Gpt2Config *cfg) {
    Gpt2Weights *w = (Gpt2Weights *)calloc(1, sizeof(Gpt2Weights));
    if (!w) return NULL;

    w->wte = alloc_tensor((size_t)cfg->vocab_size * cfg->n_embd);
    w->wpe = alloc_tensor((size_t)cfg->n_positions * cfg->n_embd);
    w->ln_f_weight = alloc_tensor(cfg->n_embd);
    w->ln_f_bias   = alloc_tensor(cfg->n_embd);

    #define ALLOC_LAYER_ARRAY(field, size) do { \
        w->field = (float **)calloc(cfg->n_layer, sizeof(float *)); \
        for (int i = 0; i < cfg->n_layer; i++) \
            w->field[i] = alloc_tensor(size); \
    } while (0)

    ALLOC_LAYER_ARRAY(ln_1_weight,   cfg->n_embd);
    ALLOC_LAYER_ARRAY(ln_1_bias,     cfg->n_embd);
    ALLOC_LAYER_ARRAY(attn_c_attn_w, (size_t)cfg->n_embd * 3 * cfg->n_embd);
    ALLOC_LAYER_ARRAY(attn_c_attn_b, 3 * cfg->n_embd);
    ALLOC_LAYER_ARRAY(attn_c_proj_w, (size_t)cfg->n_embd * cfg->n_embd);
    ALLOC_LAYER_ARRAY(attn_c_proj_b, cfg->n_embd);
    ALLOC_LAYER_ARRAY(ln_2_weight,   cfg->n_embd);
    ALLOC_LAYER_ARRAY(ln_2_bias,     cfg->n_embd);
    ALLOC_LAYER_ARRAY(mlp_c_fc_w,    (size_t)cfg->n_embd * cfg->n_inner);
    ALLOC_LAYER_ARRAY(mlp_c_fc_b,    cfg->n_inner);
    ALLOC_LAYER_ARRAY(mlp_c_proj_w,  (size_t)cfg->n_inner * cfg->n_embd);
    ALLOC_LAYER_ARRAY(mlp_c_proj_b,  cfg->n_embd);

    #undef ALLOC_LAYER_ARRAY
    return w;
}

Gpt2Weights *gpt2_weights_load(const char *path, Gpt2Config *cfg_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "weight_loader: cannot open '%s'\n", path);
        return NULL;
    }

    /* ── Read config JSON ─────────────────────────────────────────── */
    uint32_t config_len;
    if (fread(&config_len, sizeof(config_len), 1, f) != 1) {
        fprintf(stderr, "weight_loader: failed to read config length\n");
        fclose(f); return NULL;
    }
    char *config_json = (char *)malloc(config_len + 1);
    if (!config_json || fread(config_json, 1, config_len, f) != config_len) {
        fprintf(stderr, "weight_loader: failed to read config JSON\n");
        free(config_json); fclose(f); return NULL;
    }
    config_json[config_len] = '\0';

    Gpt2Config cfg;
    parse_config(config_json, &cfg);
    free(config_json);

    if (cfg_out) *cfg_out = cfg;

    /* ── Allocate weight structure ────────────────────────────────── */
    Gpt2Weights *w = gpt2_weights_alloc(&cfg);
    if (!w) { fclose(f); return NULL; }

    /* ── Read tensors ─────────────────────────────────────────────── */
    uint32_t tensor_count;
    if (fread(&tensor_count, sizeof(tensor_count), 1, f) != 1) {
        fprintf(stderr, "weight_loader: failed to read tensor count\n");
        gpt2_weights_free(w); fclose(f); return NULL;
    }

    char   name_buf[GPT2_MAX_TENSOR_NAME];
    float *data_buf = NULL;
    size_t data_buf_size = 0;

    for (uint32_t i = 0; i < tensor_count; i++) {
        /* name */
        uint32_t name_len;
        if (fread(&name_len, sizeof(name_len), 1, f) != 1) goto fail;
        if (name_len >= GPT2_MAX_TENSOR_NAME) goto fail;
        if (fread(name_buf, 1, name_len, f) != name_len) goto fail;
        name_buf[name_len] = '\0';

        /* data */
        uint32_t data_len;
        if (fread(&data_len, sizeof(data_len), 1, f) != 1) goto fail;
        if (data_len > data_buf_size) {
            data_buf_size = data_len;
            float *new_buf = (float *)realloc(data_buf, data_len);
            if (!new_buf) goto fail;
            data_buf = new_buf;
        }
        if (fread(data_buf, 1, data_len, f) != data_len) goto fail;

        assign_weight(w, &cfg, name_buf, data_buf);
    }

    free(data_buf);
    fclose(f);
    return w;

fail:
    fprintf(stderr, "weight_loader: error reading tensor %u\n", i);
    free(data_buf);
    gpt2_weights_free(w);
    fclose(f);
    return NULL;
}

void gpt2_weights_free(Gpt2Weights *w) {
    if (!w) return;
    free(w->wte);
    free(w->wpe);
    free(w->ln_f_weight);
    free(w->ln_f_bias);

    #define FREE_LAYER_ARRAY(field) do { \
        if (w->field) { \
            for (int i = 0; i < 12; i++) free(w->field[i]); \
            free(w->field); \
        } \
    } while (0)

    /* Use fixed max in case w was allocated with unknown n_layer */
    FREE_LAYER_ARRAY(ln_1_weight);
    FREE_LAYER_ARRAY(ln_1_bias);
    FREE_LAYER_ARRAY(attn_c_attn_w);
    FREE_LAYER_ARRAY(attn_c_attn_b);
    FREE_LAYER_ARRAY(attn_c_proj_w);
    FREE_LAYER_ARRAY(attn_c_proj_b);
    FREE_LAYER_ARRAY(ln_2_weight);
    FREE_LAYER_ARRAY(ln_2_bias);
    FREE_LAYER_ARRAY(mlp_c_fc_w);
    FREE_LAYER_ARRAY(mlp_c_fc_b);
    FREE_LAYER_ARRAY(mlp_c_proj_w);
    FREE_LAYER_ARRAY(mlp_c_proj_b);

    #undef FREE_LAYER_ARRAY

    free(w);
}
