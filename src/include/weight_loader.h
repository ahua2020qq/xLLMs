/*
 * nxtLLM Weight Loader — GPT-2 Model Weights
 *
 * Loads GPT-2 weights from a flat binary format produced by
 * scripts/convert_gpt2_weights.py (which reads Hugging Face
 * safetensors / PyTorch .bin files).
 */

#ifndef NXTLLM_WEIGHT_LOADER_H_
#define NXTLLM_WEIGHT_LOADER_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── GPT-2 model configuration ─────────────────────────────────────── */
typedef struct {
    int32_t vocab_size;       /* 50257 */
    int32_t n_positions;      /* 1024 */
    int32_t n_embd;           /* 768 (small), 1024 (medium), 1600 (large) */
    int32_t n_layer;          /* 12 */
    int32_t n_head;           /* 12 */
    int32_t n_inner;          /* 4 * n_embd (MLP hidden dim) */
    int32_t head_size;        /* n_embd / n_head = 64 */
    float   layer_norm_eps;   /* 1e-5 */
} Gpt2Config;

/* ── GPT-2 weight tensors (all float32, row-major) ────────────────── */
typedef struct {
    /* Token + position embeddings */
    float *wte;   /* [vocab_size, n_embd] */
    float *wpe;   /* [n_positions, n_embd] */

    /* Per-transformer-block weights */
    float **ln_1_weight;     /* [n_layer][n_embd] */
    float **ln_1_bias;       /* [n_layer][n_embd] */
    float **attn_c_attn_w;   /* [n_layer][n_embd, 3 * n_embd] */
    float **attn_c_attn_b;   /* [n_layer][3 * n_embd] */
    float **attn_c_proj_w;   /* [n_layer][n_embd, n_embd] */
    float **attn_c_proj_b;   /* [n_layer][n_embd] */
    float **ln_2_weight;     /* [n_layer][n_embd] */
    float **ln_2_bias;       /* [n_layer][n_embd] */
    float **mlp_c_fc_w;      /* [n_layer][n_embd, n_inner] */
    float **mlp_c_fc_b;      /* [n_layer][n_inner] */
    float **mlp_c_proj_w;    /* [n_layer][n_inner, n_embd] */
    float **mlp_c_proj_b;    /* [n_layer][n_embd] */

    /* Final layer norm */
    float *ln_f_weight;      /* [n_embd] */
    float *ln_f_bias;        /* [n_embd] */
} Gpt2Weights;

/* ── API ───────────────────────────────────────────────────────────── */

/** Load GPT-2 weights from a flat binary file produced by the Python converter.
 *  File layout: [config_json_len(4B)] [config_json] [tensor_count(4B)]
 *               (tensor_name_len(4B) tensor_name tensor_len(4B) tensor_data)*
 *
 *  Returns NULL on failure; call gpt2_weights_free() to release.
 */
Gpt2Weights *gpt2_weights_load(const char *path, Gpt2Config *cfg_out);

/** Free all memory associated with the weight structure. */
void gpt2_weights_free(Gpt2Weights *w);

/** Convenience: allocate and initialise a Gpt2Weights structure for the given config. */
Gpt2Weights *gpt2_weights_alloc(const Gpt2Config *cfg);

#ifdef __cplusplus
}
#endif

#endif /* NXTLLM_WEIGHT_LOADER_H_ */
