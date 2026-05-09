/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * text_gen — End-to-end text generation demo.
 * Loads a GGUF model, tokenizes a prompt, runs a full transformer forward
 * pass through all layers, samples the next token, and decodes it.
 *
 * Usage: ./text_gen <gguf_path> ["prompt text"]
 */

#include "model_loader.h"
#include "weight_loader.h"
#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#define MAX_NEW_TOKENS 256
#define MAX_SEQ_LEN    512
#define SKIP_LAYERS    0   /* TEMP: test embedding+norm+lm_head only */
#define SKIP_QK_NORM   0   /* TEMP: disable Q/K norm to test */
#define DEBUG_LOGITS   1   /* dump detailed logits for known tokens */
static int g_max_layers = 0;  /* 0=unlimited; set via -DMAXL=N for bisection */

/* ═══════════════════════════════════════════════════════════════════════
 * Q4_K and Q6_K dequantization (CPU reference)
 * Based on llama.cpp ggml-quants.c
 * ═══════════════════════════════════════════════════════════════════════ */

static void dequantize_q4_k(const uint8_t *x, float *y, int64_t n) {
    /* Q4_K: 256-element super-blocks, 144 bytes each.
     * Layout: d(2) + dmin(2) + scales(12) + qs(128)
     * 8 sub-blocks of 32 elements, each with own scale and min.
     * Scales are 6-bit values packed across 12 bytes (3 groups of 4).
     * Reference: llama.cpp ggml-quants.c */
    const int nb = 256;
    for (int64_t ib = 0; ib < n; ib += nb) {
        const uint8_t *blk = x + (ib / nb) * 144;

        float d, dmin;
        {
            uint16_t dh = blk[0] | ((uint16_t)blk[1] << 8);
            uint32_t s = (dh >> 15) & 1, e = (dh >> 10) & 0x1F, m = dh & 0x3FF;
            if (e == 0) d = (s ? -1.0f : 1.0f) * (float)m * 5.96046448e-8f;
            else if (e == 0x1F) d = (m == 0) ? INFINITY : NAN;
            else { uint32_t f32 = (s << 31) | ((e + 112) << 23) | (m << 13); memcpy(&d, &f32, sizeof(float)); }
        }
        {
            uint16_t dh = blk[2] | ((uint16_t)blk[3] << 8);
            uint32_t s = (dh >> 15) & 1, e = (dh >> 10) & 0x1F, m = dh & 0x3FF;
            if (e == 0) dmin = (s ? -1.0f : 1.0f) * (float)m * 5.96046448e-8f;
            else if (e == 0x1F) dmin = (m == 0) ? INFINITY : NAN;
            else { uint32_t f32 = (s << 31) | ((e + 112) << 23) | (m << 13); memcpy(&dmin, &f32, sizeof(float)); }
        }

        /* Unpack 12 scale bytes → 8 scales + 8 mins (6 bits each).
         * Sub-blocks 0..3: low 6 bits of bytes 0..3 / 4..7 (llama.cpp get_scale_min_k4 j<4).
         * Sub-blocks 4..7: combined from bytes 8..11 + high bits of bytes 0..3 / 4..7 (j>=4). */
        const uint8_t *sc_raw = blk + 4;
        float sc[8], mn[8];
        for (int j = 0; j < 4; j++) {
            sc[j] = d    * (float)(sc_raw[j] & 0x3F);
            mn[j] = dmin * (float)(sc_raw[j + 4] & 0x3F);
        }
        for (int j = 0; j < 4; j++) {
            sc[j + 4] = d    * (float)((sc_raw[j + 8] & 0x0F) | ((sc_raw[j] >> 6) << 4));
            mn[j + 4] = dmin * (float)((sc_raw[j + 8] >> 4)  | ((sc_raw[j + 4] >> 6) << 4));
        }

        /* Nibble layout: interleaved per group (low, high, low, high, ...).
         * group=s/2, even s=low nibble, odd s=high nibble of same group. */
        const uint8_t *qs = blk + 16;
        for (int s = 0; s < 8; s++) {
            int group = s / 2;
            int is_lo = (s % 2 == 0);
            for (int j = 0; j < 32; j++) {
                int idx = ib + s * 32 + j;
                if (idx >= n) return;
                int q = is_lo ? (qs[group * 32 + j] & 0x0F)
                              : (qs[group * 32 + j] >> 4);
                y[idx] = sc[s] * (float)q - mn[s];
            }
        }
    }
}

static void dequantize_q6_k(const uint8_t *x, float *y, int64_t n) {
    /* Q6_K: 256-element super-blocks, 210 bytes each.
     * Layout (per block): ql[128] + qh[64] + scales[16] + d[2]
     * Output: 16 sub-blocks of 16 elements each, scale per sub-block.
     * Sub-block index = q_idx*2 + is, where q_idx = 0..3 for q1..q4,
     * is = 0 for lanes 0..15, is = 1 for lanes 16..31.
     * Scale for sub-block s: sc[half*8 + s] */
    const int nb = 256;
    for (int64_t ib = 0; ib < n; ib += nb) {
        const uint8_t *blk = x + (ib / nb) * 210;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + 128;
        const int8_t  *sc  = (const int8_t *)(blk + 192);

        uint16_t dh_raw = blk[208] | ((uint16_t)blk[209] << 8);
        float d;
        {
            uint32_t sign = (dh_raw >> 15) & 1, e = (dh_raw >> 10) & 0x1F, m = dh_raw & 0x3FF;
            if (e == 0) d = (sign ? -1.0f : 1.0f) * (float)m * 5.96046448e-8f;
            else if (e == 0x1F) d = (m == 0) ? INFINITY : NAN;
            else { uint32_t f32 = (sign << 31) | ((e + 112) << 23) | (m << 13); memcpy(&d, &f32, sizeof(float)); }
        }

        for (int half = 0; half < 2; half++) {
            int ql_off = half * 64;
            int qh_off = half * 32;
            int base = ib + half * 128;
            for (int is = 0; is < 2; is++) {
                int lo = is * 16;
                for (int li = 0; li < 16; li++) {
                    int l = lo + li;
                    if (base + l >= n) return;
                    int q1 = (ql[ql_off + l] & 0xF)
                           | (((qh[qh_off + l] >> 0) & 3) << 4);
                    int q2 = (ql[ql_off + 32 + l] & 0xF)
                           | (((qh[qh_off + l] >> 2) & 3) << 4);
                    int q3 = (ql[ql_off + l] >> 4)
                           | (((qh[qh_off + l] >> 4) & 3) << 4);
                    int q4 = (ql[ql_off + 32 + l] >> 4)
                           | (((qh[qh_off + l] >> 6) & 3) << 4);
                    y[base + lo + li]       = d * (float)sc[half*8 + 0 + is] * (float)(q1 - 32);
                    y[base + 32 + lo + li]  = d * (float)sc[half*8 + 2 + is] * (float)(q2 - 32);
                    y[base + 64 + lo + li]  = d * (float)sc[half*8 + 4 + is] * (float)(q3 - 32);
                    y[base + 96 + lo + li]  = d * (float)sc[half*8 + 6 + is] * (float)(q4 - 32);
                }
            }
        }
    }
}

static void dequantize_f16(const uint8_t *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint16_t h = x[i * 2] | ((uint16_t)x[i * 2 + 1] << 8);
        uint32_t sign = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
        if (e == 0) y[i] = (sign ? -1.0f : 1.0f) * (float)m * 5.96046448e-8f;
        else if (e == 0x1F) y[i] = (m == 0) ? INFINITY : NAN;
        else { uint32_t f32 = (sign << 31) | ((e + 112) << 23) | (m << 13); memcpy(&y[i], &f32, sizeof(float)); }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Tensor loading helper
 * ═══════════════════════════════════════════════════════════════════════ */

static float *load_tensor_dequant(FILE *f, const GgufContext *ctx,
                                   const char *name, int64_t *ne_out) {
    const GgufTensorInfo *ti = gguf_get_tensor_by_name(ctx, name);
    if (!ti) {
        fprintf(stderr, "[text_gen] Tensor not found: %s\n", name);
        return NULL;
    }

    int64_t nelem = ti->ne[0] * ti->ne[1] * ti->ne[2] * ti->ne[3];
    if (nelem <= 0) nelem = 1;

    size_t raw_bytes = ti->nbytes;
    if (raw_bytes == 0) {
        if (ti->ggml_type == GGML_TYPE_F32) raw_bytes = (size_t)nelem * 4;
        else if (ti->ggml_type == GGML_TYPE_F16) raw_bytes = (size_t)nelem * 2;
        else if (ti->ggml_type == GGML_TYPE_Q4_K) raw_bytes = ((size_t)nelem + 255) / 256 * 144;
        else if (ti->ggml_type == GGML_TYPE_Q6_K) raw_bytes = ((size_t)nelem + 255) / 256 * 210;
        else raw_bytes = (size_t)nelem * 4;
    }

    uint8_t *raw = (uint8_t *)malloc(raw_bytes);
    if (!raw) return NULL;

    if (!gguf_load_tensor_data(f, ctx, ti, raw)) {
        free(raw);
        return NULL;
    }

    float *out = (float *)calloc((size_t)nelem, sizeof(float));
    if (!out) { free(raw); return NULL; }

    switch (ti->ggml_type) {
    case GGML_TYPE_F32:
        memcpy(out, raw, (size_t)nelem * sizeof(float));
        break;
    case GGML_TYPE_F16:
        dequantize_f16(raw, out, nelem);
        break;
    case GGML_TYPE_Q4_K:
        dequantize_q4_k(raw, out, nelem);
        break;
    case GGML_TYPE_Q6_K:
        dequantize_q6_k(raw, out, nelem);
        break;
    default:
        fprintf(stderr, "[text_gen] Unsupported dequant type %d for %s\n",
                ti->ggml_type, name);
        free(out); free(raw);
        return NULL;
    }

    free(raw);
    if (ne_out) *ne_out = nelem;
    return out;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Simple CPU forward ops
 * ═══════════════════════════════════════════════════════════════════════ */

static void cpu_rms_norm(const float *x, const float *weight, float *out,
                          int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float rms = 1.0f / sqrtf(sum_sq / (float)dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * rms * weight[i];
}

static void matmul(const float *a, const float *b, float *c,
                        int m, int k, int n) {
    /* a: [m, k], b: [n, k] stored as GGUF (output_dim x input_dim).
     * Computes c = a @ b^T:  c[i][j] = sum_p a[i][p] * b[j][p] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++)
                sum += a[i * k + p] * b[j * k + p];
            c[i * n + j] = sum;
        }
    }
}

#ifdef USE_CUDA
/* ── GPU-accelerated matmul via cuBLAS ─────────────────────────────────
 * Uses lazy-init handle + cached GPU buffers.  For row-major data:
 *   C[m,n] = A[m,k] @ B[n,k]^T
 * cuBLAS call (col-major equivalent):
 *   cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &a, B, k, A, k, &b, C_out, n)
 */
static cublasHandle_t g_cublas_handle = NULL;
static float *g_d_a = NULL, *g_d_b = NULL, *g_d_c = NULL;
static size_t g_d_a_sz = 0, g_d_b_sz = 0, g_d_c_sz = 0;

static void gpu_matmul(const float *a, const float *b, float *c,
                       int m, int k, int n) {
    if (!g_cublas_handle) {
        cublasCreate(&g_cublas_handle);
        fprintf(stderr, "[text_gen] cuBLAS initialized (V100)\n");
    }

    size_t sz_a = (size_t)m * (size_t)k * sizeof(float);
    size_t sz_b = (size_t)n * (size_t)k * sizeof(float);
    size_t sz_c = (size_t)m * (size_t)n * sizeof(float);

    if (sz_a > g_d_a_sz) {
        if (g_d_a) cudaFree(g_d_a);
        cudaMalloc((void **)&g_d_a, sz_a);
        g_d_a_sz = sz_a;
    }
    if (sz_b > g_d_b_sz) {
        if (g_d_b) cudaFree(g_d_b);
        cudaMalloc((void **)&g_d_b, sz_b);
        g_d_b_sz = sz_b;
    }
    if (sz_c > g_d_c_sz) {
        if (g_d_c) cudaFree(g_d_c);
        cudaMalloc((void **)&g_d_c, sz_c);
        g_d_c_sz = sz_c;
    }

    cudaMemcpy(g_d_a, a, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_b, b, sz_b, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                g_d_b, k,  /* B [n,k] row-major, ldb=k */
                g_d_a, k,  /* A [m,k] row-major, lda=k */
                &beta,
                g_d_c, n); /* C [m,n] row-major, ldc=n */

    cudaMemcpy(c, g_d_c, sz_c, cudaMemcpyDeviceToHost);
}

static void gpu_matmul_cleanup(void) {
    if (g_d_a) { cudaFree(g_d_a); g_d_a = NULL; g_d_a_sz = 0; }
    if (g_d_b) { cudaFree(g_d_b); g_d_b = NULL; g_d_b_sz = 0; }
    if (g_d_c) { cudaFree(g_d_c); g_d_c = NULL; g_d_c_sz = 0; }
    if (g_cublas_handle) { cublasDestroy(g_cublas_handle); g_cublas_handle = NULL; }
}

#define matmul gpu_matmul   /* GPU via cuBLAS */
#else
#endif

static void cpu_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        /* Numerically stable SiLU: x * sigmoid(x) = x / (1 + exp(-x)) */
        if (x[i] > 10.0f) {
            /* sigmoid ≈ 1, result ≈ x[i], keep as-is */
        } else if (x[i] < -10.0f) {
            x[i] = 0.0f; /* sigmoid ≈ 0, result ≈ 0 */
        } else {
            x[i] = x[i] / (1.0f + expf(-x[i]));
        }
    }
}

/* Simple dot-product attention (single-token query, no KV cache) */
static void cpu_attention(const float *q, const float *k, const float *v,
                           float *out, int n_head, int head_dim,
                           int seq_len, float scale) {
    int total_dim = n_head * head_dim;
    for (int h = 0; h < n_head; h++) {
        const float *qh = q + h * head_dim;
        float *oh = out + h * head_dim;
        memset(oh, 0, (size_t)head_dim * sizeof(float));

        float *scores = (float *)calloc((size_t)seq_len, sizeof(float));
        float max_val = -1e9f;
        for (int t = 0; t < seq_len; t++) {
            float dot = 0.0f;
            const float *kt = k + t * total_dim + h * head_dim;
            for (int d = 0; d < head_dim; d++)
                dot += qh[d] * kt[d];
            scores[t] = dot * scale;
            if (scores[t] > max_val) max_val = scores[t];
        }

        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_val);
            sum_exp += scores[t];
        }

        for (int t = 0; t < seq_len; t++) {
            float w = scores[t] / sum_exp;
            const float *vt = v + t * total_dim + h * head_dim;
            for (int d = 0; d < head_dim; d++)
                oh[d] += w * vt[d];
        }
        free(scores);
    }
}

/* RoPE type: NORM = adjacent pairs (Llama), NEOX = offset by n_rot/2 (Qwen, Falcon, etc.) */
#define ROPE_TYPE_NORM 0
#define ROPE_TYPE_NEOX 1

/* Apply RoPE to a single token's Q or K */
static void cpu_rope(float *x, int n_head, int head_dim, int n_rot, int position,
                      float theta, int rope_type) {
    if (rope_type == ROPE_TYPE_NEOX) {
        /* NeoX-style: pair (d, d + n_rot/2) for d in [0, n_rot/2) */
        for (int h = 0; h < n_head; h++) {
            for (int d = 0; d < n_rot / 2; d++) {
                float freq = 1.0f / powf(theta, (float)d / (float)(n_rot / 2));
                float angle = (float)position * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                int idx0 = h * head_dim + d;
                int idx1 = h * head_dim + d + n_rot / 2;
                float x0 = x[idx0];
                float x1 = x[idx1];
                x[idx0] = x0 * cos_a - x1 * sin_a;
                x[idx1] = x0 * sin_a + x1 * cos_a;
            }
        }
    } else {
        /* Normal: adjacent pairs (d, d+1) */
        for (int h = 0; h < n_head; h++) {
            for (int d = 0; d < head_dim; d += 2) {
                float freq = 1.0f / powf(theta, (float)d / (float)head_dim);
                float angle = (float)position * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                int idx = h * head_dim + d;
                float x0 = x[idx];
                float x1 = x[idx + 1];
                x[idx]     = x0 * cos_a - x1 * sin_a;
                x[idx + 1] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

/* Greedy sampling */
/* Temperature sampling with softmax + multinomial.
 * temperature=0.0 or <=0 → greedy argmax; temperature>0 → multinomial */
static int32_t sample_token(const float *logits, int32_t n, float temperature) {
    if (temperature <= 0.0f) {
        int32_t best = 0;
        float best_val = logits[0];
        for (int32_t i = 1; i < n; i++)
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        return best;
    }

    /* Softmax with temperature */
    float max_val = logits[0];
    for (int32_t i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum = 0.0f;
    float *probs = (float *)malloc((size_t)n * sizeof(float));
    if (!probs) {
        /* OOM fallback — pure greedy */
        int32_t best = 0; float best_val = logits[0];
        for (int32_t i = 1; i < n; i++)
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        return best;
    }

    for (int32_t i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }

    /* Multinomial sample */
    float r = (float)rand() / (float)RAND_MAX * sum;
    float cum = 0.0f;
    int32_t chosen = n - 1;
    for (int32_t i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) { chosen = i; break; }
    }

    free(probs);
    return chosen;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Per-layer weight bundle
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    float *attn_norm;    /* [dim] */
    float *ffn_norm;     /* [dim] */
    float *attn_q;       /* [dim, dim] */
    float *attn_k;       /* [dim, kv_dim] */
    float *attn_v;       /* [dim, kv_dim] */
    float *attn_o;       /* [dim, dim] */
    float *ffn_gate;     /* [dim, ffn_dim] */
    float *ffn_up;       /* [dim, ffn_dim] */
    float *ffn_down;     /* [ffn_dim, dim] */
    float *attn_q_norm;  /* [dim] — Qwen3 Q/K norms */
    float *attn_k_norm;  /* [kv_dim] */
} LayerWeights;

static LayerWeights load_layer_weights(FILE *f, const GgufContext *ctx,
                                        int layer) {
    LayerWeights lw;
    memset(&lw, 0, sizeof(lw));
    char name[64];

    int missing = 0;
    (void)missing;

    snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", layer);
    lw.attn_norm = load_tensor_dequant(f, ctx, name, NULL);
    if (!lw.attn_norm) missing++;

    snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", layer);
    lw.ffn_norm  = load_tensor_dequant(f, ctx, name, NULL);
    if (!lw.ffn_norm) missing++;

    snprintf(name, sizeof(name), "blk.%d.attn_q.weight", layer);
    lw.attn_q    = load_tensor_dequant(f, ctx, name, NULL);
    if (!lw.attn_q) missing++;

    snprintf(name, sizeof(name), "blk.%d.attn_k.weight", layer);
    lw.attn_k    = load_tensor_dequant(f, ctx, name, NULL);
    if (!lw.attn_k) missing++;

    snprintf(name, sizeof(name), "blk.%d.attn_v.weight", layer);
    lw.attn_v    = load_tensor_dequant(f, ctx, name, NULL);
    if (!lw.attn_v) missing++;

    snprintf(name, sizeof(name), "blk.%d.attn_output.weight", layer);
    lw.attn_o    = load_tensor_dequant(f, ctx, name, NULL);
    if (!lw.attn_o) missing++;

    snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", layer);
    lw.ffn_gate  = load_tensor_dequant(f, ctx, name, NULL);
    if (!lw.ffn_gate) missing++;

    snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", layer);
    lw.ffn_up    = load_tensor_dequant(f, ctx, name, NULL);
    if (!lw.ffn_up) missing++;

    snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", layer);
    lw.ffn_down  = load_tensor_dequant(f, ctx, name, NULL);
    if (!lw.ffn_down) missing++;

    /* Q/K norms (Qwen3, optional — not present in Llama) */
    int64_t qn_nelem = 0, kn_nelem = 0;
    snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", layer);
    lw.attn_q_norm = load_tensor_dequant(f, ctx, name, &qn_nelem);

    snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", layer);
    lw.attn_k_norm = load_tensor_dequant(f, ctx, name, &kn_nelem);

    if (missing)
        fprintf(stderr, "[text_gen] Layer %d: %d tensors missing\n", layer, missing);

    return lw;
}

static void free_layer_weights(LayerWeights *lw) {
    free(lw->attn_norm); free(lw->ffn_norm);
    free(lw->attn_q); free(lw->attn_k); free(lw->attn_v); free(lw->attn_o);
    free(lw->ffn_gate); free(lw->ffn_up); free(lw->ffn_down);
    free(lw->attn_q_norm); free(lw->attn_k_norm);
    memset(lw, 0, sizeof(*lw));
}

/* Preload all layer weights into memory.  Called once at startup;
 * prefill and decode reuse the cached weights instead of hitting disk. */
static LayerWeights *preload_all_weights(FILE *f, const GgufContext *ctx,
                                         int n_layers) {
    LayerWeights *w = (LayerWeights *)calloc((size_t)n_layers, sizeof(LayerWeights));
    if (!w) return NULL;
    for (int i = 0; i < n_layers; i++) {
        w[i] = load_layer_weights(f, ctx, i);
    }
    fprintf(stderr, "[text_gen] Preloaded all %d layers weights\n", n_layers);
    return w;
}

static void free_all_weights(LayerWeights *w, int n_layers) {
    if (!w) return;
    for (int i = 0; i < n_layers; i++) free_layer_weights(&w[i]);
    free(w);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Model context
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    Gpt2Config cfg;
    int32_t    n_kv_head;
    int32_t    kv_dim;
    int32_t    head_dim;
    float      rope_theta;
    int        rope_type;   /* ROPE_TYPE_NORM or ROPE_TYPE_NEOX */
    int        rotary_dim;  /* dims to apply RoPE (Qwen3: head_dim/2, others: head_dim) */
} ModelInfo;

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <gguf_path> [\"prompt\"] [-DMAXL=N]\n", argv[0]);
        return 1;
    }

    const char *gguf_path = argv[1];
    const char *prompt    = NULL;
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "-DMAXL=", 7) == 0)
            g_max_layers = atoi(argv[i] + 7);
        else if (!prompt)
            prompt = argv[i];
    }
    if (!prompt)
        prompt = "Hello World，山野村夫向vLLM、LlamaCpp、Ollama前輩致敬~~~";

    /* ── Parse GGUF header ─────────────────────────────────────────── */
    fprintf(stderr, "[text_gen] Loading model: %s\n", gguf_path);
    GgufContext *ctx = gguf_init_from_file(gguf_path);
    if (!ctx) { fprintf(stderr, "[text_gen] Failed to parse GGUF\n"); return 1; }

    const char *arch = gguf_get_val_str(ctx, GGUF_KEY_ARCHITECTURE);
    const char *name = gguf_get_val_str(ctx, GGUF_KEY_MODEL_NAME);
    fprintf(stderr, "[text_gen] Architecture: %s, Model: %s\n",
            arch ? arch : "?", name ? name : "?");

    ModelInfo m;
    memset(&m, 0, sizeof(m));
    gguf_load_model_config(ctx, &m.cfg);

    {
        char kv_key[256];
        snprintf(kv_key, sizeof(kv_key), "%s.attention.head_count_kv", arch);
        m.n_kv_head = (int32_t)gguf_get_val_int(ctx, kv_key, -1);
        if (m.n_kv_head <= 0)
            m.n_kv_head = (int32_t)gguf_get_val_int(ctx,
                "llama.attention.head_count_kv", m.cfg.n_head);
    }
    m.head_dim   = m.cfg.n_embd / m.cfg.n_head;
    m.kv_dim     = m.n_kv_head * m.head_dim;
    /* Try architecture-specific rope.freq_base first (e.g. qwen3.rope.freq_base),
     * fall back to llama.rope.freq_base */
    char rope_key[256];
    snprintf(rope_key, sizeof(rope_key), "%s.rope.freq_base", arch);
    m.rope_theta = gguf_get_val_f32(ctx, rope_key, 0.0f);
    if (m.rope_theta <= 0.0f)
        m.rope_theta = gguf_get_val_f32(ctx, "llama.rope.freq_base", 10000.0f);

    /* Detect RoPE type from architecture.
     * NeoX-style (pairs offset by n_rot/2): Qwen, Falcon, GPT-NeoX, Phi, Gemma, etc.
     * Normal (adjacent pairs): Llama, Mistral, DeepSeek, etc. */
    m.rope_type = ROPE_TYPE_NORM;  /* default */
    if (arch) {
        if (strstr(arch, "qwen") || strstr(arch, "Qwen") ||
            strstr(arch, "falcon") || strstr(arch, "Falcon") ||
            strstr(arch, "gptneox") || strstr(arch, "stablelm") ||
            strstr(arch, "phi") || strstr(arch, "gemma") ||
            strstr(arch, "deepseek2") || strstr(arch, "deepseek3") ||
            strstr(arch, "nemotron") || strstr(arch, "orion") ||
            strstr(arch, "olmo2") || strstr(arch, "olmoe") ||
            strstr(arch, "exaone") || strstr(arch, "bitnet") ||
            strstr(arch, "starcoder2") || strstr(arch, "openelm") ||
            strstr(arch, "minicpm3") || strstr(arch, "arcee") ||
            strstr(arch, "jais") || strstr(arch, "seed") ||
            strstr(arch, "plamo") || strstr(arch, "grok") ||
            strstr(arch, "dbrx") || strstr(arch, "codeshell"))
            m.rope_type = ROPE_TYPE_NEOX;
    }
    fprintf(stderr, "[text_gen] RoPE type: %s (theta=%.0f)\n",
            m.rope_type == ROPE_TYPE_NEOX ? "NeoX" : "Normal", m.rope_theta);

    /* llama.cpp asserts n_embd_head == n_rot for Qwen3; full 128-dim RoPE */
    m.rotary_dim = m.head_dim;

    /* Derive actual kv_dim and ffn from tensor shapes, not metadata keys.
     * Qwen3 and some other models have metadata that doesn't match the
     * actual weight dimensions (e.g. GQA kv_heads advertised as full n_head,
     * or ffn different from llama.feed_forward_length). */
    {
        const GgufTensorInfo *ti_k = gguf_get_tensor_by_name(ctx, "blk.0.attn_k.weight");
        if (ti_k && ti_k->ne[1] > 0 && ti_k->ne[0] == m.cfg.n_embd) {
            int32_t actual_kv_dim = (int32_t)ti_k->ne[1];
            if (actual_kv_dim != m.kv_dim) {
                fprintf(stderr, "[text_gen] NOTE: correcting kv_dim from %d to %d "
                        "(from tensor shape)\n", m.kv_dim, actual_kv_dim);
                m.kv_dim = actual_kv_dim;
                m.n_kv_head = m.kv_dim / m.head_dim;
            }
        }
        const GgufTensorInfo *ti_ffn = gguf_get_tensor_by_name(ctx, "blk.0.ffn_gate.weight");
        if (ti_ffn && ti_ffn->ne[1] > 0 && ti_ffn->ne[0] == m.cfg.n_embd) {
            int32_t actual_ffn = (int32_t)ti_ffn->ne[1];
            if (actual_ffn != m.cfg.n_inner) {
                fprintf(stderr, "[text_gen] NOTE: correcting ffn from %d to %d "
                        "(from tensor shape)\n", m.cfg.n_inner, actual_ffn);
                m.cfg.n_inner = actual_ffn;
            }
        }
        const GgufTensorInfo *ti_out = gguf_get_tensor_by_name(ctx, "output.weight");
        if (ti_out && ti_out->ne[1] > 0) {
            int32_t actual_vocab = (int32_t)ti_out->ne[1];
            if (actual_vocab != m.cfg.vocab_size) {
                fprintf(stderr, "[text_gen] NOTE: correcting vocab from %d to %d "
                        "(from tensor shape)\n", m.cfg.vocab_size, actual_vocab);
                m.cfg.vocab_size = actual_vocab;
            }
        }
    }

    fprintf(stderr, "[text_gen] Config: vocab=%d ctx=%d embd=%d layers=%d "
            "heads=%d kv_heads=%d head_dim=%d ffn=%d rope_theta=%.0f\n",
            m.cfg.vocab_size, m.cfg.n_positions, m.cfg.n_embd,
            m.cfg.n_layer, m.cfg.n_head, m.n_kv_head, m.head_dim,
            m.cfg.n_inner, m.rope_theta);

    /* ── Load tokenizer ────────────────────────────────────────────── */
    XllmTokenizer *tok = xllm_tokenizer_load_from_gguf(gguf_path);
    if (!tok) { gguf_free(ctx); return 1; }

    /* Tokenize prompt */
    int32_t tokens[XLLM_MAX_ENCODED];
    int32_t n_tokens = xllm_tokenizer_encode(tok, prompt, tokens, XLLM_MAX_ENCODED);
    fprintf(stderr, "[text_gen] Prompt: \"%s\" -> %d tokens\n", prompt, n_tokens);
    for (int i = 0; i < (n_tokens < 10 ? n_tokens : 10); i++)
        fprintf(stderr, "  token[%d] = %d\n", i, tokens[i]);

    /* ── Open model file for tensor loading ────────────────────────── */
    FILE *f = fopen(gguf_path, "rb");
    if (!f) { xllm_tokenizer_free(tok); gguf_free(ctx); return 1; }

    /* ── Load embedding table and final norm (needed throughout) ───── */
    int64_t embd_elems = 0;
    float *embd = load_tensor_dequant(f, ctx, "token_embd.weight", &embd_elems);
    if (!embd) {
        fprintf(stderr, "[text_gen] Failed to load token_embd.weight\n");
        fclose(f); xllm_tokenizer_free(tok); gguf_free(ctx); return 1;
    }
    fprintf(stderr, "[text_gen] Loaded token_embd.weight: %" PRId64 " elems\n", embd_elems);

    float *ln_f = load_tensor_dequant(f, ctx, "output_norm.weight", NULL);
    float *lm_head = load_tensor_dequant(f, ctx, "output.weight", NULL);
    fprintf(stderr, "[text_gen] output_norm: %s, lm_head: %s\n",
            ln_f ? "loaded" : "MISSING", lm_head ? "loaded" : "MISSING");

    if (!ln_f || !lm_head) {
        fprintf(stderr, "[text_gen] Missing output weights, aborting\n");
        free(embd); if (ln_f) free(ln_f); if (lm_head) free(lm_head);
        fclose(f); xllm_tokenizer_free(tok); gguf_free(ctx); return 1;
    }

    /* ── EOS token ───────────────────────────────────────────────── */
    int32_t eos_token = tok->eos_token_id;
    fprintf(stderr, "[text_gen] EOS token: %d, temperature=0.8\n", eos_token);

    /* ── Forward pass ──────────────────────────────────────────────── */
    int dim   = m.cfg.n_embd;
    int ffn   = m.cfg.n_inner;
    int n_h   = m.cfg.n_head;
    int kv_h  = m.n_kv_head;
    int hd    = m.head_dim;
    int kv_dim = m.kv_dim;
    float eps = m.cfg.layer_norm_eps;

    int seq_len = n_tokens;
    int heads_per_kv = n_h / kv_h;
    float attn_scale = 1.0f / sqrtf((float)hd);
    float temperature = 0.8f;
    srand((unsigned)time(NULL));

    /* ── KV cache: [n_layer][max_seq][kv_dim] ─────────────────── */
    size_t cache_per_layer = (size_t)MAX_SEQ_LEN * (size_t)kv_dim;
    float *k_cache = (float *)calloc((size_t)m.cfg.n_layer * cache_per_layer, sizeof(float));
    float *v_cache = (float *)calloc((size_t)m.cfg.n_layer * cache_per_layer, sizeof(float));
    if (!k_cache || !v_cache) {
        fprintf(stderr, "[text_gen] Failed to allocate KV cache\n");
        free(embd); free(ln_f); free(lm_head);
        fclose(f); xllm_tokenizer_free(tok); gguf_free(ctx); return 1;
    }

    /* ── Temp buffers (reused across layers) ──────────────────── */
    float *x_seq = (float *)calloc((size_t)seq_len * (size_t)dim, sizeof(float));
    float *q_seq = (float *)calloc((size_t)seq_len * (size_t)dim, sizeof(float));
    float *k_seq = (float *)calloc((size_t)seq_len * (size_t)kv_dim, sizeof(float));
    float *v_seq = (float *)calloc((size_t)seq_len * (size_t)kv_dim, sizeof(float));
    float *residual = (float *)calloc((size_t)dim, sizeof(float));
    float *xn       = (float *)calloc((size_t)dim, sizeof(float));
    float *ao       = (float *)calloc((size_t)dim, sizeof(float));
    float *gate     = (float *)calloc((size_t)ffn, sizeof(float));
    float *up       = (float *)calloc((size_t)ffn, sizeof(float));
    float *ffn_out  = (float *)calloc((size_t)dim, sizeof(float));
    float *logits   = (float *)calloc((size_t)m.cfg.vocab_size, sizeof(float));

    /* Ones fallback for missing norm weights */
    float *ones_norm = (float *)calloc((size_t)dim, sizeof(float));
    for (int d = 0; d < dim; d++) ones_norm[d] = 1.0f;

    /* ── Embed prompt tokens ──────────────────────────────────── */
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < dim; d++)
            x_seq[t * dim + d] = embd[(int64_t)tokens[t] * dim + d];
    }
    /* Debug: inspect embeddings, norm, lm_head numerically */
    {
        float *lx = x_seq + (seq_len - 1) * dim;
        float esum = 0, emin = 1e9f, emax = -1e9f;
        for (int d = 0; d < dim; d++) { float v = lx[d]; esum += v*v; if(v<emin)emin=v; if(v>emax)emax=v; }
        fprintf(stderr, "[dbg] embed last tok: |x|=%.1f rng[%.3f,%.3f] first10:[", sqrtf(esum), emin, emax);
        for (int d = 0; d < 10; d++) fprintf(stderr, "%.3f ", lx[d]);
        fprintf(stderr, "]\n");
        /* Check lm_head first 10 rows (token 0..9) dotted with embedding */
        float top10_logits[10] = {0};
        for (int j = 0; j < 10; j++) {
            for (int d = 0; d < dim; d++)
                top10_logits[j] += lx[d] * lm_head[j * dim + d];
        }
        fprintf(stderr, "[dbg] embed@lm_head (no norm) first10 tok: [");
        for (int j = 0; j < 10; j++) fprintf(stderr, "%.1f ", top10_logits[j]);
        fprintf(stderr, "]\n");
        /* Check known Chinese tokens without norm */
        int ct[] = {151643, 151644, 151645, 77091, 233, 100168, 99478};
        fprintf(stderr, "[dbg] embed@lm_head Chinese: ");
        for (int ci = 0; ci < 7; ci++) {
            float s = 0;
            for (int d = 0; d < dim; d++) s += lx[d] * lm_head[ct[ci] * dim + d];
            fprintf(stderr, "#%d=>%.1f ", ct[ci], s);
        }
        fprintf(stderr, "\n");
        /* Check output_norm weight */
        float nmin = 1e9f, nmax = -1e9f, nsum = 0;
        for (int d = 0; d < dim; d++) { float v = ln_f[d]; if(v<nmin)nmin=v; if(v>nmax)nmax=v; nsum+=v; }
        fprintf(stderr, "[dbg] output_norm wt: rng[%.4f,%.4f] mean=%.4f first10:[",
                nmin, nmax, nsum/dim);
        for (int d = 0; d < 10; d++) fprintf(stderr, "%.4f ", ln_f[d]);
        fprintf(stderr, "]\n");
        /* After norm */
        float *dbg_norm = (float *)calloc(dim, sizeof(float));
        cpu_rms_norm(lx, ln_f, dbg_norm, dim, eps);
        float nsq = 0;
        for (int d = 0; d < dim; d++) nsq += dbg_norm[d] * dbg_norm[d];
        fprintf(stderr, "[dbg] after norm: |x|=%.1f first10:[", sqrtf(nsq));
        for (int d = 0; d < 10; d++) fprintf(stderr, "%.3f ", dbg_norm[d]);
        fprintf(stderr, "]\n");
        /* logits for first 10 tokens after norm */
        for (int j = 0; j < 10; j++) {
            float s = 0;
            for (int d = 0; d < dim; d++) s += dbg_norm[d] * lm_head[j * dim + d];
            top10_logits[j] = s;
        }
        fprintf(stderr, "[dbg] norm@lm_head first10 tok: [");
        for (int j = 0; j < 10; j++) fprintf(stderr, "%.1f ", top10_logits[j]);
        fprintf(stderr, "]\n");
        free(dbg_norm);
    }

    /* ── Preload all layer weights once ───────────────────────── */
    int n_layers_to_load = g_max_layers ? g_max_layers : m.cfg.n_layer;
    LayerWeights *g_weights = preload_all_weights(f, ctx, n_layers_to_load);
    if (!g_weights) {
        fprintf(stderr, "[text_gen] Failed to preload weights\n");
        goto done_generating;
    }

    /* ── Batched temp buffers for prefill ──────────────────── */
    float *xn_seq  = (float *)calloc((size_t)seq_len * (size_t)dim, sizeof(float));
    float *ao_seq  = (float *)calloc((size_t)seq_len * (size_t)dim, sizeof(float));
    float *res_seq = (float *)calloc((size_t)seq_len * (size_t)dim, sizeof(float));
    float *gate_seq= (float *)calloc((size_t)seq_len * (size_t)ffn, sizeof(float));
    float *up_seq  = (float *)calloc((size_t)seq_len * (size_t)ffn, sizeof(float));
    float *ff_seq  = (float *)calloc((size_t)seq_len * (size_t)dim, sizeof(float));

    fprintf(stderr, "[text_gen] === PREFILL: seq_len=%d, %d layers ===\n",
            seq_len, m.cfg.n_layer);

#if SKIP_LAYERS
    fprintf(stderr, "[text_gen] *** SKIPPING ALL TRANSFORMER LAYERS (debug) ***\n");
#else
    /* ══════════════════════════════════════════════════════════════
     * PREFILL: process all prompt tokens, cache K/V per layer
     * ══════════════════════════════════════════════════════════════ */
    for (int layer = 0; layer < m.cfg.n_layer; layer++) {
if (g_max_layers && layer >= g_max_layers) break;
        fprintf(stderr, "[text_gen] Prefill layer %d...\n", layer);
        LayerWeights lw = g_weights[layer];

        if (!lw.attn_norm) lw.attn_norm = ones_norm;
        if (!lw.ffn_norm)  lw.ffn_norm  = ones_norm;

        /* Debug: RMS norm weight stats for first and last layer */
        if (layer == 0 || layer == m.cfg.n_layer - 1) {
            float an_min=1e9f, an_max=-1e9f, an_sum=0;
            float fn_min=1e9f, fn_max=-1e9f, fn_sum=0;
            for (int d=0;d<dim;d++){
                float av=lw.attn_norm[d], fv=lw.ffn_norm[d];
                if(av<an_min)an_min=av; if(av>an_max)an_max=av; an_sum+=av;
                if(fv<fn_min)fn_min=fv; if(fv>fn_max)fn_max=fv; fn_sum+=fv;
            }
            fprintf(stderr,"[layer %d] attn_norm: rng[%.4f,%.4f] mean=%.4f  "
                    "ffn_norm: rng[%.4f,%.4f] mean=%.4f\n",
                    layer, an_min, an_max, an_sum/(float)dim,
                    fn_min, fn_max, fn_sum/(float)dim);
        }

        if (!lw.attn_q || !lw.attn_k || !lw.attn_v || !lw.attn_o ||
            !lw.ffn_gate || !lw.ffn_up || !lw.ffn_down) {
            fprintf(stderr, "[text_gen] Incomplete weights layer %d, stopping\n", layer);
            break;
        }

        /* Debug: print first values of layer 0 weights */
        if (layer == 0) {
            fprintf(stderr, "[dbg L0] attn_q first10: [");
            for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", lw.attn_q[i]);
            fprintf(stderr, "]\n");
            fprintf(stderr, "[dbg L0] attn_k first10: [");
            for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", lw.attn_k[i]);
            fprintf(stderr, "]\n");
            fprintf(stderr, "[dbg L0] attn_v first10: [");
            for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", lw.attn_v[i]);
            fprintf(stderr, "]\n");
            fprintf(stderr, "[dbg L0] attn_o first10: [");
            for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", lw.attn_o[i]);
            fprintf(stderr, "]\n");
            fprintf(stderr, "[dbg L0] ffn_gate first10: [");
            for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", lw.ffn_gate[i]);
            fprintf(stderr, "]\n");
            if (lw.attn_q_norm) {
                fprintf(stderr, "[dbg L0] attn_q_norm first10: [");
                for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", lw.attn_q_norm[i]);
                fprintf(stderr, "]\n");
            }
            if (lw.attn_k_norm) {
                fprintf(stderr, "[dbg L0] attn_k_norm first10: [");
                for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", lw.attn_k_norm[i]);
                fprintf(stderr, "]\n");
            }
        }

        /* RMS-norm all tokens → xn_seq, then batch Q/K/V projections */
        for (int t = 0; t < seq_len; t++)
            cpu_rms_norm(x_seq + t * dim, lw.attn_norm,
                         xn_seq + t * dim, dim, eps);
        matmul(xn_seq, lw.attn_q, q_seq, seq_len, dim, dim);
        matmul(xn_seq, lw.attn_k, k_seq, seq_len, dim, kv_dim);
        matmul(xn_seq, lw.attn_v, v_seq, seq_len, dim, kv_dim);

        /* Q/K norms (Qwen3) — per-head, weight shape = [head_dim] */
#if !SKIP_QK_NORM
        if (lw.attn_q_norm) {
            for (int t = 0; t < seq_len; t++)
                for (int h = 0; h < n_h; h++)
                    cpu_rms_norm(q_seq + t * dim + h * hd, lw.attn_q_norm,
                                 q_seq + t * dim + h * hd, hd, eps);
        }
        if (lw.attn_k_norm) {
            for (int t = 0; t < seq_len; t++)
                for (int h = 0; h < kv_h; h++)
                    cpu_rms_norm(k_seq + t * kv_dim + h * hd, lw.attn_k_norm,
                                 k_seq + t * kv_dim + h * hd, hd, eps);
        }
#endif

        /* Debug: inspect Q/K/V after projection + norm (before RoPE) at layer 0, first token */
        if (layer == 0) {
            float q_n0 = 0, k_n0 = 0, v_n0 = 0;
            int k_nan = 0, k_inf = 0;
            float k_min = 1e9f, k_max = -1e9f;
            for (int d = 0; d < dim; d++) q_n0 += q_seq[d] * q_seq[d];
            for (int d = 0; d < kv_dim; d++) {
                float v = k_seq[d];
                k_n0 += v * v;
                if (isnan(v)) k_nan++;
                if (isinf(v)) k_inf++;
                if (v < k_min) k_min = v;
                if (v > k_max) k_max = v;
            }
            for (int d = 0; d < kv_dim; d++) v_n0 += v_seq[d] * v_seq[d];
            fprintf(stderr, "[dbg L0] after proj+norm, token0: |Q|=%.1f |K|=%.1f |V|=%.1f\n",
                    sqrtf(q_n0), sqrtf(k_n0), sqrtf(v_n0));
            fprintf(stderr, "[dbg L0] K NaN=%d Inf=%d min=%.3f max=%.3f\n", k_nan, k_inf, k_min, k_max);
            /* Per-head K RMS after norm */
            fprintf(stderr, "[dbg L0] K per-head RMS after norm: [");
            for (int h = 0; h < kv_h; h++) {
                float hsq = 0;
                for (int d = 0; d < hd; d++) {
                    float v = k_seq[h * hd + d];
                    hsq += v * v;
                }
                fprintf(stderr, "%.2f%s", sqrtf(hsq/(float)hd), h < kv_h-1 ? ", " : "");
            }
            fprintf(stderr, "]\n");
            /* Check K head 0 first 20 values */
            fprintf(stderr, "[dbg L0] K head0 first20: [");
            for (int i = 0; i < 20; i++) fprintf(stderr, "%.3f ", k_seq[i]);
            fprintf(stderr, "]\n");
            /* Check all attn_k_norm weight range */
            if (lw.attn_k_norm) {
                float kn_min = 1e9f, kn_max = -1e9f;
                for (int i = 0; i < hd; i++) {
                    if (lw.attn_k_norm[i] < kn_min) kn_min = lw.attn_k_norm[i];
                    if (lw.attn_k_norm[i] > kn_max) kn_max = lw.attn_k_norm[i];
                }
                fprintf(stderr, "[dbg L0] attn_k_norm wt: rng[%.4f, %.4f]\n", kn_min, kn_max);
            }
            /* Also show Q/K/V BEFORE norm by temporarily re-projecting */
            {
                float q_before[4096], k_before[1024], v_before[1024];
                /* DEBUG: print raw embedding for position 0 */
                float esq0 = 0;
                for(int ddd=0;ddd<dim;ddd++) esq0 += x_seq[ddd] * x_seq[ddd];
                fprintf(stderr, "[dbg L0] x_seq[0] (token0 embd): |x|=%.1f first10:[",
                        sqrtf(esq0));
                for (int ddd = 0; ddd < 10; ddd++) fprintf(stderr, "%.6f ", x_seq[ddd]);
                fprintf(stderr, "]\n");
                matmul(xn_seq, lw.attn_q, q_before, 1, dim, dim);
                matmul(xn_seq, lw.attn_k, k_before, 1, dim, kv_dim);
                matmul(xn_seq, lw.attn_v, v_before, 1, dim, kv_dim);
                float qbsq = 0, kbsq = 0, vbsq = 0;
                for (int d = 0; d < dim; d++) qbsq += q_before[d] * q_before[d];
                for (int d = 0; d < kv_dim; d++) kbsq += k_before[d] * k_before[d];
                for (int d = 0; d < kv_dim; d++) vbsq += v_before[d] * v_before[d];
                fprintf(stderr, "[dbg L0] BEFORE norm: |Q|=%.1f |K|=%.1f |V|=%.1f\n",
                        sqrtf(qbsq), sqrtf(kbsq), sqrtf(vbsq));
                fprintf(stderr, "[dbg L0] xn_seq (input) first10: [");
                for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", xn_seq[i]);
                fprintf(stderr, "]\n");
                /* Dump ALL 128 K head0 BEFORE norm for Python comparison */
                fprintf(stderr, "[dbg L0] K_head0_BEFORE=[");
                for (int i = 0; i < 128; i++) fprintf(stderr, "%.6f%s", k_before[i], i<127?",":"");
                fprintf(stderr, "]\n");
                /* Dump ALL 128 K head0 AFTER norm (from k_seq) */
                fprintf(stderr, "[dbg L0] K_head0_AFTER=[");
                for (int i = 0; i < 128; i++) fprintf(stderr, "%.6f%s", k_seq[i], i<127?",":"");
                fprintf(stderr, "]\n");
                /* Dump ALL 128 Q head0 AFTER norm */
                fprintf(stderr, "[dbg L0] Q_head0_AFTER=[");
                for (int i = 0; i < 128; i++) fprintf(stderr, "%.6f%s", q_seq[i], i<127?",":"");
                fprintf(stderr, "]\n");
            }
            /* Also show Q before norm */
            {
                float q_before[4096];
                matmul(xn_seq, lw.attn_q, q_before, 1, dim, dim);
                float qbsq = 0;
                for (int d = 0; d < dim; d++) qbsq += q_before[d] * q_before[d];
                fprintf(stderr, "[dbg L0] Q BEFORE norm: |Q|=%.1f first10: [", sqrtf(qbsq));
                for (int i = 0; i < 10; i++) fprintf(stderr, "%.3f ", q_before[i]);
                fprintf(stderr, "]\n");
            }
        }

        /* RoPE + save to cache */
        float *kc = k_cache + layer * cache_per_layer;
        float *vc = v_cache + layer * cache_per_layer;
        for (int t = 0; t < seq_len; t++) {
            cpu_rope(q_seq + t * dim, n_h, hd, m.rotary_dim, t, m.rope_theta, m.rope_type);
            cpu_rope(k_seq + t * kv_dim, kv_h, hd, m.rotary_dim, t, m.rope_theta, m.rope_type);
        }
        memcpy(kc, k_seq, (size_t)seq_len * (size_t)kv_dim * sizeof(float));
        memcpy(vc, v_seq, (size_t)seq_len * (size_t)kv_dim * sizeof(float));

        /* Save residuals before attention overwrites x_seq */
        memcpy(res_seq, x_seq, (size_t)seq_len * (size_t)dim * sizeof(float));

        /* Causal attention per position (CPU) */
        float xn_sum = 0, attn_sum = 0, ao_sum = 0, gate_sum = 0, ff_sum = 0;
        for (int t = 0; t < seq_len; t++) {
            float *xt = x_seq + t * dim;
            int ctx_len = t + 1;
            for (int h = 0; h < n_h; h++) {
                int kh = h / heads_per_kv;
                const float *qh = q_seq + t * dim + h * hd;
                float *oh = xt + h * hd;

                float *scores = (float *)calloc((size_t)ctx_len, sizeof(float));
                float max_val = -1e9f;
                for (int s = 0; s < ctx_len; s++) {
                    float dot = 0.0f;
                    const float *ks = k_seq + s * kv_dim + kh * hd;
                    for (int d = 0; d < hd; d++) dot += qh[d] * ks[d];
                    scores[s] = dot * attn_scale;
                    if (scores[s] > max_val) max_val = scores[s];
                }
                float sum_exp = 0.0f;
                for (int s = 0; s < ctx_len; s++) {
                    scores[s] = expf(scores[s] - max_val);
                    sum_exp += scores[s];
                }
                memset(oh, 0, (size_t)hd * sizeof(float));
                for (int s = 0; s < ctx_len; s++) {
                    float w = scores[s] / sum_exp;
                    const float *vs = v_seq + s * kv_dim + kh * hd;
                    for (int d = 0; d < hd; d++) oh[d] += w * vs[d];
                }
                free(scores);
            }
            /* Debug: attention weights for last token at layer 0 */
            if (layer == 0 && t == seq_len - 1) {
                fprintf(stderr, "[attn dbg] layer 0, last token (pos=%d):\n", t);
                for (int hh = 0; hh < (n_h < 4 ? n_h : 4); hh++) {
                    fprintf(stderr, "  head %d:", hh);
                    int khh = hh / heads_per_kv;
                    const float *qhh = q_seq + t * dim + hh * hd;
                    const float *khh0 = k_seq + 0 * kv_dim + khh * hd;
                    const float *khht = k_seq + t * kv_dim + khh * hd;
                    float d0 = 0, dt = 0;
                    for (int dd = 0; dd < hd; dd++) {
                        d0 += qhh[dd] * khh0[dd];
                        dt += qhh[dd] * khht[dd];
                    }
                    fprintf(stderr, " Q·K[pos0]=%.1f Q·K[pos%d]=%.1f", d0, t, dt);
                    /* full attention weights */
                    float *test_scores = (float *)calloc((size_t)ctx_len, sizeof(float));
                    float test_max = -1e9f;
                    for (int s = 0; s < ctx_len; s++) {
                        float dot = 0.0f;
                        const float *kss = k_seq + s * kv_dim + khh * hd;
                        for (int dd = 0; dd < hd; dd++) dot += qhh[dd] * kss[dd];
                        test_scores[s] = dot * attn_scale;
                        if (test_scores[s] > test_max) test_max = test_scores[s];
                    }
                    float test_sum = 0;
                    for (int s = 0; s < ctx_len; s++) {
                        test_scores[s] = expf(test_scores[s] - test_max);
                        test_sum += test_scores[s];
                    }
                    fprintf(stderr, " attn:[");
                    for (int s = 0; s < ctx_len; s++)
                        fprintf(stderr, "%.2f%s", test_scores[s]/test_sum,
                                s < ctx_len-1 ? "," : "]");
                    fprintf(stderr, "\n");
                    free(test_scores);
                }
            }
            for (int d = 0; d < dim; d++) attn_sum += xt[d] * xt[d];
        }

        /* Batched attention output projection + residual */
        matmul(x_seq, lw.attn_o, ao_seq, seq_len, dim, dim);
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < dim; d++) {
                ao_sum += ao_seq[t * dim + d] * ao_seq[t * dim + d];
                x_seq[t * dim + d] = res_seq[t * dim + d] + ao_seq[t * dim + d];
            }
        }

        /* RMS-norm all tokens → xn_seq, then batch FFN */
        for (int t = 0; t < seq_len; t++)
            cpu_rms_norm(x_seq + t * dim, lw.ffn_norm,
                         xn_seq + t * dim, dim, eps);
        for (int t = 0; t < seq_len; t++)
            for (int d = 0; d < dim; d++) xn_sum += xn_seq[t * dim + d] * xn_seq[t * dim + d];

        matmul(xn_seq, lw.ffn_gate, gate_seq, seq_len, dim, ffn);
        matmul(xn_seq, lw.ffn_up,   up_seq,   seq_len, dim, ffn);

        /* SiLU gating per token (element-wise on batched data) */
        for (int t = 0; t < seq_len; t++) {
            float *g = gate_seq + t * ffn;
            float *u = up_seq   + t * ffn;
            cpu_silu(g, ffn);
            for (int d = 0; d < ffn; d++) {
                g[d] *= u[d];
                gate_sum += g[d] * g[d];
            }
        }

        /* Batched FFN down projection + residual */
        matmul(gate_seq, lw.ffn_down, ff_seq, seq_len, ffn, dim);
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < dim; d++) {
                ff_sum += ff_seq[t * dim + d] * ff_seq[t * dim + d];
                /* x_seq = (res_seq + ao_seq) + ff_seq = residual_after_attn + ffn_out */
                x_seq[t * dim + d] = res_seq[t * dim + d] + ao_seq[t * dim + d]
                                   + ff_seq[t * dim + d];
            }
        }

        /* Stats for last token */
        float *x_last = x_seq + (seq_len - 1) * dim;
        float x_n = 0, x_min = 1e9f, x_max = -1e9f;
        int x_nan = 0;
        for (int d = 0; d < dim; d++) {
            x_n += x_last[d] * x_last[d];
            if (isnan(x_last[d])) x_nan++;
            else { if (x_last[d] < x_min) x_min = x_last[d]; if (x_last[d] > x_max) x_max = x_last[d]; }
        }
        fprintf(stderr,
            "[layer %2d] |xn|=%.0f |attn|=%.0f |ao|=%.0f "
            "|gate|=%.0f |ffn|=%.0f |x_last|=%.0f rng[%.1f,%.1f] NaN=%d\n",
            layer, sqrtf(xn_sum / (seq_len * 2)),
            sqrtf(attn_sum / seq_len), sqrtf(ao_sum / seq_len),
            sqrtf(gate_sum / seq_len), sqrtf(ff_sum / seq_len),
            sqrtf(x_n), x_min, x_max, x_nan);

        /* Inspect what each layer's hidden state predicts (via final norm + lm_head) */
        {
            float *insp_norm = (float *)calloc((size_t)dim, sizeof(float));
            float *insp_logits = (float *)calloc((size_t)m.cfg.vocab_size, sizeof(float));
            cpu_rms_norm(x_last, ln_f, insp_norm, dim, eps);
            matmul(insp_norm, lm_head, insp_logits, 1, dim, m.cfg.vocab_size);
            /* Find top 5 token IDs */
            int top5[5] = {0};
            float top5v[5] = {-1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
            for (int v = 0; v < m.cfg.vocab_size; v++) {
                float val = insp_logits[v];
                for (int r = 0; r < 5; r++) {
                    if (val > top5v[r]) {
                        for (int s = 4; s > r; s--) { top5[s] = top5[s-1]; top5v[s] = top5v[s-1]; }
                        top5[r] = v; top5v[r] = val;
                        break;
                    }
                }
            }
            fprintf(stderr, "[layer %2d] top5: ", layer);
            for (int r = 0; r < 5; r++) {
                char tbuf[16];
                xllm_tokenizer_decode(tok, &top5[r], 1, tbuf, sizeof(tbuf));
                fprintf(stderr, "#%d(%.0f,'%s') ", top5[r], top5v[r], tbuf);
            }
            fprintf(stderr, "\n");
            free(insp_norm); free(insp_logits);
        }
    }
#endif /* !SKIP_LAYERS */

    free(xn_seq); free(ao_seq); free(res_seq);
    free(gate_seq); free(up_seq); free(ff_seq);

    /* ── Sample first token ────────────────────────────────────── */
    float *x_last = x_seq + (seq_len - 1) * dim;
    cpu_rms_norm(x_last, ln_f, xn, dim, eps);
    matmul(xn, lm_head, logits, 1, dim, m.cfg.vocab_size);
    int32_t next_token = sample_token(logits, m.cfg.vocab_size, temperature);
    fprintf(stderr, "[text_gen] Prefill done, first token: %d\n", next_token);

    /* Print first token */
    {
        char dbuf[128];
        xllm_tokenizer_decode(tok, &next_token, 1, dbuf, sizeof(dbuf));
        fprintf(stderr, "[gen] %s", dbuf);
        fflush(stderr);
    }

    /* Free prefill-only buffers */
    free(x_seq); free(q_seq); free(k_seq); free(v_seq);

    /* ══════════════════════════════════════════════════════════════
     * DECODE: autoregressive loop with KV cache
     * ══════════════════════════════════════════════════════════════ */
    printf("\n[");
    fflush(stdout);

    float *x = (float *)calloc((size_t)dim, sizeof(float));
    float *q_tok = (float *)calloc((size_t)dim, sizeof(float));
    float *k_tok = (float *)calloc((size_t)kv_dim, sizeof(float));
    float *v_tok = (float *)calloc((size_t)kv_dim, sizeof(float));

    int cur_len = seq_len;
    int n_generated = 0;

    for (int step = 0; step < MAX_NEW_TOKENS; step++) {
        if (next_token == eos_token) {
            fprintf(stderr, "\n[text_gen] EOS reached at step %d\n", step);
            break;
        }

        /* Embed current token */
        for (int d = 0; d < dim; d++)
            x[d] = embd[(int64_t)next_token * dim + d];

        /* Forward through all layers */
#if !SKIP_LAYERS
        for (int layer = 0; layer < m.cfg.n_layer; layer++) {
            if (g_max_layers && layer >= g_max_layers) break;
            LayerWeights lw = g_weights[layer];
            if (!lw.attn_norm) lw.attn_norm = ones_norm;
            if (!lw.ffn_norm)  lw.ffn_norm  = ones_norm;

            if (!lw.attn_q || !lw.attn_k || !lw.attn_v || !lw.attn_o ||
                !lw.ffn_gate || !lw.ffn_up || !lw.ffn_down) {
                goto done_generating;
            }

            /* RMS norm + project Q, K, V for this single token */
            cpu_rms_norm(x, lw.attn_norm, xn, dim, eps);
            matmul(xn, lw.attn_q, q_tok, 1, dim, dim);
            matmul(xn, lw.attn_k, k_tok, 1, dim, kv_dim);
            matmul(xn, lw.attn_v, v_tok, 1, dim, kv_dim);

            /* Q/K norms (Qwen3) — per-head, weight shape = [head_dim] */
#if !SKIP_QK_NORM
            if (lw.attn_q_norm) {
                for (int h = 0; h < n_h; h++)
                    cpu_rms_norm(q_tok + h * hd, lw.attn_q_norm,
                                 q_tok + h * hd, hd, eps);
            }
            if (lw.attn_k_norm) {
                for (int h = 0; h < kv_h; h++)
                    cpu_rms_norm(k_tok + h * hd, lw.attn_k_norm,
                                 k_tok + h * hd, hd, eps);
            }
#endif

            /* RoPE at current position */
            cpu_rope(q_tok, n_h, hd, m.rotary_dim, cur_len, m.rope_theta, m.rope_type);
            cpu_rope(k_tok, kv_h, hd, m.rotary_dim, cur_len, m.rope_theta, m.rope_type);

            /* Save to KV cache */
            float *kc = k_cache + layer * cache_per_layer;
            float *vc = v_cache + layer * cache_per_layer;
            memcpy(kc + cur_len * kv_dim, k_tok, (size_t)kv_dim * sizeof(float));
            memcpy(vc + cur_len * kv_dim, v_tok, (size_t)kv_dim * sizeof(float));

            /* Attention: Q attends to all cached K[0..cur_len], V[0..cur_len] */
            int ctx_len = cur_len + 1;
            memcpy(residual, x, (size_t)dim * sizeof(float));

            for (int h = 0; h < n_h; h++) {
                int kh = h / heads_per_kv;
                const float *qh = q_tok + h * hd;
                float *oh = x + h * hd;

                float *scores = (float *)calloc((size_t)ctx_len, sizeof(float));
                float max_val = -1e9f;
                for (int s = 0; s < ctx_len; s++) {
                    float dot = 0.0f;
                    const float *ks = kc + s * kv_dim + kh * hd;
                    for (int d = 0; d < hd; d++) dot += qh[d] * ks[d];
                    scores[s] = dot * attn_scale;
                    if (scores[s] > max_val) max_val = scores[s];
                }
                float sum_exp = 0.0f;
                for (int s = 0; s < ctx_len; s++) {
                    scores[s] = expf(scores[s] - max_val);
                    sum_exp += scores[s];
                }
                memset(oh, 0, (size_t)hd * sizeof(float));
                for (int s = 0; s < ctx_len; s++) {
                    float w = scores[s] / sum_exp;
                    const float *vs = vc + s * kv_dim + kh * hd;
                    for (int d = 0; d < hd; d++) oh[d] += w * vs[d];
                }
                free(scores);
            }

            /* Output projection + residual */
            matmul(x, lw.attn_o, ao, 1, dim, dim);
            for (int d = 0; d < dim; d++) x[d] = residual[d] + ao[d];

            /* RMS norm + SwiGLU FFN */
            memcpy(residual, x, (size_t)dim * sizeof(float));
            cpu_rms_norm(x, lw.ffn_norm, xn, dim, eps);
            matmul(xn, lw.ffn_gate, gate, 1, dim, ffn);
            matmul(xn, lw.ffn_up,   up,   1, dim, ffn);
            cpu_silu(gate, ffn);
            for (int d = 0; d < ffn; d++) gate[d] *= up[d];
            matmul(gate, lw.ffn_down, ffn_out, 1, ffn, dim);
            for (int d = 0; d < dim; d++) x[d] = residual[d] + ffn_out[d];

        }
#endif /* !SKIP_LAYERS */

        /* Final norm + lm_head */
        cpu_rms_norm(x, ln_f, xn, dim, eps);
        matmul(xn, lm_head, logits, 1, dim, m.cfg.vocab_size);

        /* Sample */
        next_token = sample_token(logits, m.cfg.vocab_size, temperature);

        /* Debug: first 3 decode steps */
        if (n_generated < 3) {
            float xn_n = 0, lg_max = -1e9f, lg_sum = 0;
            int lg_nan = 0;
            for (int d = 0; d < dim; d++) xn_n += xn[d] * xn[d];
            for (int32_t v = 0; v < m.cfg.vocab_size; v++) {
                if (isnan(logits[v])) lg_nan++;
                if (logits[v] > lg_max) lg_max = logits[v];
            }
            /* top-30 tokens */
            int32_t top[30] = {0};
            float topv[30];
            for (int i = 0; i < 30; i++) topv[i] = -1e9f;
            for (int32_t v = 0; v < m.cfg.vocab_size; v++) {
                for (int i = 0; i < 30; i++) {
                    if (logits[v] > topv[i]) {
                        for (int j = 29; j > i; j--) { top[j] = top[j-1]; topv[j] = topv[j-1]; }
                        top[i] = v; topv[i] = logits[v];
                        break;
                    }
                }
            }
            fprintf(stderr, "[gen %d] |xn|=%.0f lg_max=%.1f lg_nan=%d top-30:\n",
                    n_generated, sqrtf(xn_n), lg_max, lg_nan);
            for (int i = 0; i < 30; i++) {
                char tbuf[32];
                xllm_tokenizer_decode(tok, &top[i], 1, tbuf, sizeof(tbuf));
                fprintf(stderr, "  %d:%.0f[%s]", top[i], topv[i], tbuf);
                if (i % 5 == 4) fprintf(stderr, "\n");
            }
            /* Also sample a few known Chinese tokens */
            int32_t check_ids[] = {151643, 151644, 151645, 77091, 233, 100168, 99478};
            const char *check_names[] = {"BOS", "?", "EOS", "?", "?", "好?", "的?"};
            int n_check = sizeof(check_ids) / sizeof(check_ids[0]);
            fprintf(stderr, "  Chinese spot-check:");
            for (int i = 0; i < n_check; i++) {
                if (check_ids[i] < m.cfg.vocab_size)
                    fprintf(stderr, " [%d]=%.1f", check_ids[i], logits[check_ids[i]]);
            }
            fprintf(stderr, "\n");
        }

        /* Decode and print */
        {
            char dbuf[128];
            xllm_tokenizer_decode(tok, &next_token, 1, dbuf, sizeof(dbuf));
            printf("%s", dbuf);
            fflush(stdout);
        }

        cur_len++;
        n_generated++;

        /* Progress to stderr every 16 tokens */
        if ((n_generated & 15) == 0)
            fprintf(stderr, "[text_gen] %d tokens generated (pos=%d)\n",
                    n_generated, cur_len);
    }

done_generating:
    printf("]\n");
    fprintf(stderr, "[text_gen] Generated %d tokens (total seq=%d)\n",
            n_generated, cur_len);

    /* Cleanup */
    free_all_weights(g_weights, n_layers_to_load);
    free(x); free(q_tok); free(k_tok); free(v_tok);
    free(residual); free(xn);
    free(ao);
    free(gate); free(up); free(ffn_out);
    free(logits);
    free(ones_norm);
    free(k_cache); free(v_cache);
    free(ln_f); free(lm_head);
    free(embd);
    fclose(f);
    xllm_tokenizer_free(tok);
    gguf_free(ctx);
#ifdef USE_CUDA
    gpu_matmul_cleanup();
#endif

    return 0;
}
