/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * GGUF Model Loader — full GGUF v3 format parser in pure C11.
 *
 * Implements two-pass loading:
 *   Pass 1 (gguf_init_from_file): parse header, KV pairs, tensor metadata
 *   Pass 2 (gguf_load_tensor_data): load individual tensor data
 *
 * Design based on llama.cpp's gguf.cpp / llama-model-loader.cpp.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stddef.h>
#include <math.h>

#include "model_loader.h"
#include "weight_loader.h"

/* ── Internal padding macro ────────────────────────────────────────────── */

#define GGML_PAD(x, n) ((((x) + (n) - 1) / (n)) * (n))

/* ── GGUF type size table ──────────────────────────────────────────────── */

static const size_t gguf_type_size_table[] = {
    [GGUF_TYPE_UINT8]   = 1,
    [GGUF_TYPE_INT8]    = 1,
    [GGUF_TYPE_UINT16]  = 2,
    [GGUF_TYPE_INT16]   = 2,
    [GGUF_TYPE_UINT32]  = 4,
    [GGUF_TYPE_INT32]   = 4,
    [GGUF_TYPE_FLOAT32] = 4,
    [GGUF_TYPE_BOOL]    = 1,
    [GGUF_TYPE_STRING]  = 0,  /* variable length */
    [GGUF_TYPE_ARRAY]   = 0,  /* wrapper type */
    [GGUF_TYPE_UINT64]  = 8,
    [GGUF_TYPE_INT64]   = 8,
    [GGUF_TYPE_FLOAT64] = 8,
};

static const char * const gguf_type_name_table[] = {
    [GGUF_TYPE_UINT8]   = "u8",
    [GGUF_TYPE_INT8]    = "i8",
    [GGUF_TYPE_UINT16]  = "u16",
    [GGUF_TYPE_INT16]   = "i16",
    [GGUF_TYPE_UINT32]  = "u32",
    [GGUF_TYPE_INT32]   = "i32",
    [GGUF_TYPE_FLOAT32] = "f32",
    [GGUF_TYPE_BOOL]    = "bool",
    [GGUF_TYPE_STRING]  = "str",
    [GGUF_TYPE_ARRAY]   = "arr",
    [GGUF_TYPE_UINT64]  = "u64",
    [GGUF_TYPE_INT64]   = "i64",
    [GGUF_TYPE_FLOAT64] = "f64",
};

static const char * const ggml_type_name_table[] = {
    [GGML_TYPE_F32]     = "f32",
    [GGML_TYPE_F16]     = "f16",
    [GGML_TYPE_Q4_0]    = "q4_0",
    [GGML_TYPE_Q4_1]    = "q4_1",
    [GGML_TYPE_Q5_0]    = "q5_0",
    [GGML_TYPE_Q5_1]    = "q5_1",
    [GGML_TYPE_Q8_0]    = "q8_0",
    [GGML_TYPE_Q8_1]    = "q8_1",
    [GGML_TYPE_Q2_K]    = "q2_K",
    [GGML_TYPE_Q3_K]    = "q3_K",
    [GGML_TYPE_Q4_K]    = "q4_K",
    [GGML_TYPE_Q5_K]    = "q5_K",
    [GGML_TYPE_Q6_K]    = "q6_K",
    [GGML_TYPE_Q8_K]    = "q8_K",
    [GGML_TYPE_IQ2_XXS] = "iq2_xxs",
    [GGML_TYPE_IQ2_XS]  = "iq2_xs",
    [GGML_TYPE_IQ3_XXS] = "iq3_xxs",
    [GGML_TYPE_IQ1_S]   = "iq1_s",
    [GGML_TYPE_BF16]    = "bf16",
};

static const size_t ggml_type_size_table[] = {
    [GGML_TYPE_F32]     = 4,
    [GGML_TYPE_F16]     = 2,
    [GGML_TYPE_Q4_0]    = 0,  /* quantized — use ggml_type_size() */
    [GGML_TYPE_Q4_1]    = 0,
    [GGML_TYPE_Q5_0]    = 0,
    [GGML_TYPE_Q5_1]    = 0,
    [GGML_TYPE_Q8_0]    = 0,
    [GGML_TYPE_Q8_1]    = 0,
    [GGML_TYPE_Q2_K]    = 0,
    [GGML_TYPE_Q3_K]    = 0,
    [GGML_TYPE_Q4_K]    = 0,
    [GGML_TYPE_Q5_K]    = 0,
    [GGML_TYPE_Q6_K]    = 0,
    [GGML_TYPE_Q8_K]    = 0,
    [GGML_TYPE_IQ2_XXS] = 0,
    [GGML_TYPE_IQ2_XS]  = 0,
    [GGML_TYPE_IQ3_XXS] = 0,
    [GGML_TYPE_IQ1_S]   = 0,
    [GGML_TYPE_BF16]    = 2,
};

/* ── Block sizes for quantized types ──────────────────────────────────── */

static const int64_t ggml_blck_size_table[] = {
    [GGML_TYPE_F32]     = 1,
    [GGML_TYPE_F16]     = 1,
    [GGML_TYPE_Q4_0]    = 32,
    [GGML_TYPE_Q4_1]    = 32,
    [GGML_TYPE_Q5_0]    = 32,
    [GGML_TYPE_Q5_1]    = 32,
    [GGML_TYPE_Q8_0]    = 32,
    [GGML_TYPE_Q8_1]    = 32,
    [GGML_TYPE_Q2_K]    = 256,
    [GGML_TYPE_Q3_K]    = 256,
    [GGML_TYPE_Q4_K]    = 256,
    [GGML_TYPE_Q5_K]    = 256,
    [GGML_TYPE_Q6_K]    = 256,
    [GGML_TYPE_Q8_K]    = 256,
    [GGML_TYPE_IQ2_XXS] = 256,
    [GGML_TYPE_IQ2_XS]  = 256,
    [GGML_TYPE_IQ3_XXS] = 256,
    [GGML_TYPE_IQ1_S]   = 256,
    [GGML_TYPE_BF16]    = 1,
};

/* Quantized type element sizes (bytes per block / block_size).
 * These are approximate — actual size depends on shape alignment. */
static const size_t q_type_block_bytes[] = {
    [GGML_TYPE_Q4_0]    = 18,   /* 16 * 0.5 + 2 bytes scale */
    [GGML_TYPE_Q4_1]    = 20,   /* 16 * 0.5 + 2 bytes scale + 2 bytes min */
    [GGML_TYPE_Q5_0]    = 22,   /* 32 * 0.5 + 4 + 2 bytes scale */
    [GGML_TYPE_Q5_1]    = 24,   /* similar with min */
    [GGML_TYPE_Q8_0]    = 34,   /* 32 * 1 + 2 bytes scale */
    [GGML_TYPE_Q8_1]    = 36,
    [GGML_TYPE_Q2_K]    = 82,   /* 256 * 2/8 + scale/min buffers */
    [GGML_TYPE_Q3_K]    = 110,
    [GGML_TYPE_Q4_K]    = 144,
    [GGML_TYPE_Q5_K]    = 176,
    [GGML_TYPE_Q6_K]    = 210,
    [GGML_TYPE_Q8_K]    = 292,
    [GGML_TYPE_IQ2_XXS] = 66,
    [GGML_TYPE_IQ2_XS]  = 74,
    [GGML_TYPE_IQ3_XXS] = 98,
    [GGML_TYPE_IQ1_S]   = 54,
};

/* ── GGUF context (opaque) ─────────────────────────────────────────────── */

struct GgufContext {
    uint32_t         version;
    size_t           alignment;
    uint64_t         data_offset;     /* file offset where data section starts */
    size_t           data_size;       /* total size of data section */

    GgufKV          *kv_pairs;        /* dynamic array */
    int64_t          n_kv;            /* number of KV pairs */
    int64_t          kv_capacity;     /* allocated capacity */

    GgufTensorInfo  *tensors;         /* dynamic array */
    int64_t          n_tensors;       /* number of tensors */
    int64_t          tensor_capacity; /* allocated capacity */
};

/* ── Public type utilities ─────────────────────────────────────────────── */

size_t gguf_type_size(GgufType type) {
    if (type < 0 || type >= GGUF_TYPE_COUNT) return 0;
    return gguf_type_size_table[type];
}

const char *gguf_type_name(GgufType type) {
    if (type < 0 || type >= GGUF_TYPE_COUNT) return NULL;
    return gguf_type_name_table[type];
}

const char *ggml_type_name(GgmlType type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) return NULL;
    return ggml_type_name_table[type];
}

size_t ggml_type_size(GgmlType type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) return 0;
    return ggml_type_size_table[type];
}

int64_t ggml_blck_size(GgmlType type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) return 1;
    return ggml_blck_size_table[type];
}

/* ── Low-level file I/O helpers ────────────────────────────────────────── */

static uint32_t read_u32_le(FILE *f) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    return (uint32_t)buf[0]
         | ((uint32_t)buf[1] << 8)
         | ((uint32_t)buf[2] << 16)
         | ((uint32_t)buf[3] << 24);
}

static int32_t read_i32_le(FILE *f) {
    return (int32_t)read_u32_le(f);
}

static uint64_t read_u64_le(FILE *f) {
    uint32_t lo = read_u32_le(f);
    uint32_t hi = read_u32_le(f);
    return (uint64_t)lo | ((uint64_t)hi << 32);
}

static int64_t read_i64_le(FILE *f) {
    return (int64_t)read_u64_le(f);
}

static float read_f32_le(FILE *f) {
    uint32_t u = read_u32_le(f);
    float val;
    memcpy(&val, &u, sizeof(val));
    return val;
}

static double read_f64_le(FILE *f) {
    uint64_t u = read_u64_le(f);
    double d;
    memcpy(&d, &u, sizeof(d));
    return d;
}

static char *read_string(FILE *f, size_t max_len) {
    uint64_t len = read_u64_le(f);
    if (len > max_len) {
        fprintf(stderr, "[xLLM] String too long: %" PRIu64 " > %zu\n", len, max_len);
        return NULL;
    }
    char *s = (char *)calloc(len + 1, 1);
    if (!s) return NULL;
    if (len > 0 && fread(s, 1, len, f) != len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

/* ── GGUF header parser ────────────────────────────────────────────────── */

static bool parse_kv_pairs(FILE *f, GgufContext *ctx, int64_t n_kv) {
    ctx->kv_pairs = (GgufKV *)calloc((size_t)n_kv, sizeof(GgufKV));
    if (!ctx->kv_pairs) return false;
    ctx->n_kv = n_kv;
    ctx->kv_capacity = n_kv;

    for (int64_t i = 0; i < n_kv; i++) {
        GgufKV *kv = &ctx->kv_pairs[i];

        /* Read key string */
        kv->key = read_string(f, GGUF_MAX_STRING_LENGTH);
        if (!kv->key) return false;

        /* Check for duplicate keys */
        for (int64_t j = 0; j < i; j++) {
            if (strcmp(kv->key, ctx->kv_pairs[j].key) == 0) {
                fprintf(stderr, "[xLLM] Duplicate GGUF key: '%s'\n", kv->key);
                return false;
            }
        }

        /* Read type */
        int32_t type_raw = read_i32_le(f);
        if (type_raw < 0 || type_raw >= GGUF_TYPE_COUNT) {
            fprintf(stderr, "[xLLM] Invalid GGUF type %d for key '%s'\n",
                    type_raw, kv->key);
            return false;
        }
        kv->type = (GgufType)type_raw;

        /* Handle array type */
        if (kv->type == GGUF_TYPE_ARRAY) {
            kv->is_array = true;
            int32_t arr_type_raw = read_i32_le(f);
            if (arr_type_raw < 0 || arr_type_raw >= GGUF_TYPE_COUNT
                || arr_type_raw == GGUF_TYPE_ARRAY) {
                fprintf(stderr, "[xLLM] Invalid array type %d for key '%s'\n",
                        arr_type_raw, kv->key);
                return false;
            }
            kv->array_type = (GgufType)arr_type_raw;
            kv->type = kv->array_type;  /* store element type as main type */
            kv->array_count = read_u64_le(f);
        } else {
            kv->is_array = false;
            kv->array_type = kv->type;
            kv->array_count = 1;
        }

        /* Read value(s) */
        kv->value_offset = ftell(f);
        size_t tsize = gguf_type_size(kv->type);
        uint64_t count = kv->array_count;

        /* For strings, handle single values and arrays of strings */
        if (kv->type == GGUF_TYPE_STRING) {
            if (count != 1) {
                /* Skip string array elements: each is a 64-bit length + data */
                for (uint64_t j = 0; j < count; j++) {
                    uint64_t slen = read_u64_le(f);
                    if (fseek(f, (long)slen, SEEK_CUR) != 0) return false;
                }
                continue;
            }
            kv->string_val = read_string(f, GGUF_MAX_STRING_LENGTH);
            if (!kv->string_val) return false;
            continue;
        }

        /* Non-string values */
        if (tsize == 0) {
            fprintf(stderr, "[xLLM] Unknown type size for type %d\n", kv->type);
            return false;
        }

        /* Read first (or only) element into union */
        switch (kv->type) {
        case GGUF_TYPE_UINT8:   kv->uint8_val   = (uint8_t)fgetc(f); break;
        case GGUF_TYPE_INT8:    kv->int8_val    = (int8_t)fgetc(f); break;
        case GGUF_TYPE_BOOL:    kv->bool_val    = (fgetc(f) != 0); break;
        case GGUF_TYPE_UINT16:  kv->uint16_val  = (uint16_t)(read_u32_le(f) & 0xFFFF); break;
        case GGUF_TYPE_INT16:   kv->int16_val   = (int16_t)(read_i32_le(f) & 0xFFFF); break;
        case GGUF_TYPE_UINT32:  kv->uint32_val  = read_u32_le(f); break;
        case GGUF_TYPE_INT32:   kv->int32_val   = read_i32_le(f); break;
        case GGUF_TYPE_FLOAT32: kv->float32_val = read_f32_le(f); break;
        case GGUF_TYPE_UINT64:  kv->uint64_val  = read_u64_le(f); break;
        case GGUF_TYPE_INT64:   kv->int64_val   = read_i64_le(f); break;
        case GGUF_TYPE_FLOAT64: kv->float64_val = read_f64_le(f); break;
        default:
            fprintf(stderr, "[xLLM] Unhandled GGUF type %d\n", kv->type);
            return false;
        }

        /* Skip remaining array elements if count > 1 */
        if (count > 1) {
            size_t skip = (size_t)((count - 1) * tsize);
            if (fseek(f, (long)skip, SEEK_CUR) != 0) return false;
        }
    }
    return true;
}

static bool parse_tensor_info(FILE *f, GgufContext *ctx, int64_t n_tensors) {
    ctx->tensors = (GgufTensorInfo *)calloc((size_t)n_tensors, sizeof(GgufTensorInfo));
    if (!ctx->tensors) return false;
    ctx->n_tensors = n_tensors;
    ctx->tensor_capacity = n_tensors;

    for (int64_t i = 0; i < n_tensors; i++) {
        GgufTensorInfo *ti = &ctx->tensors[i];

        /* Read tensor name */
        char *name = read_string(f, GGUF_MAX_STRING_LENGTH);
        if (!name) return false;
        if (strlen(name) >= sizeof(ti->name)) {
            fprintf(stderr, "[xLLM] Tensor name too long: '%s'\n", name);
            free(name);
            return false;
        }
        strncpy(ti->name, name, sizeof(ti->name) - 1);
        ti->name[sizeof(ti->name) - 1] = '\0';
        free(name);

        /* Check for duplicate names */
        for (int64_t j = 0; j < i; j++) {
            if (strcmp(ti->name, ctx->tensors[j].name) == 0) {
                fprintf(stderr, "[xLLM] Duplicate tensor name: '%s'\n", ti->name);
                return false;
            }
        }

        /* Read dimensions */
        ti->n_dims = read_u32_le(f);
        if (ti->n_dims > GGUF_MAX_DIMS) {
            fprintf(stderr, "[xLLM] Tensor '%s': n_dims %u > %d\n",
                    ti->name, ti->n_dims, GGUF_MAX_DIMS);
            return false;
        }

        /* Initialize all dimensions to 1 */
        for (int d = 0; d < GGUF_MAX_DIMS; d++) {
            ti->ne[d] = 1;
        }

        /* Read each dimension */
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            ti->ne[d] = read_i64_le(f);
            if (ti->ne[d] < 0) {
                fprintf(stderr, "[xLLM] Tensor '%s': negative dimension %" PRId64 "\n",
                        ti->name, ti->ne[d]);
                return false;
            }
        }

        /* Read ggml type */
        int32_t type_raw = read_i32_le(f);
        if (type_raw < 0 || type_raw >= GGML_TYPE_COUNT) {
            fprintf(stderr, "[xLLM] Tensor '%s': invalid ggml type %d\n",
                    ti->name, type_raw);
            return false;
        }
        ti->ggml_type = (GgmlType)type_raw;

        /* Read offset */
        ti->offset = read_u64_le(f);

        /* Calculate byte strides and total nbytes */
        size_t elem_size = ggml_type_size(ti->ggml_type);
        int64_t blck = ggml_blck_size(ti->ggml_type);
        size_t block_bytes = 0;

        /* For quantized types, calculate effective bytes */
        if (elem_size == 0 && blck > 1) {
            /* Quantized: use block-based size calculation */
            if (ti->ggml_type >= 0 && ti->ggml_type < GGML_TYPE_COUNT) {
                block_bytes = q_type_block_bytes[ti->ggml_type];
            }
            if (block_bytes == 0) {
                fprintf(stderr, "[xLLM] Tensor '%s': unsupported quantized type %s (%d)\n",
                        ti->name, ggml_type_name(ti->ggml_type), ti->ggml_type);
                return false;
            }
            ti->nb[0] = block_bytes / (size_t)blck;  /* approximate byte stride */
        } else if (elem_size > 0) {
            ti->nb[0] = elem_size;
        } else {
            ti->nb[0] = 4;  /* default to f32 */
        }

        ti->nb[1] = ti->nb[0] * (size_t)(ti->ne[0] / (blck > 0 ? blck : 1));
        ti->nb[2] = ti->nb[1] * (size_t)ti->ne[1];
        ti->nb[3] = ti->nb[2] * (size_t)ti->ne[2];

        /* Compute nbytes directly: total elements / block_size * block_bytes */
        {
            int64_t nelem = ti->ne[0] * ti->ne[1] * ti->ne[2] * ti->ne[3];
            if (blck > 1 && block_bytes > 0) {
                ti->nbytes = (size_t)(((size_t)nelem + (size_t)blck - 1)
                            / (size_t)blck * block_bytes);
            } else {
                ti->nbytes = (size_t)nelem * elem_size;
            }
        }
    }
    return true;
}

GgufContext *gguf_init_from_file(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[xLLM] Cannot open GGUF file: %s\n", path);
        return NULL;
    }

    /* Validate magic */
    uint32_t magic = read_u32_le(f);
    if (magic != GGUF_MAGIC_U32) {
        fprintf(stderr, "[xLLM] Not a GGUF file (magic 0x%08X, expected 0x%08X)\n",
                magic, GGUF_MAGIC_U32);
        fclose(f);
        return NULL;
    }

    GgufContext *ctx = (GgufContext *)calloc(1, sizeof(GgufContext));
    if (!ctx) { fclose(f); return NULL; }

    /* Read version */
    ctx->version = read_u32_le(f);
    if (ctx->version < 2 || ctx->version > GGUF_VERSION) {
        fprintf(stderr, "[xLLM] Unsupported GGUF version %u (expected 2 or 3)\n",
                ctx->version);
        goto fail;
    }

    /* Check endianness mismatch */
    if ((ctx->version & 0x0000FFFF) == 0x00000000) {
        fprintf(stderr, "[xLLM] Possible endianness mismatch in GGUF file\n");
        goto fail;
    }

    /* Read tensor and KV counts */
    int64_t n_tensors = read_i64_le(f);
    int64_t n_kv = read_i64_le(f);

    if (n_tensors < 0 || n_tensors > GGUF_MAX_TENSORS) {
        fprintf(stderr, "[xLLM] Invalid tensor count: %" PRId64 "\n", n_tensors);
        goto fail;
    }
    if (n_kv < 0 || n_kv > GGUF_MAX_KV_PAIRS) {
        fprintf(stderr, "[xLLM] Invalid KV pair count: %" PRId64 "\n", n_kv);
        goto fail;
    }

    /* Parse KV pairs */
    if (n_kv > 0) {
        if (!parse_kv_pairs(f, ctx, n_kv)) {
            fprintf(stderr, "[xLLM] Failed to parse KV pairs\n");
            goto fail;
        }
    }

    /* Extract alignment from metadata */
    ctx->alignment = (size_t)gguf_get_val_u32(ctx,
        GGUF_KEY_GENERAL_ALIGNMENT, GGUF_DEFAULT_ALIGNMENT);
    if (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
        fprintf(stderr, "[xLLM] Alignment %zu is not a power of 2\n",
                ctx->alignment);
        goto fail;
    }

    /* Parse tensor info */
    if (n_tensors > 0) {
        if (!parse_tensor_info(f, ctx, n_tensors)) {
            fprintf(stderr, "[xLLM] Failed to parse tensor info\n");
            goto fail;
        }
    }

    /* Align to data section */
    {
        long current_pos = ftell(f);
        long aligned = (long)GGML_PAD((size_t)current_pos, ctx->alignment);
        if (aligned > current_pos) {
            if (fseek(f, aligned, SEEK_SET) != 0) {
                fprintf(stderr, "[xLLM] Failed to seek to data section\n");
                goto fail;
            }
        }
        ctx->data_offset = (uint64_t)aligned;
    }

    /* Compute total data section size by summing all tensor sizes */
    ctx->data_size = 0;
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        GgufTensorInfo *ti = &ctx->tensors[i];
        size_t padded = GGML_PAD(ti->nbytes, ctx->alignment);
        ctx->data_size += padded;
    }

    fclose(f);
    return ctx;

fail:
    fclose(f);
    gguf_free(ctx);
    return NULL;
}

void gguf_free(GgufContext *ctx) {
    if (!ctx) return;
    for (int64_t i = 0; i < ctx->n_kv; i++) {
        free(ctx->kv_pairs[i].key);
        if (ctx->kv_pairs[i].type == GGUF_TYPE_STRING) {
            free(ctx->kv_pairs[i].string_val);
        }
    }
    free(ctx->kv_pairs);
    free(ctx->tensors);
    free(ctx);
}

/* ── Context accessors ─────────────────────────────────────────────────── */

uint32_t gguf_get_version(const GgufContext *ctx) { return ctx->version; }
size_t gguf_get_alignment(const GgufContext *ctx) { return ctx->alignment; }
uint64_t gguf_get_data_offset(const GgufContext *ctx) { return ctx->data_offset; }
size_t gguf_get_data_size(const GgufContext *ctx) { return ctx->data_size; }

int64_t gguf_get_n_kv(const GgufContext *ctx) { return ctx->n_kv; }

int64_t gguf_find_key(const GgufContext *ctx, const char *key) {
    for (int64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv_pairs[i].key, key) == 0)
            return i;
    }
    return -1;
}

const char *gguf_get_key(const GgufContext *ctx, int64_t key_id) {
    if (key_id < 0 || key_id >= ctx->n_kv) return NULL;
    return ctx->kv_pairs[key_id].key;
}

GgufType gguf_get_kv_type(const GgufContext *ctx, int64_t key_id) {
    if (key_id < 0 || key_id >= ctx->n_kv) return (GgufType)(-1);
    return ctx->kv_pairs[key_id].is_array ?
        GGUF_TYPE_ARRAY : ctx->kv_pairs[key_id].type;
}

/* ── Typed value getters ───────────────────────────────────────────────── */

const char *gguf_get_val_str(const GgufContext *ctx, const char *key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return NULL;
    if (ctx->kv_pairs[idx].type != GGUF_TYPE_STRING) return NULL;
    return ctx->kv_pairs[idx].string_val;
}

int64_t gguf_get_val_int(const GgufContext *ctx, const char *key, int64_t default_val) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    const GgufKV *kv = &ctx->kv_pairs[idx];
    switch (kv->type) {
    case GGUF_TYPE_UINT8:   return kv->uint8_val;
    case GGUF_TYPE_INT8:    return kv->int8_val;
    case GGUF_TYPE_UINT16:  return kv->uint16_val;
    case GGUF_TYPE_INT16:   return kv->int16_val;
    case GGUF_TYPE_UINT32:  return kv->uint32_val;
    case GGUF_TYPE_INT32:   return kv->int32_val;
    case GGUF_TYPE_UINT64:  return (int64_t)kv->uint64_val;
    case GGUF_TYPE_INT64:   return kv->int64_val;
    case GGUF_TYPE_BOOL:    return kv->bool_val ? 1 : 0;
    default: return default_val;
    }
}

float gguf_get_val_f32(const GgufContext *ctx, const char *key, float default_val) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    if (ctx->kv_pairs[idx].type == GGUF_TYPE_FLOAT32)
        return ctx->kv_pairs[idx].float32_val;
    return default_val;
}

uint32_t gguf_get_val_u32(const GgufContext *ctx, const char *key, uint32_t default_val) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    const GgufKV *kv = &ctx->kv_pairs[idx];
    switch (kv->type) {
    case GGUF_TYPE_UINT8:   return kv->uint8_val;
    case GGUF_TYPE_UINT16:  return kv->uint16_val;
    case GGUF_TYPE_UINT32:  return kv->uint32_val;
    default: return default_val;
    }
}

bool gguf_get_val_bool(const GgufContext *ctx, const char *key, bool default_val) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    if (ctx->kv_pairs[idx].type == GGUF_TYPE_BOOL)
        return ctx->kv_pairs[idx].bool_val;
    return default_val;
}

/* ── Tensor info access ────────────────────────────────────────────────── */

int64_t gguf_get_n_tensors(const GgufContext *ctx) { return ctx->n_tensors; }

int64_t gguf_find_tensor(const GgufContext *ctx, const char *name) {
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0)
            return i;
    }
    return -1;
}

const GgufTensorInfo *gguf_get_tensor_info(const GgufContext *ctx, int64_t tensor_id) {
    if (tensor_id < 0 || tensor_id >= ctx->n_tensors) return NULL;
    return &ctx->tensors[tensor_id];
}

const GgufTensorInfo *gguf_get_tensor_by_name(const GgufContext *ctx, const char *name) {
    int64_t id = gguf_find_tensor(ctx, name);
    if (id < 0) return NULL;
    return &ctx->tensors[id];
}

/* ── Architecture detection ────────────────────────────────────────────── */

ModelArch gguf_detect_architecture(const GgufContext *ctx) {
    const char *arch = gguf_get_val_str(ctx, GGUF_KEY_ARCHITECTURE);
    if (!arch) return ARCH_UNKNOWN;

    if (strstr(arch, "llama") || strstr(arch, "LLaMA"))   return ARCH_LLAMA;
    if (strstr(arch, "mistral"))   return ARCH_MISTRAL;
    if (strstr(arch, "qwen2"))     return ARCH_QWEN2;
    if (strstr(arch, "deepseek"))  return ARCH_DEEPSEEK;
    if (strstr(arch, "falcon"))    return ARCH_FALCON;
    if (strstr(arch, "gemma"))     return ARCH_GEMMA;
    if (strstr(arch, "phi"))       return ARCH_PHI;

    return ARCH_UNKNOWN;
}

const char *gguf_detect_arch_from_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    uint32_t magic = read_u32_le(f);
    if (magic != GGUF_MAGIC_U32) {
        fclose(f);
        return NULL;
    }

    /* Skip version, tensor count, kv count */
    fseek(f, 16, SEEK_SET);  /* 4 + 4 + 8 = 16 bytes from start */

    /* Quick scan for general.architecture KV */
    GgufContext *ctx = gguf_init_from_file(path);
    if (!ctx) return NULL;

    const char *name = model_arch_name(gguf_detect_architecture(ctx));
    gguf_free(ctx);
    return name;
}

/* ── Model config loading from GGUF metadata ───────────────────────────── */

/* Helper: try arch.key first, then llama.key, then default */
static int64_t gguf_get_val_int_arch(const GgufContext *ctx, const char *arch,
                                      const char *suffix, int64_t default_val) {
    char key[256];
    if (arch && arch[0]) {
        snprintf(key, sizeof(key), "%s.%s", arch, suffix);
        int64_t v = gguf_get_val_int(ctx, key, INT64_MIN);
        if (v != INT64_MIN) return v;
    }
    snprintf(key, sizeof(key), "llama.%s", suffix);
    return gguf_get_val_int(ctx, key, default_val);
}

static float gguf_get_val_f32_arch(const GgufContext *ctx, const char *arch,
                                    const char *suffix, float default_val) {
    char key[256];
    if (arch && arch[0]) {
        snprintf(key, sizeof(key), "%s.%s", arch, suffix);
        int64_t idx = gguf_find_key(ctx, key);
        if (idx >= 0 && ctx->kv_pairs[idx].type == GGUF_TYPE_FLOAT32)
            return ctx->kv_pairs[idx].float32_val;
    }
    snprintf(key, sizeof(key), "llama.%s", suffix);
    return gguf_get_val_f32(ctx, key, default_val);
}

bool gguf_load_model_config(const GgufContext *ctx, Gpt2Config *cfg_out) {
    if (!ctx || !cfg_out) return false;
    memset(cfg_out, 0, sizeof(*cfg_out));

    const char *arch = gguf_get_val_str(ctx, GGUF_KEY_ARCHITECTURE);

    /* Read config with architecture-specific key fallback */
    cfg_out->vocab_size  = (int32_t)gguf_get_val_int_arch(ctx, arch,
        "vocab_size", 32000);
    cfg_out->n_positions = (int32_t)gguf_get_val_int_arch(ctx, arch,
        "context_length", 40960);
    cfg_out->n_embd      = (int32_t)gguf_get_val_int_arch(ctx, arch,
        "embedding_length", 4096);
    cfg_out->n_layer     = (int32_t)gguf_get_val_int_arch(ctx, arch,
        "block_count", 32);
    cfg_out->n_head      = (int32_t)gguf_get_val_int_arch(ctx, arch,
        "attention.head_count", 32);

    /* FFN dimension — try arch key, then llama, then use 8/3 ratio */
    int64_t ffn = gguf_get_val_int_arch(ctx, arch, "feed_forward_length", 0);
    if (ffn > 0) {
        cfg_out->n_inner = (int32_t)ffn;
    } else {
        cfg_out->n_inner = cfg_out->n_embd * 8 / 3;  /* common Llama ratio */
    }

    cfg_out->head_size      = cfg_out->n_embd / cfg_out->n_head;
    cfg_out->layer_norm_eps = gguf_get_val_f32_arch(ctx, arch,
        "attention.layer_norm_rms_epsilon", 1e-5f);

    /* Validate */
    if (cfg_out->n_embd == 0 || cfg_out->n_layer == 0 || cfg_out->n_head == 0) {
        fprintf(stderr, "[xLLM] Invalid model config from GGUF: "
                "embd=%d layers=%d heads=%d\n",
                cfg_out->n_embd, cfg_out->n_layer, cfg_out->n_head);
        return false;
    }
    if (cfg_out->n_embd % cfg_out->n_head != 0) {
        fprintf(stderr, "[xLLM] n_embd (%d) not divisible by n_head (%d)\n",
                cfg_out->n_embd, cfg_out->n_head);
        return false;
    }

    return true;
}

/* ── Tensor data loading ───────────────────────────────────────────────── */

bool gguf_load_tensor_data(FILE *file, const GgufContext *ctx,
                           const GgufTensorInfo *ti, void *dst) {
    if (!file || !ctx || !ti || !dst) return false;

    uint64_t abs_offset = ctx->data_offset + ti->offset;
#ifdef _WIN32
    if (_fseeki64(file, (int64_t)abs_offset, SEEK_SET) != 0) {
#else
    if (fseeko(file, (off_t)abs_offset, SEEK_SET) != 0) {
#endif
        fprintf(stderr, "[xLLM] Failed to seek to tensor '%s' at offset %" PRIu64 "\n",
                ti->name, abs_offset);
        return false;
    }

    size_t nread = fread(dst, 1, ti->nbytes, file);
    if (nread != ti->nbytes) {
        fprintf(stderr, "[xLLM] Failed to read tensor '%s': "
                "expected %zu bytes, got %zu\n", ti->name, ti->nbytes, nread);
        return false;
    }

    return true;
}

/* ── Summary printer ───────────────────────────────────────────────────── */

void gguf_print_summary(const GgufContext *ctx) {
    if (!ctx) {
        printf("(null GGUF context)\n");
        return;
    }

    printf("═══ GGUF Model Summary ═══\n");
    printf("GGUF Version:    %u\n", ctx->version);
    printf("Alignment:       %zu bytes\n", ctx->alignment);
    printf("KV Pairs:        %" PRId64 "\n", ctx->n_kv);
    printf("Tensors:         %" PRId64 "\n", ctx->n_tensors);
    printf("Data Offset:     %" PRIu64 "\n", ctx->data_offset);
    printf("Data Size:       %zu bytes (%.2f MiB)\n",
           ctx->data_size, (double)ctx->data_size / (1024.0 * 1024.0));

    /* Architecture */
    printf("Architecture:    %s\n", model_arch_name(gguf_detect_architecture(ctx)));

    /* Key metadata */
    const char *model_name = gguf_get_val_str(ctx, GGUF_KEY_MODEL_NAME);
    if (model_name) printf("Model Name:      %s\n", model_name);

    int64_t vocab = gguf_get_val_int(ctx, "llama.vocab_size", -1);
    int64_t ctx_len = gguf_get_val_int(ctx, "llama.context_length", -1);
    int64_t n_embd = gguf_get_val_int(ctx, "llama.embedding_length", -1);
    int64_t n_layer = gguf_get_val_int(ctx, "llama.block_count", -1);
    int64_t n_head = gguf_get_val_int(ctx, "llama.attention.head_count", -1);
    int64_t n_head_kv = gguf_get_val_int(ctx, "llama.attention.head_count_kv", -1);
    int64_t n_ff = gguf_get_val_int(ctx, "llama.feed_forward_length", -1);
    int64_t n_experts = gguf_get_val_int(ctx, "llama.expert_count", 0);
    float rope_theta = gguf_get_val_f32(ctx, "llama.rope.freq_base", 0.0f);
    float norm_eps = gguf_get_val_f32(ctx,
        "llama.attention.layer_norm_rms_epsilon", 0.0f);

    printf("\n── Model Hyperparameters ──\n");
    if (vocab > 0)     printf("vocab_size:           %" PRId64 "\n", vocab);
    if (ctx_len > 0)   printf("context_length:       %" PRId64 "\n", ctx_len);
    if (n_embd > 0)    printf("embedding_length:     %" PRId64 "\n", n_embd);
    if (n_layer > 0)   printf("block_count:          %" PRId64 "\n", n_layer);
    if (n_head > 0)    printf("attn_head_count:      %" PRId64 "\n", n_head);
    if (n_head_kv > 0) printf("attn_head_count_kv:   %" PRId64 " (GQA)\n", n_head_kv);
    if (n_ff > 0)      printf("feed_forward_length:  %" PRId64 "\n", n_ff);
    if (n_experts > 0) printf("expert_count:         %" PRId64 " (MoE)\n", n_experts);
    if (rope_theta > 0) printf("rope_theta:           %.2f\n", rope_theta);
    if (norm_eps > 0)  printf("rms_norm_epsilon:     %.6e\n", norm_eps);

    /* Tokenizer info */
    const char *tok_model = gguf_get_val_str(ctx, "tokenizer.ggml.model");
    if (tok_model) {
        printf("\n── Tokenizer ──\n");
        printf("model:               %s\n", tok_model);
        int64_t bos = gguf_get_val_int(ctx, "tokenizer.ggml.bos_token_id", -1);
        int64_t eos = gguf_get_val_int(ctx, "tokenizer.ggml.eos_token_id", -1);
        if (bos >= 0) printf("bos_token_id:        %" PRId64 "\n", bos);
        if (eos >= 0) printf("eos_token_id:        %" PRId64 "\n", eos);
    }

    /* Tensor summary (first 10) */
    printf("\n── Tensors (first 10 of %" PRId64 ") ──\n", ctx->n_tensors);
    int64_t show = ctx->n_tensors > 10 ? 10 : ctx->n_tensors;
    for (int64_t i = 0; i < show; i++) {
        const GgufTensorInfo *ti = &ctx->tensors[i];
        printf("  [%3" PRId64 "] %-50s  shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
               "]  type=%s(%d)  offset=%" PRIu64 "  size=%zu\n",
               i, ti->name,
               ti->ne[0], ti->ne[1], ti->ne[2], ti->ne[3],
               ggml_type_name(ti->ggml_type), ti->ggml_type,
               ti->offset, ti->nbytes);
    }
    if (ctx->n_tensors > 10) {
        printf("  ... and %" PRId64 " more tensors\n", ctx->n_tensors - 10);
    }

    /* Data quality check */
    {
        int64_t f32_count = 0, f16_count = 0, quant_count = 0;
        for (int64_t i = 0; i < ctx->n_tensors; i++) {
            switch (ctx->tensors[i].ggml_type) {
            case GGML_TYPE_F32: f32_count++; break;
            case GGML_TYPE_F16: f16_count++; break;
            default: quant_count++; break;
            }
        }
        printf("\n── Data Types ──\n");
        printf("F32 tensors:   %" PRId64 "\n", f32_count);
        printf("F16 tensors:   %" PRId64 "\n", f16_count);
        printf("Quant tensors: %" PRId64 "\n", quant_count);
    }

    printf("══════════════════════════\n");
}

/* ── Array data readers (re-open file, seek to value_offset) ──────────── */

char **gguf_read_arr_str(const char *path, const char *key, uint64_t *count_out) {
    if (!path || !key || !count_out) return NULL;
    *count_out = 0;

    /* Parse header to get the KV context */
    GgufContext *ctx = gguf_init_from_file(path);
    if (!ctx) return NULL;

    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0 || !ctx->kv_pairs[idx].is_array
        || ctx->kv_pairs[idx].array_type != GGUF_TYPE_STRING) {
        gguf_free(ctx);
        return NULL;
    }

    uint64_t count = ctx->kv_pairs[idx].array_count;
    long    offset = ctx->kv_pairs[idx].value_offset;
    gguf_free(ctx);

    if (count == 0) return NULL;

    /* Re-open file and seek to value data */
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, offset, SEEK_SET) != 0) { fclose(f); return NULL; }

    char **out = (char **)calloc((size_t)count, sizeof(char *));
    if (!out) { fclose(f); return NULL; }

    for (uint64_t i = 0; i < count; i++) {
        uint64_t slen = read_u64_le(f);
        if (slen > GGUF_MAX_STRING_LENGTH) {
            for (uint64_t j = 0; j < i; j++) free(out[j]);
            free(out);
            fclose(f);
            return NULL;
        }
        out[i] = (char *)calloc(slen + 1, 1);
        if (!out[i] || (slen > 0 && fread(out[i], 1, slen, f) != slen)) {
            if (out[i]) free(out[i]);
            for (uint64_t j = 0; j < i; j++) free(out[j]);
            free(out);
            fclose(f);
            return NULL;
        }
        out[i][slen] = '\0';
    }

    fclose(f);
    *count_out = count;
    return out;
}

int32_t *gguf_read_arr_int(const char *path, const char *key, uint64_t *count_out) {
    if (!path || !key || !count_out) return NULL;
    *count_out = 0;

    GgufContext *ctx = gguf_init_from_file(path);
    if (!ctx) return NULL;

    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0 || !ctx->kv_pairs[idx].is_array
        || ctx->kv_pairs[idx].array_type != GGUF_TYPE_INT32) {
        gguf_free(ctx);
        return NULL;
    }

    uint64_t count = ctx->kv_pairs[idx].array_count;
    long    offset = ctx->kv_pairs[idx].value_offset;
    gguf_free(ctx);

    if (count == 0) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, offset, SEEK_SET) != 0) { fclose(f); return NULL; }

    int32_t *out = (int32_t *)calloc((size_t)count, sizeof(int32_t));
    if (!out) { fclose(f); return NULL; }

    for (uint64_t i = 0; i < count; i++) {
        out[i] = read_i32_le(f);
    }

    fclose(f);
    *count_out = count;
    return out;
}

float *gguf_read_arr_f32(const char *path, const char *key, uint64_t *count_out) {
    if (!path || !key || !count_out) return NULL;
    *count_out = 0;

    GgufContext *ctx = gguf_init_from_file(path);
    if (!ctx) return NULL;

    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0 || !ctx->kv_pairs[idx].is_array
        || ctx->kv_pairs[idx].array_type != GGUF_TYPE_FLOAT32) {
        gguf_free(ctx);
        return NULL;
    }

    uint64_t count = ctx->kv_pairs[idx].array_count;
    long    offset = ctx->kv_pairs[idx].value_offset;
    gguf_free(ctx);

    if (count == 0) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, offset, SEEK_SET) != 0) { fclose(f); return NULL; }

    float *out = (float *)calloc((size_t)count, sizeof(float));
    if (!out) { fclose(f); return NULL; }

    for (uint64_t i = 0; i < count; i++) {
        out[i] = read_f32_le(f);
    }

    fclose(f);
    *count_out = count;
    return out;
}

/* ─────────────────────────────────────────────────────────────────────────
 * GPT-2 Weight Loader (backward-compatible with existing API)
 *
 * Loads GPT-2 formatted weights from GGUF or legacy flat binary format.
 * ─────────────────────────────────────────────────────────────────────── */

/* ── Legacy flat binary format ──────────────────────────────────────────── */

#define PARSE_INT(json, key, field) do { \
    char *p = strstr((json), "\"" key "\":"); \
    if (p) { \
        (field) = (int32_t)strtol(p + strlen("\"" key "\":") + 1, NULL, 10); \
    } \
} while(0)

typedef struct {
    GgufContext *gguf_ctx;
    FILE        *file;
    bool         is_gguf;
} ModelFile;

static ModelFile *model_file_open(const char *path) {
    ModelFile *mf = (ModelFile *)calloc(1, sizeof(ModelFile));
    if (!mf) return NULL;

    mf->file = fopen(path, "rb");
    if (!mf->file) {
        fprintf(stderr, "[xLLM] Cannot open model file: %s\n", path);
        free(mf);
        return NULL;
    }

    /* Check if GGUF */
    uint32_t magic;
    if (fread(&magic, 4, 1, mf->file) != 1) {
        fclose(mf->file);
        free(mf);
        return NULL;
    }
    rewind(mf->file);

    if (magic == GGUF_MAGIC_U32) {
        mf->gguf_ctx = gguf_init_from_file(path);
        if (!mf->gguf_ctx) {
            fclose(mf->file);
            free(mf);
            return NULL;
        }
        mf->is_gguf = true;
    }

    return mf;
}

static void model_file_close(ModelFile *mf) {
    if (!mf) return;
    if (mf->file) fclose(mf->file);
    gguf_free(mf->gguf_ctx);
    free(mf);
}

/* ── Q4_K / Q6_K dequantization (CPU reference, based on llama.cpp) ──────── */

static void dequantize_q4_k(const uint8_t *x, float *y, int64_t n) {
    const int nb = 256;
    for (int64_t ib = 0; ib < n; ib += nb) {
        const uint8_t *blk = x + (ib / nb) * 144;

        uint16_t dh_raw = blk[0] | ((uint16_t)blk[1] << 8);
        uint16_t dm_raw = blk[2] | ((uint16_t)blk[3] << 8);
        float d, dmin;
        {
            uint32_t sign = (dh_raw >> 15) & 1, exp = (dh_raw >> 10) & 0x1F, mant = dh_raw & 0x3FF;
            if (exp == 0) d = (sign ? -1.0f : 1.0f) * (float)mant * 5.96046448e-8f;
            else if (exp == 0x1F) d = (mant == 0) ? INFINITY : NAN;
            else { uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13); memcpy(&d, &f32, sizeof(float)); }
        }
        {
            uint32_t sign = (dm_raw >> 15) & 1, exp = (dm_raw >> 10) & 0x1F, mant = dm_raw & 0x3FF;
            if (exp == 0) dmin = (sign ? -1.0f : 1.0f) * (float)mant * 5.96046448e-8f;
            else if (exp == 0x1F) dmin = (mant == 0) ? INFINITY : NAN;
            else { uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13); memcpy(&dmin, &f32, sizeof(float)); }
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
     * Output: 16 sub-blocks of 16 elements each, scale per sub-block.
     * See text_gen.c for the verified implementation. */
    const int nb = 256;
    for (int64_t ib = 0; ib < n; ib += nb) {
        const uint8_t *blk = x + (ib / nb) * 210;

        const uint8_t *ql     = blk;
        const uint8_t *qh     = blk + 128;
        const int8_t  *scales = (const int8_t *)(blk + 192);

        uint16_t dh_raw = blk[208] | ((uint16_t)blk[209] << 8);
        float d;
        {
            uint32_t sign = (dh_raw >> 15) & 1, exp = (dh_raw >> 10) & 0x1F, mant = dh_raw & 0x3FF;
            if (exp == 0) d = (sign ? -1.0f : 1.0f) * (float)mant * 5.96046448e-8f;
            else if (exp == 0x1F) d = (mant == 0) ? INFINITY : NAN;
            else { uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13); memcpy(&d, &f32, sizeof(float)); }
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
                    y[base + lo + li]       = d * (float)scales[half*8 + 0 + is] * (float)(q1 - 32);
                    y[base + 32 + lo + li]  = d * (float)scales[half*8 + 2 + is] * (float)(q2 - 32);
                    y[base + 64 + lo + li]  = d * (float)scales[half*8 + 4 + is] * (float)(q3 - 32);
                    y[base + 96 + lo + li]  = d * (float)scales[half*8 + 6 + is] * (float)(q4 - 32);
                }
            }
        }
    }
}

/* ── GGUF weight loading into Gpt2Weights ───────────────────────────────── */

static bool load_gguf_weight_into(ModelFile *mf, const char *tensor_name,
                                  float *dst, size_t expected_bytes) {
    const GgufTensorInfo *ti = gguf_get_tensor_by_name(mf->gguf_ctx, tensor_name);
    if (!ti) {
        fprintf(stderr, "[xLLM] Tensor not found in GGUF: '%s'\n", tensor_name);
        return false;
    }
    /* For quantized types, raw byte count differs from F32 expected count.
     * Skip exact size check — the caller's buffer is sized for F32 output. */
    if (ti->ggml_type != GGML_TYPE_Q4_K && ti->ggml_type != GGML_TYPE_Q6_K) {
        if (ti->nbytes != expected_bytes) {
            fprintf(stderr, "[xLLM] Tensor '%s' size mismatch: expected %zu, got %zu\n",
                    tensor_name, expected_bytes, ti->nbytes);
            return false;
        }
    }

    FILE *f = mf->file;
    uint64_t abs_offset = mf->gguf_ctx->data_offset + ti->offset;
#ifdef _WIN32
    if (_fseeki64(f, (int64_t)abs_offset, SEEK_SET) != 0) return false;
#else
    if (fseeko(f, (off_t)abs_offset, SEEK_SET) != 0) return false;
#endif

    /* For F32 tensors, read directly */
    if (ti->ggml_type == GGML_TYPE_F32) {
        if (fread(dst, 1, ti->nbytes, f) != ti->nbytes) return false;
        return true;
    }

    /* For F16 tensors, convert to F32 */
    if (ti->ggml_type == GGML_TYPE_F16) {
        size_t count = ti->nbytes / 2;
        for (size_t j = 0; j < count; j++) {
            uint16_t f16;
            if (fread(&f16, 2, 1, f) != 1) return false;
            /* Convert F16 to F32 */
            uint32_t sign     = (f16 >> 15) & 1;
            uint32_t exponent = (f16 >> 10) & 0x1F;
            uint32_t mantissa = f16 & 0x3FF;
            uint32_t f32;
            if (exponent == 0) {
                /* Subnormal or zero */
                f32 = (sign << 31) | ((mantissa) << (23 - 10));
            } else if (exponent == 0x1F) {
                /* Infinity or NaN */
                f32 = (sign << 31) | (0xFF << 23) | (mantissa << (23 - 10));
            } else {
                /* Normal */
                f32 = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << (23 - 10));
            }
            memcpy(&dst[j], &f32, sizeof(float));
        }
        return true;
    }

    /* For quantized types, read raw bytes then dequantize into dst */
    if (ti->ggml_type == GGML_TYPE_Q4_K || ti->ggml_type == GGML_TYPE_Q6_K) {
        uint8_t *raw = (uint8_t *)malloc(ti->nbytes);
        if (!raw) return false;
        if (fread(raw, 1, ti->nbytes, f) != ti->nbytes) { free(raw); return false; }

        int64_t nelem = ti->ne[0] * ti->ne[1] * ti->ne[2] * ti->ne[3];
        if (nelem <= 0) nelem = 1;

        if (ti->ggml_type == GGML_TYPE_Q4_K)
            dequantize_q4_k(raw, dst, nelem);
        else
            dequantize_q6_k(raw, dst, nelem);

        free(raw);
        return true;
    }
    /* Unknown quantized type — read raw bytes with warning */
    {
        if (fread(dst, 1, ti->nbytes, f) != ti->nbytes) return false;
        fprintf(stderr, "[xLLM] Warning: '%s' is quantized (%s), loaded as raw bytes\n",
                tensor_name, ggml_type_name(ti->ggml_type));
        return true;
    }
}

static bool load_gguf_weights_to_gpt2(ModelFile *mf, Gpt2Weights *w,
                                       const Gpt2Config *cfg) {
    /* Token embeddings */
    {
        const GgufTensorInfo *ti = gguf_get_tensor_by_name(mf->gguf_ctx, "token_embd.weight");
        if (ti) {
            size_t expected = (size_t)cfg->vocab_size * (size_t)cfg->n_embd * sizeof(float);
            load_gguf_weight_into(mf, "token_embd.weight", w->wte, expected);
        }
    }

    /* Output norm */
    {
        const GgufTensorInfo *ti = gguf_get_tensor_by_name(mf->gguf_ctx, "output_norm.weight");
        if (ti) {
            size_t expected = (size_t)cfg->n_embd * sizeof(float);
            load_gguf_weight_into(mf, "output_norm.weight", w->ln_f_weight, expected);
        }
    }

    /* Per-layer weights */
    char name_buf[256];
    for (int32_t layer = 0; layer < cfg->n_layer; layer++) {
        /* Input norm (attention) */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", layer);
        {
            const GgufTensorInfo *ti = gguf_get_tensor_by_name(mf->gguf_ctx, name_buf);
            if (ti) {
                size_t expected = (size_t)cfg->n_embd * sizeof(float);
                load_gguf_weight_into(mf, name_buf, w->ln_1_weight[layer], expected);
            }
        }

        /* QKV weight — try fused first, then separated (Llama 3 GQA) */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_qkv.weight", layer);
        {
            const GgufTensorInfo *ti = gguf_get_tensor_by_name(mf->gguf_ctx, name_buf);
            if (ti) {
                size_t expected = (size_t)cfg->n_embd * 3 * (size_t)cfg->n_embd * sizeof(float);
                load_gguf_weight_into(mf, name_buf, w->attn_c_attn_w[layer], expected);
            } else {
                /* Fallback: separated Q, K, V tensors */
                snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q.weight", layer);
                const GgufTensorInfo *ti_q = gguf_get_tensor_by_name(mf->gguf_ctx, name_buf);
                if (ti_q) {
                    int64_t n_kv_head = gguf_get_val_int(mf->gguf_ctx,
                        "llama.attention.head_count_kv", cfg->n_head);
                    int32_t kv_dim = (int32_t)n_kv_head * (cfg->n_embd / cfg->n_head);

                    size_t q_expected = (size_t)cfg->n_embd * (size_t)cfg->n_embd * sizeof(float);
                    size_t k_expected = (size_t)cfg->n_embd * (size_t)kv_dim * sizeof(float);
                    size_t v_expected = (size_t)cfg->n_embd * (size_t)kv_dim * sizeof(float);

                    float *q_buf = w->attn_c_attn_w[layer];
                    float *k_buf = q_buf + (size_t)cfg->n_embd * (size_t)cfg->n_embd;
                    float *v_buf = k_buf + (size_t)cfg->n_embd * (size_t)kv_dim;

                    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q.weight", layer);
                    bool ok = load_gguf_weight_into(mf, name_buf, q_buf, q_expected);
                    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k.weight", layer);
                    ok = load_gguf_weight_into(mf, name_buf, k_buf, k_expected) && ok;
                    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_v.weight", layer);
                    ok = load_gguf_weight_into(mf, name_buf, v_buf, v_expected) && ok;

                    if (!ok) {
                        fprintf(stderr, "[xLLM] Failed to load separated Q/K/V for layer %d\n",
                                layer);
                    }
                }
            }
        }

        /* Output projection */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_output.weight", layer);
        {
            const GgufTensorInfo *ti = gguf_get_tensor_by_name(mf->gguf_ctx, name_buf);
            if (ti) {
                size_t expected = (size_t)cfg->n_embd * (size_t)cfg->n_embd * sizeof(float);
                load_gguf_weight_into(mf, name_buf, w->attn_c_proj_w[layer], expected);
            }
        }

        /* FFN norm */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", layer);
        {
            const GgufTensorInfo *ti = gguf_get_tensor_by_name(mf->gguf_ctx, name_buf);
            if (ti) {
                size_t expected = (size_t)cfg->n_embd * sizeof(float);
                load_gguf_weight_into(mf, name_buf, w->ln_2_weight[layer], expected);
            }
        }

        /* FFN gate (SwiGLU) */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate.weight", layer);
        {
            const GgufTensorInfo *ti = gguf_get_tensor_by_name(mf->gguf_ctx, name_buf);
            if (ti) {
                size_t expected = (size_t)cfg->n_embd * (size_t)cfg->n_inner * sizeof(float);
                load_gguf_weight_into(mf, name_buf, w->mlp_c_fc_w[layer], expected);
            }
        }

        /* FFN down projection */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down.weight", layer);
        {
            const GgufTensorInfo *ti = gguf_get_tensor_by_name(mf->gguf_ctx, name_buf);
            if (ti) {
                size_t expected = (size_t)cfg->n_inner * (size_t)cfg->n_embd * sizeof(float);
                load_gguf_weight_into(mf, name_buf, w->mlp_c_proj_w[layer], expected);
            }
        }
    }

    return true;
}

/* ── GPT-2 Weight Loader: Public API ───────────────────────────────────── */

Gpt2Weights *gpt2_weights_load(const char *path, Gpt2Config *cfg_out) {
    if (!path || !cfg_out) return NULL;

    ModelFile *mf = model_file_open(path);
    if (!mf) return NULL;

    if (mf->is_gguf) {
        /* GGUF path */
        if (!gguf_load_model_config(mf->gguf_ctx, cfg_out)) {
            model_file_close(mf);
            return NULL;
        }

        /* Log architecture info */
        fprintf(stderr, "[xLLM] Detected architecture: %s\n",
                model_arch_name(gguf_detect_architecture(mf->gguf_ctx)));
        fprintf(stderr, "[xLLM] Config: vocab=%d, ctx=%d, embd=%d, "
                "layers=%d, heads=%d, ff=%d\n",
                cfg_out->vocab_size, cfg_out->n_positions, cfg_out->n_embd,
                cfg_out->n_layer, cfg_out->n_head, cfg_out->n_inner);

        Gpt2Weights *w = gpt2_weights_alloc(cfg_out);
        if (!w) {
            model_file_close(mf);
            return NULL;
        }

        if (!load_gguf_weights_to_gpt2(mf, w, cfg_out)) {
            gpt2_weights_free(w);
            model_file_close(mf);
            return NULL;
        }

        model_file_close(mf);
        return w;
    }

    /* ── Legacy flat binary format ─────────────────────────────────── */

    /* Read config JSON length */
    uint32_t json_len;
    if (fread(&json_len, 4, 1, mf->file) != 1) {
        model_file_close(mf);
        return NULL;
    }

    char *json = (char *)calloc(json_len + 1, 1);
    if (!json || fread(json, 1, json_len, mf->file) != json_len) {
        free(json);
        model_file_close(mf);
        return NULL;
    }

    memset(cfg_out, 0, sizeof(*cfg_out));
    PARSE_INT(json, "vocab_size",  cfg_out->vocab_size);
    PARSE_INT(json, "n_positions", cfg_out->n_positions);
    PARSE_INT(json, "n_embd",      cfg_out->n_embd);
    PARSE_INT(json, "n_layer",     cfg_out->n_layer);
    PARSE_INT(json, "n_head",      cfg_out->n_head);
    PARSE_INT(json, "n_inner",     cfg_out->n_inner);

    cfg_out->head_size      = cfg_out->n_embd / cfg_out->n_head;
    cfg_out->layer_norm_eps = 1e-5f;

    free(json);

    /* Read tensor count */
    uint32_t tensor_count;
    if (fread(&tensor_count, 4, 1, mf->file) != 1) {
        model_file_close(mf);
        return NULL;
    }

    Gpt2Weights *w = gpt2_weights_alloc(cfg_out);
    if (!w) {
        model_file_close(mf);
        return NULL;
    }

    /* Load tensors from legacy format */
    for (uint32_t i = 0; i < tensor_count; i++) {
        uint32_t name_len;
        if (fread(&name_len, 4, 1, mf->file) != 1) break;

        char *tname = (char *)calloc(name_len + 1, 1);
        if (!tname || fread(tname, 1, name_len, mf->file) != name_len) {
            free(tname);
            break;
        }

        uint32_t data_len;
        if (fread(&data_len, 4, 1, mf->file) != 1) {
            free(tname);
            break;
        }

        /* Map and copy tensor data */
        if (strcmp(tname, "wte.weight") == 0
            || strcmp(tname, "transformer.wte.weight") == 0) {
            if (w->wte && data_len >=
                (uint32_t)(cfg_out->vocab_size * cfg_out->n_embd * sizeof(float))) {
                size_t sz = (size_t)cfg_out->vocab_size * (size_t)cfg_out->n_embd
                            * sizeof(float);
                fread(w->wte, 1, sz, mf->file);
            } else {
                fseek(mf->file, data_len, SEEK_CUR);
            }
        } else if (strcmp(tname, "wpe.weight") == 0
                   || strcmp(tname, "transformer.wpe.weight") == 0) {
            if (w->wpe && data_len >=
                (uint32_t)(cfg_out->n_positions * cfg_out->n_embd * sizeof(float))) {
                size_t sz = (size_t)cfg_out->n_positions * (size_t)cfg_out->n_embd
                            * sizeof(float);
                fread(w->wpe, 1, sz, mf->file);
            } else {
                fseek(mf->file, data_len, SEEK_CUR);
            }
        } else {
            fseek(mf->file, data_len, SEEK_CUR);
        }

        free(tname);
    }

    model_file_close(mf);
    return w;
}

/* ── Weight allocation ─────────────────────────────────────────────────── */

Gpt2Weights *gpt2_weights_alloc(const Gpt2Config *cfg) {
    if (!cfg || cfg->n_layer <= 0 || cfg->n_layer > 128) return NULL;

    Gpt2Weights *w = (Gpt2Weights *)calloc(1, sizeof(Gpt2Weights));
    if (!w) return NULL;

    w->n_layer = cfg->n_layer;

    /* Token + position embeddings */
    w->wte = (float *)calloc((size_t)cfg->vocab_size * (size_t)cfg->n_embd,
                             sizeof(float));
    w->wpe = (float *)calloc((size_t)cfg->n_positions * (size_t)cfg->n_embd,
                             sizeof(float));

    /* Final layer norm */
    w->ln_f_weight = (float *)calloc((size_t)cfg->n_embd, sizeof(float));
    w->ln_f_bias   = (float *)calloc((size_t)cfg->n_embd, sizeof(float));

    /* Per-layer arrays */
#define ALLOC_LAYER(name, elems) do { \
    w->name = (float **)calloc((size_t)(cfg->n_layer), sizeof(float *)); \
    for (int32_t i = 0; i < cfg->n_layer; i++) \
        w->name[i] = (float *)calloc((size_t)(elems), sizeof(float)); \
} while(0)

    ALLOC_LAYER(ln_1_weight,    cfg->n_embd);
    ALLOC_LAYER(ln_1_bias,      cfg->n_embd);
    ALLOC_LAYER(attn_c_attn_w,  cfg->n_embd * 3 * cfg->n_embd);
    ALLOC_LAYER(attn_c_attn_b,  3 * cfg->n_embd);
    ALLOC_LAYER(attn_c_proj_w,  cfg->n_embd * cfg->n_embd);
    ALLOC_LAYER(attn_c_proj_b,  cfg->n_embd);
    ALLOC_LAYER(ln_2_weight,    cfg->n_embd);
    ALLOC_LAYER(ln_2_bias,      cfg->n_embd);
    ALLOC_LAYER(mlp_c_fc_w,     cfg->n_embd * cfg->n_inner);
    ALLOC_LAYER(mlp_c_fc_b,     cfg->n_inner);
    ALLOC_LAYER(mlp_c_proj_w,   cfg->n_inner * cfg->n_embd);
    ALLOC_LAYER(mlp_c_proj_b,   cfg->n_embd);

#undef ALLOC_LAYER

    return w;
}

/* ── Weight deallocation ───────────────────────────────────────────────── */

void gpt2_weights_free(Gpt2Weights *w) {
    if (!w) return;
    free(w->wte);
    free(w->wpe);
    free(w->ln_f_weight);
    free(w->ln_f_bias);
    int32_t n_layer = w->n_layer;
    for (int32_t i = 0; i < n_layer; i++) {
        free(w->ln_1_weight    ? w->ln_1_weight[i]    : NULL);
        free(w->ln_1_bias      ? w->ln_1_bias[i]      : NULL);
        free(w->attn_c_attn_w  ? w->attn_c_attn_w[i]  : NULL);
        free(w->attn_c_attn_b  ? w->attn_c_attn_b[i]  : NULL);
        free(w->attn_c_proj_w  ? w->attn_c_proj_w[i]  : NULL);
        free(w->attn_c_proj_b  ? w->attn_c_proj_b[i]  : NULL);
        free(w->ln_2_weight    ? w->ln_2_weight[i]    : NULL);
        free(w->ln_2_bias      ? w->ln_2_bias[i]      : NULL);
        free(w->mlp_c_fc_w     ? w->mlp_c_fc_w[i]     : NULL);
        free(w->mlp_c_fc_b     ? w->mlp_c_fc_b[i]     : NULL);
        free(w->mlp_c_proj_w   ? w->mlp_c_proj_w[i]   : NULL);
        free(w->mlp_c_proj_b   ? w->mlp_c_proj_b[i]   : NULL);
    }
    free(w->ln_1_weight);
    free(w->ln_1_bias);
    free(w->attn_c_attn_w);
    free(w->attn_c_attn_b);
    free(w->attn_c_proj_w);
    free(w->attn_c_proj_b);
    free(w->ln_2_weight);
    free(w->ln_2_bias);
    free(w->mlp_c_fc_w);
    free(w->mlp_c_fc_b);
    free(w->mlp_c_proj_w);
    free(w->mlp_c_proj_b);
    free(w);
}

/* ── Legacy public API (used by existing tests) ─────────────────────────── */

const char *nxt_model_loader_detect_arch(const char *path) {
    if (!path) return NULL;
    const char *arch = gguf_detect_arch_from_file(path);
    if (arch && strcmp(arch, "unknown") != 0) return arch;
    /* Try legacy format */
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return NULL; }
    fclose(f);
    if (magic == GGUF_MAGIC_U32) return "unknown";
    return "gpt2";
}

bool nxt_model_loader_load_config(const char *path, Gpt2Config *cfg_out) {
    if (!path || !cfg_out) return false;

    FILE *f = fopen(path, "rb");
    if (!f) return false;

    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return false; }
    rewind(f);

    if (magic == GGUF_MAGIC_U32) {
        GgufContext *ctx = gguf_init_from_file(path);
        fclose(f);
        if (!ctx) return false;
        bool ok = gguf_load_model_config(ctx, cfg_out);
        gguf_free(ctx);
        return ok;
    }

    fclose(f);
    return false;
}
