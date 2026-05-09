/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * GGUF Model Loader — GGUF format constants, types, and parser API.
 * Implements full GGUF v3 binary format parsing in pure C11.
 *
 * Design based on llama.cpp's gguf.h / gguf.cpp with two-pass loading:
 *   Pass 1: parse header + KV pairs + tensor metadata (no weight data)
 *   Pass 2: load tensor data via mmap or direct read
 */

#ifndef XLLM_MODEL_LOADER_H_
#define XLLM_MODEL_LOADER_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── GGUF magic and version constants ───────────────────────────────────── */

#define GGUF_MAGIC_BYTES    "GGUF"
#define GGUF_MAGIC_U32      0x46554747u   /* "GGUF" in little-endian */
#define GGUF_VERSION        3
#define GGUF_DEFAULT_ALIGNMENT  32
#define GGUF_MAX_KV_PAIRS       10000
#define GGUF_MAX_TENSORS        100000
#define GGUF_MAX_STRING_LENGTH  (64 * 1024 * 1024)  /* 64 MiB safety limit */
#define GGUF_MAX_KEY_LENGTH     256
#define GGUF_MAX_DIMS            4

#define GGUF_KEY_GENERAL_ALIGNMENT "general.alignment"
#define GGUF_KEY_ARCHITECTURE      "general.architecture"
#define GGUF_KEY_MODEL_NAME        "general.name"

/* ── GGUF value types ──────────────────────────────────────────────────── */

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT
} GgufType;

/* ── GGML tensor type (subset used by xLLM) ─────────────────────────────── */

typedef enum {
    GGML_TYPE_F32    = 0,
    GGML_TYPE_F16    = 1,
    GGML_TYPE_Q4_0   = 2,
    GGML_TYPE_Q4_1   = 3,
    GGML_TYPE_Q5_0   = 6,
    GGML_TYPE_Q5_1   = 7,
    GGML_TYPE_Q8_0   = 8,
    GGML_TYPE_Q8_1   = 9,
    GGML_TYPE_Q2_K   = 10,
    GGML_TYPE_Q3_K   = 11,
    GGML_TYPE_Q4_K   = 12,
    GGML_TYPE_Q5_K   = 13,
    GGML_TYPE_Q6_K   = 14,
    GGML_TYPE_Q8_K   = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_BF16   = 30,
    GGML_TYPE_COUNT
} GgmlType;

/* ── GGUF KV pair ──────────────────────────────────────────────────────── */

typedef struct GgufKV {
    char     *key;           /* null-terminated key string */
    GgufType  type;          /* value type (for scalars) or array element type (for arrays) */
    bool      is_array;      /* true if this KV is an array */
    GgufType  array_type;    /* element type when is_array=true */
    uint64_t  array_count;   /* number of elements when is_array=true */
    long      value_offset;  /* file offset where value data begins */

    /* Value storage — union of all GGUF scalar types + string pointer */
    union {
        uint8_t   uint8_val;
        int8_t    int8_val;
        uint16_t  uint16_val;
        int16_t   int16_val;
        uint32_t  uint32_val;
        int32_t   int32_val;
        float     float32_val;
        bool      bool_val;
        char     *string_val;
        uint64_t  uint64_val;
        int64_t   int64_val;
        double    float64_val;
    };
} GgufKV;

/* ── GGUF tensor info ─────────────────────────────────────────────────── */

typedef struct GgufTensorInfo {
    char      name[256];     /* tensor name (e.g. "blk.0.attn_q.weight") */
    uint32_t  n_dims;        /* number of dimensions (1-4) */
    int64_t   ne[GGUF_MAX_DIMS];  /* dimension sizes */
    GgmlType  ggml_type;     /* element type */
    uint64_t  offset;        /* byte offset from start of data section */
    size_t    nbytes;        /* total bytes for this tensor (computed) */
    /* Strides (nb[0..3]) computed during parsing:
     *   nb[0] = type_size
     *   nb[1] = nb[0] * (ne[0] / block_size)   -- block_size=1 for F32/F16
     *   nb[2] = nb[1] * ne[1]
     *   nb[3] = nb[2] * ne[2]
     */
    size_t    nb[GGUF_MAX_DIMS];
} GgufTensorInfo;

/* ── GGUF context (opaque, defined in .c) ──────────────────────────────── */

typedef struct GgufContext GgufContext;

/* ── Model architecture enum ───────────────────────────────────────────── */

typedef enum {
    ARCH_UNKNOWN  = 0,
    ARCH_LLAMA    = 1,
    ARCH_MISTRAL  = 2,
    ARCH_QWEN2    = 3,
    ARCH_DEEPSEEK = 4,
    ARCH_FALCON   = 5,
    ARCH_GEMMA    = 6,
    ARCH_PHI      = 7,
} ModelArch;

/** Return human-readable name for a ModelArch value. */
static inline const char *model_arch_name(ModelArch arch) {
    switch (arch) {
    case ARCH_LLAMA:    return "llama";
    case ARCH_MISTRAL:  return "mistral";
    case ARCH_QWEN2:    return "qwen2";
    case ARCH_DEEPSEEK: return "deepseek";
    case ARCH_FALCON:   return "falcon";
    case ARCH_GEMMA:    return "gemma";
    case ARCH_PHI:      return "phi";
    default:            return "unknown";
    }
}

/* ── GGUF type utilities ───────────────────────────────────────────────── */

/** Return the size in bytes for a given GGUF type. Returns 0 for STRING/ARRAY. */
size_t gguf_type_size(GgufType type);

/** Return human-readable name for a GGUF type (e.g. "f32", "str"). */
const char *gguf_type_name(GgufType type);

/** Return human-readable name for a GGML type (e.g. "f32", "f16", "q4_0"). */
const char *ggml_type_name(GgmlType type);

/** Return element size in bytes for a GGML tensor type. */
size_t ggml_type_size(GgmlType type);

/** Return block size for quantized types (1 for F32/F16/BF16). */
int64_t ggml_blck_size(GgmlType type);

/* ── GGUF context lifecycle ────────────────────────────────────────────── */

/**
 * Parse a GGUF file header, KV pairs, and tensor metadata.
 * This is Pass 1 — no tensor data is loaded.
 *
 * @param path  File path to a GGUF format model
 * @return      Parsed context, or NULL on error (prints to stderr)
 */
GgufContext *gguf_init_from_file(const char *path);

/**
 * Free a GGUF context and all associated memory.
 * Safe to call with NULL.
 */
void gguf_free(GgufContext *ctx);

/* ── GGUF header accessors ─────────────────────────────────────────────── */

/** GGUF file version (2 or 3). */
uint32_t gguf_get_version(const GgufContext *ctx);

/** Data section alignment in bytes (default 32). */
size_t gguf_get_alignment(const GgufContext *ctx);

/** Byte offset in the file where tensor data begins. */
uint64_t gguf_get_data_offset(const GgufContext *ctx);

/** Total size of the tensor data section in bytes. */
size_t gguf_get_data_size(const GgufContext *ctx);

/* ── KV pair access ────────────────────────────────────────────────────── */

/** Number of key-value pairs in the GGUF metadata. */
int64_t gguf_get_n_kv(const GgufContext *ctx);

/** Find a KV pair by key name. Returns index, or -1 if not found. */
int64_t gguf_find_key(const GgufContext *ctx, const char *key);

/** Get the key string at a given index. */
const char *gguf_get_key(const GgufContext *ctx, int64_t key_id);

/** Get the type of the KV pair at a given index. */
GgufType gguf_get_kv_type(const GgufContext *ctx, int64_t key_id);

/* ── Typed KV value getters (convenience wrappers) ──────────────────────── */

/** Get string value by key. Returns NULL if not found or wrong type. */
const char *gguf_get_val_str(const GgufContext *ctx, const char *key);

/** Get integer value by key (coerces all int types). Returns default_val if not found. */
int64_t gguf_get_val_int(const GgufContext *ctx, const char *key, int64_t default_val);

/** Get float value by key. Returns default_val if not found. */
float gguf_get_val_f32(const GgufContext *ctx, const char *key, float default_val);

/** Get uint32 value by key. Returns default_val if not found. */
uint32_t gguf_get_val_u32(const GgufContext *ctx, const char *key, uint32_t default_val);

/** Get bool value by key. Returns default_val if not found. */
bool gguf_get_val_bool(const GgufContext *ctx, const char *key, bool default_val);

/**
 * Read a string array from a GGUF file (re-opens the file).
 * @param path      Original GGUF file path
 * @param key       KV key to read (e.g. "tokenizer.ggml.tokens")
 * @param count_out Receives the number of strings read
 * @return          Array of strings (caller must free each and the array), or NULL
 */
char **gguf_read_arr_str(const char *path, const char *key, uint64_t *count_out);

/**
 * Read an i32 array from a GGUF file (re-opens the file).
 * @return Array of int32_t values (caller must free), or NULL
 */
int32_t *gguf_read_arr_int(const char *path, const char *key, uint64_t *count_out);

/**
 * Read an f32 array from a GGUF file (re-opens the file).
 * @return Array of float values (caller must free), or NULL
 */
float *gguf_read_arr_f32(const char *path, const char *key, uint64_t *count_out);

/* ── Tensor info access ────────────────────────────────────────────────── */

/** Number of tensors in the GGUF file. */
int64_t gguf_get_n_tensors(const GgufContext *ctx);

/** Find a tensor by name. Returns index, or -1 if not found. */
int64_t gguf_find_tensor(const GgufContext *ctx, const char *name);

/** Get tensor info by index. Returns NULL if index out of range. */
const GgufTensorInfo *gguf_get_tensor_info(const GgufContext *ctx, int64_t tensor_id);

/** Get tensor info by name. Returns NULL if not found. */
const GgufTensorInfo *gguf_get_tensor_by_name(const GgufContext *ctx, const char *name);

/* ── Architecture detection ────────────────────────────────────────────── */

/**
 * Detect model architecture from GGUF metadata.
 * Reads "general.architecture" KV pair and maps to ModelArch enum.
 */
ModelArch gguf_detect_architecture(const GgufContext *ctx);

/**
 * Quick architecture detection from file path (opens file, reads header only).
 * Returns a string literal from MODEL_ARCH_NAMES, or NULL on error.
 */
const char *gguf_detect_arch_from_file(const char *path);

/* ── Model config loading ──────────────────────────────────────────────── */

/* Forward declaration for Gpt2Config (defined in weight_loader.h) */
typedef struct Gpt2Config Gpt2Config;

/**
 * Load model configuration (vocab_size, n_embd, n_layer, etc.) from GGUF metadata.
 * Reads standard llama.* keys.
 *
 * @return true on success, false if required keys are missing or invalid.
 */
bool gguf_load_model_config(const GgufContext *ctx, Gpt2Config *cfg_out);

/* ── Weight loading ────────────────────────────────────────────────────── */

/**
 * Load tensor weight data from a GGUF file.
 *
 * This is Pass 2 — data is read from the file at the tensor's data_offset + offset.
 *
 * @param file  Open file handle positioned past the GGUF header
 * @param ctx   Parsed GGUF context (from gguf_init_from_file)
 * @param ti    Tensor info entry describing the tensor
 * @param dst   Destination buffer (must be at least ti->nbytes)
 * @return      true on success
 */
bool gguf_load_tensor_data(FILE *file, const GgufContext *ctx,
                           const GgufTensorInfo *ti, void *dst);

/* ── Utility: print GGUF summary ───────────────────────────────────────── */

/** Print a human-readable summary of a GGUF file to stdout. */
void gguf_print_summary(const GgufContext *ctx);

/* ── Legacy API (used by existing tests) ────────────────────────────────── */

/** Quick architecture detection from a model file path. */
const char *nxt_model_loader_detect_arch(const char *path);

/** Load model configuration from a model file path. */
bool nxt_model_loader_load_config(const char *path, Gpt2Config *cfg_out);

#ifdef __cplusplus
}
#endif

#endif /* XLLM_MODEL_LOADER_H_ */
