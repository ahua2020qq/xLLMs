/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Triton-style backend manager: load/unload backend shared libraries,
 * manage model lifecycle, and schedule inference requests.
 */

#include "backend.h"
#include "xllm_config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#define dlopen(path, flags)  LoadLibraryA(path)
#define dlsym(handle, name)  ((void*)GetProcAddress((HMODULE)(handle), name))
#define dlclose(handle)      FreeLibrary((HMODULE)(handle))
#define dlerror()            "dynamic loading not supported on this platform"
#endif

/* ── Global state ─────────────────────────────────────────────────────── */

static struct {
    char        *backend_dir;
    NxtBackend  *backends;
    uint32_t     backend_count;
    uint32_t     backend_capacity;
    NxtModel    *models;
    uint32_t     model_count;
    uint32_t     model_capacity;
    NxtSchedulerConfig  sched_config;
    uint64_t     completed_count;
    uint64_t     queued_count;
    double       total_batch_size;
    uint64_t     batch_count;
    bool         initialized;
} g_state;

#define MAX_BACKENDS 64
#define MAX_MODELS   256

/* ── Backend Manager ──────────────────────────────────────────────────── */

int nxt_backend_manager_init(const char *backend_dir) {
    if (g_state.initialized) return 0;
    memset(&g_state, 0, sizeof(g_state));

    g_state.backend_dir = backend_dir ? strdup(backend_dir) : strdup(".");
    g_state.backend_capacity = MAX_BACKENDS;
    g_state.backends = calloc(MAX_BACKENDS, sizeof(NxtBackend));
    if (!g_state.backends) return -1;

    g_state.model_capacity = MAX_MODELS;
    g_state.models = calloc(MAX_MODELS, sizeof(NxtModel));
    if (!g_state.models) {
        free(g_state.backends);
        free(g_state.backend_dir);
        return -1;
    }

    /* Default scheduler config */
    g_state.sched_config.policy                  = NXT_SCHED_DYNAMIC;
    g_state.sched_config.max_preferred_batch_size = 32;
    g_state.sched_config.max_queue_delay_ms       = 100.0;
    g_state.sched_config.preserve_ordering        = true;
    g_state.sched_config.priority_levels          = 1;
    g_state.sched_config.max_queue_size           = 2048;

    /* Initialize the continuous batching scheduler */
    nxt_scheduler_init(&g_state.sched_config);

    g_state.initialized = true;
    return 0;
}

int nxt_backend_manager_fini(void) {
    if (!g_state.initialized) return 0;

    for (uint32_t i = 0; i < g_state.backend_count; i++) {
        NxtBackend *b = &g_state.backends[i];
        if (b->api.backend_fini) b->api.backend_fini(b);
        if (b->dl_handle) dlclose(b->dl_handle);
        free(b->name);
        free(b->so_path);
    }
    free(g_state.backends);

    for (uint32_t i = 0; i < g_state.model_count; i++) {
        NxtModel *m = &g_state.models[i];
        free(m->name);
        free(m->config.name);
        free(m->config.backend_name);
        free(m->config.model_path);
    }
    free(g_state.models);
    free(g_state.backend_dir);

    nxt_scheduler_fini();

    memset(&g_state, 0, sizeof(g_state));
    return 0;
}

/* ── Backend discovery ─────────────────────────────────────────────────── */

static bool nxt_file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static char *nxt_backend_find_so(const char *dir, const char *so_name) {
    if (!dir || !so_name) return NULL;
    size_t len = strlen(dir) + 1 + strlen(so_name) + 1;
    char *path = malloc(len);
    if (!path) return NULL;
    snprintf(path, len, "%s/%s", dir, so_name);
    if (nxt_file_exists(path)) return path;
    free(path);
    return NULL;
}

/*
 * nxt_backend_discover — auto-discover and load available backends.
 *
 * Search order for each built-in backend:
 *   1. XLLM_CUDA_BACKEND_PATH  env var  (exact .so path)
 *   2. XLLM_BACKEND_DIR        env var  (directory containing .so files)
 *   3. XLLM_BACKEND_INSTALL_DIR          (build-time default)
 *   4. ./backends                           (relative to CWD)
 */
int nxt_backend_discover(void) {
    if (!g_state.initialized) return -1;

    int loaded = 0;
    const char *so_name = XLLM_CUDA_BACKEND_SO;
    char *so_path = NULL;

    /* 1. Exact path via environment variable */
    const char *env_exact = getenv("XLLM_CUDA_BACKEND_PATH");
    if (env_exact && nxt_file_exists(env_exact)) {
        so_path = strdup(env_exact);
    }

    /* 2. Directory via environment variable */
    if (!so_path) {
        const char *env_dir = getenv("XLLM_BACKEND_DIR");
        so_path = nxt_backend_find_so(env_dir, so_name);
    }

    /* 3. Build-time install directory */
    if (!so_path) {
        so_path = nxt_backend_find_so(XLLM_BACKEND_INSTALL_DIR, so_name);
    }

    /* 4. ./backends relative to CWD */
    if (!so_path) {
        so_path = nxt_backend_find_so("backends", so_name);
    }

    if (so_path) {
        if (nxt_backend_load(so_path, "cuda", 0) == 0) {
            fprintf(stderr, "[xLLM] auto-loaded CUDA backend: %s\n", so_path);
            loaded++;
        }
        free(so_path);
    } else {
        fprintf(stderr, "[xLLM] CUDA backend (%s) not found — "
                        "set XLLM_CUDA_BACKEND_PATH or XLLM_BACKEND_DIR\n",
                so_name);
    }

    return loaded;
}

/* ── Backend registration ─────────────────────────────────────────────── */

int nxt_backend_register(const char *so_path) {
    if (!g_state.initialized || !so_path) return -1;
    if (g_state.backend_count >= g_state.backend_capacity) return -1;

    void *handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "[xLLM] dlopen(%s): %s\n", so_path, dlerror());
        return -1;
    }

    NxtBackendAPI api;
    memset(&api, 0, sizeof(api));

    api.backend_init   = (NxtBackendInitFn)dlsym(handle, "nxt_backend_init");
    api.backend_run    = (NxtBackendRunFn)dlsym(handle, "nxt_backend_run");
    api.backend_fini   = (NxtBackendFiniFn)dlsym(handle, "nxt_backend_fini");
    api.model_init     = (NxtModelInitFn)dlsym(handle, "nxt_model_init");
    api.model_fini     = (NxtModelFiniFn)dlsym(handle, "nxt_model_fini");
    api.instance_init  = (NxtModelInstanceInitFn)dlsym(handle, "nxt_model_instance_init");
    api.instance_fini  = (NxtModelInstanceFiniFn)dlsym(handle, "nxt_model_instance_fini");
    api.instance_exec  = (NxtModelInstanceExecFn)dlsym(handle, "nxt_model_instance_exec");

    if (!api.backend_init && !api.backend_run && !api.instance_exec) {
        fprintf(stderr, "[xLLM] %s: no backend API symbols found\n", so_path);
        dlclose(handle);
        return -1;
    }

    NxtBackend *b = &g_state.backends[g_state.backend_count];
    memset(b, 0, sizeof(*b));

    /* Derive name from shared object path */
    const char *base = strrchr(so_path, '/');
    base = base ? base + 1 : so_path;
    b->name = strdup(base);
    b->so_path = strdup(so_path);
    b->dl_handle = handle;
    b->api = api;
    b->state = NXT_BACKEND_STATE_UNINITIALIZED;
    b->priority = 0;

    if (b->api.backend_init) {
        int rc = b->api.backend_init(b);
        if (rc != 0) {
            fprintf(stderr, "[xLLM] backend_init(%s) failed: %d\n", base, rc);
            free(b->name);
            free(b->so_path);
            dlclose(handle);
            return -1;
        }
    }
    b->state = NXT_BACKEND_STATE_READY;
    g_state.backend_count++;
    return 0;
}

int nxt_backend_unregister(const char *name) {
    for (uint32_t i = 0; i < g_state.backend_count; i++) {
        if (strcmp(g_state.backends[i].name, name) == 0) {
            NxtBackend *b = &g_state.backends[i];
            if (b->api.backend_fini) b->api.backend_fini(b);
            if (b->dl_handle) dlclose(b->dl_handle);
            free(b->name);
            free(b->so_path);

            /* Compact array */
            if (i < g_state.backend_count - 1)
                memmove(&g_state.backends[i], &g_state.backends[i + 1],
                        (g_state.backend_count - i - 1) * sizeof(NxtBackend));
            g_state.backend_count--;
            return 0;
        }
    }
    return -1;
}

/* ── Backend load / unload / run ────────────────────────────────────────── */

int nxt_backend_load(const char *so_path, const char *name, int32_t priority) {
    if (!g_state.initialized || !so_path || !name) return -1;
    if (g_state.backend_count >= g_state.backend_capacity) return -1;

    /* Check for duplicate name */
    if (nxt_backend_find(name)) {
        fprintf(stderr, "[xLLM] backend '%s' already loaded\n", name);
        return -1;
    }

    void *handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "[xLLM] dlopen(%s): %s\n", so_path, dlerror());
        return -1;
    }

    NxtBackendAPI api;
    memset(&api, 0, sizeof(api));

    api.backend_init = (NxtBackendInitFn)dlsym(handle, "nxt_backend_init");
    api.backend_run  = (NxtBackendRunFn)dlsym(handle, "nxt_backend_run");
    api.backend_fini = (NxtBackendFiniFn)dlsym(handle, "nxt_backend_fini");

    if (!api.backend_init && !api.backend_run) {
        fprintf(stderr, "[xLLM] %s: no nxt_backend_init or nxt_backend_run found\n", so_path);
        dlclose(handle);
        return -1;
    }

    NxtBackend *b = &g_state.backends[g_state.backend_count];
    memset(b, 0, sizeof(*b));
    b->name      = strdup(name);
    b->so_path   = strdup(so_path);
    b->dl_handle = handle;
    b->api       = api;
    b->state     = NXT_BACKEND_STATE_UNINITIALIZED;
    b->priority  = priority;

    if (b->api.backend_init) {
        b->state = NXT_BACKEND_STATE_INITIALIZING;
        int rc = b->api.backend_init(b);
        if (rc != 0) {
            fprintf(stderr, "[xLLM] backend_init(%s) failed: %d\n", name, rc);
            b->state = NXT_BACKEND_STATE_ERROR;
            free(b->name);
            free(b->so_path);
            dlclose(handle);
            return -1;
        }
    }
    b->state = NXT_BACKEND_STATE_READY;
    g_state.backend_count++;
    return 0;
}

int nxt_backend_unload(const char *name) {
    for (uint32_t i = 0; i < g_state.backend_count; i++) {
        if (strcmp(g_state.backends[i].name, name) == 0) {
            NxtBackend *b = &g_state.backends[i];
            b->state = NXT_BACKEND_STATE_UNLOADING;
            if (b->api.backend_fini) b->api.backend_fini(b);
            if (b->dl_handle) dlclose(b->dl_handle);
            free(b->name);
            free(b->so_path);

            if (i < g_state.backend_count - 1)
                memmove(&g_state.backends[i], &g_state.backends[i + 1],
                        (g_state.backend_count - i - 1) * sizeof(NxtBackend));
            g_state.backend_count--;
            return 0;
        }
    }
    return -1;
}

int nxt_backend_run(const char *name, void *input, void *output) {
    NxtBackend *b = nxt_backend_find(name);
    if (!b || !b->api.backend_run) return -1;
    if (b->state != NXT_BACKEND_STATE_READY) return -1;
    return b->api.backend_run(b, input, output);
}

NxtBackend *nxt_backend_find(const char *name) {
    for (uint32_t i = 0; i < g_state.backend_count; i++)
        if (strcmp(g_state.backends[i].name, name) == 0)
            return &g_state.backends[i];
    return NULL;
}

uint32_t nxt_backend_count(void) { return g_state.backend_count; }

NxtBackend *nxt_backend_list(void) { return g_state.backends; }

/* ── Model lifecycle ──────────────────────────────────────────────────── */

int nxt_model_create(const char *name, const NxtModelConfig *config) {
    if (!name || !config || !config->backend_name) return -1;
    if (g_state.model_count >= g_state.model_capacity) return -1;

    NxtBackend *backend = nxt_backend_find(config->backend_name);
    if (!backend) return -1;

    NxtModel *m = &g_state.models[g_state.model_count];
    memset(m, 0, sizeof(*m));
    m->name    = strdup(name);
    m->version = 1;
    m->config  = *config;
    m->config.name         = strdup(config->name ? config->name : name);
    m->config.backend_name = strdup(config->backend_name);
    m->config.model_path   = config->model_path ? strdup(config->model_path) : NULL;
    m->backend = backend;
    m->state   = NXT_BACKEND_STATE_UNINITIALIZED;

    if (backend->api.model_init) {
        int rc = backend->api.model_init(m);
        if (rc != 0) {
            free(m->name);
            free(m->config.name);
            free(m->config.backend_name);
            free(m->config.model_path);
            return -1;
        }
    }
    m->state = NXT_BACKEND_STATE_READY;
    backend->model_count++;
    g_state.model_count++;
    return 0;
}

int nxt_model_destroy(NxtModel *model) {
    if (!model) return -1;
    if (model->backend && model->backend->api.model_fini)
        model->backend->api.model_fini(model);
    free(model->name);
    free(model->config.name);
    free(model->config.backend_name);
    free(model->config.model_path);
    return 0;
}

NxtModel *nxt_model_find(const char *name, int32_t version) {
    for (uint32_t i = 0; i < g_state.model_count; i++) {
        NxtModel *m = &g_state.models[i];
        if (strcmp(m->name, name) != 0) continue;
        if (version == 0) return m;  /* LATEST — first match */
        if (version == -1) return m; /* ALL — first match */
        if (m->version == version) return m;
    }
    return NULL;
}

uint32_t nxt_model_count(void) { return g_state.model_count; }

int nxt_model_set_version_policy(const char *name, const NxtModelVersionConfig *config) {
    if (!name || !config) return -1;
    NxtModel *m = nxt_model_find(name, 0);
    if (!m) return -1;
    m->config.version_config = *config;
    return 0;
}

/* ── Scheduler delegation ───────────────────────────────────────────────
 * Public API implementations are in src/core/scheduler.c
 * Declared here for internal use by backend manager components.
 */

extern int  nxt_scheduler_init(const NxtSchedulerConfig *config);
extern int  nxt_scheduler_fini(void);
extern int  nxt_scheduler_enqueue(NxtInferRequest *request);
extern int  nxt_scheduler_poll(void);
extern uint64_t nxt_scheduler_completed_count(void);
extern uint64_t nxt_scheduler_queued_count(void);
extern double   nxt_scheduler_avg_batch_size(void);

/* ── Health check ─────────────────────────────────────────────────────── */

bool nxt_server_is_live(void)  { return g_state.initialized; }
bool nxt_server_is_ready(void) { return g_state.initialized && g_state.backend_count > 0; }

bool nxt_model_is_ready(const char *name, int32_t version) {
    NxtModel *m = nxt_model_find(name, version);
    return m && m->state == NXT_BACKEND_STATE_READY;
}

int nxt_model_stats(const char *name, int32_t version, NxtModelStats *stats) {
    if (!name || !stats) return -1;
    NxtModel *m = nxt_model_find(name, version);
    if (!m) return -1;

    memset(stats, 0, sizeof(*stats));
    stats->name = m->name;
    stats->version = m->version;
    stats->execution_count = m->instances ? m->instances->exec_count : 0;
    if (m->instances && m->instances->exec_count > 0)
        stats->avg_latency_ms = m->instances->total_exec_ms / (double)m->instances->exec_count;
    return 0;
}

/* ── Response helpers ─────────────────────────────────────────────────── */

NxtInferResponse *nxt_response_alloc(const char *request_id) {
    NxtInferResponse *r = calloc(1, sizeof(NxtInferResponse));
    if (r && request_id) r->request_id = strdup(request_id);
    return r;
}

void nxt_response_free(NxtInferResponse *response) {
    if (!response) return;
    free(response->request_id);
    free(response->error_message);
    if (response->outputs) {
        for (uint32_t i = 0; i < response->output_count; i++)
            free(response->outputs[i].shape);
        free(response->outputs);
    }
    free(response);
}

int nxt_response_set_output(NxtInferResponse *response,
                             const char *name, NxtDataType dtype,
                             const int64_t *shape, uint32_t dims,
                             const void *data, uint64_t byte_size) {
    if (!response || !name) return -1;

    uint32_t idx = response->output_count;
    NxtTensor *new_outputs = realloc(response->outputs,
                                      (idx + 1) * sizeof(NxtTensor));
    if (!new_outputs) return -1;
    response->outputs = new_outputs;

    NxtTensor *t = &response->outputs[idx];
    memset(t, 0, sizeof(*t));
    t->name       = strdup(name);
    t->dtype      = dtype;
    t->dims_count = dims;
    t->byte_size  = byte_size;
    if (dims > 0 && shape) {
        t->shape = malloc(dims * sizeof(int64_t));
        if (t->shape) memcpy(t->shape, shape, dims * sizeof(int64_t));
    }
    response->output_count++;
    return 0;
}

/* ── String utilities ─────────────────────────────────────────────────── */

const char *nxt_datatype_str(NxtDataType dtype) {
    switch (dtype) {
    case NXT_TYPE_BOOL:   return "BOOL";
    case NXT_TYPE_UINT8:  return "UINT8";
    case NXT_TYPE_UINT16: return "UINT16";
    case NXT_TYPE_UINT32: return "UINT32";
    case NXT_TYPE_UINT64: return "UINT64";
    case NXT_TYPE_INT8:   return "INT8";
    case NXT_TYPE_INT16:  return "INT16";
    case NXT_TYPE_INT32:  return "INT32";
    case NXT_TYPE_INT64:  return "INT64";
    case NXT_TYPE_FP16:   return "FP16";
    case NXT_TYPE_FP32:   return "FP32";
    case NXT_TYPE_FP64:   return "FP64";
    case NXT_TYPE_BYTES:  return "BYTES";
    default:              return "UNKNOWN";
    }
}

const char *nxt_memory_type_str(NxtMemoryType mem) {
    switch (mem) {
    case NXT_MEM_CPU:        return "CPU";
    case NXT_MEM_GPU:        return "GPU";
    case NXT_MEM_CPU_PINNED: return "CPU_PINNED";
    default:                 return "UNKNOWN";
    }
}

const char *nxt_backend_state_str(NxtBackendState state) {
    switch (state) {
    case NXT_BACKEND_STATE_UNINITIALIZED: return "UNINITIALIZED";
    case NXT_BACKEND_STATE_INITIALIZING:  return "INITIALIZING";
    case NXT_BACKEND_STATE_READY:         return "READY";
    case NXT_BACKEND_STATE_ERROR:         return "ERROR";
    case NXT_BACKEND_STATE_UNLOADING:     return "UNLOADING";
    default:                              return "UNKNOWN";
    }
}
