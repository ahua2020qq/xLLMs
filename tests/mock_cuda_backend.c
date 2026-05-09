/*
 * xLLM — Mock CUDA backend stub for dlopen testing.
 * Self-contained: avoids the naming conflict between backend.h:211
 * (manager API) and cuda_backend.h:47 (plugin entry point).
 *
 * Exports: nxt_backend_init / nxt_backend_run / nxt_backend_fini
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Minimal type definitions matching include/backend.h */
typedef enum {
    NXT_BACKEND_STATE_UNINITIALIZED = 0,
    NXT_BACKEND_STATE_INITIALIZING  = 1,
    NXT_BACKEND_STATE_READY         = 2,
    NXT_BACKEND_STATE_ERROR         = 3,
    NXT_BACKEND_STATE_UNLOADING     = 4,
} NxtBackendState;

/*
 * NxtBackend — must match include/backend.h:141-151 layout exactly.
 * NxtBackendAPI = 8 function pointers = 64 bytes on x86_64.
 */
typedef struct {
    char            *name;               /* offset  0 */
    char            *so_path;            /* offset  8 */
    void            *dl_handle;          /* offset 16 */
    void            *api_pad[8];         /* offset 24 — 64-byte NxtBackendAPI */
    NxtBackendState  state;              /* offset 88 */
    int32_t          priority;           /* offset 92 */
    uint32_t         model_count;        /* offset 96 */
    uint32_t         _pad;               /* offset 100 */
    void            *models;             /* offset 104 */
    void            *backend_state;      /* offset 112 */
} NxtBackend;

int nxt_backend_init(NxtBackend *backend) {
    if (!backend) return -1;
    void *st = calloc(1, 64);
    if (!st) return -1;
    backend->backend_state = st;
    backend->state = NXT_BACKEND_STATE_READY;
    return 0;
}

int nxt_backend_run(NxtBackend *backend, void *input, void *output) {
    if (!backend || !backend->backend_state) return -1;
    if (!input || !output) return -1;
    /* stub: no-op, return success */
    return 0;
}

int nxt_backend_fini(NxtBackend *backend) {
    if (!backend) return -1;
    if (backend->backend_state) {
        free(backend->backend_state);
        backend->backend_state = NULL;
    }
    backend->state = NXT_BACKEND_STATE_UNINITIALIZED;
    return 0;
}
