#ifndef XLLM_BACKEND_H
#define XLLM_BACKEND_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Data types ─────────────────────────────────────────────────────── */

typedef enum {
    NXT_TYPE_BOOL      = 1,
    NXT_TYPE_UINT8     = 2,
    NXT_TYPE_UINT16    = 3,
    NXT_TYPE_UINT32    = 4,
    NXT_TYPE_UINT64    = 5,
    NXT_TYPE_INT8      = 6,
    NXT_TYPE_INT16     = 7,
    NXT_TYPE_INT32     = 8,
    NXT_TYPE_INT64     = 9,
    NXT_TYPE_FP16      = 10,
    NXT_TYPE_FP32      = 11,
    NXT_TYPE_FP64      = 12,
    NXT_TYPE_BYTES     = 13,
} NxtDataType;

typedef enum {
    NXT_MEM_CPU        = 0,
    NXT_MEM_GPU        = 1,
    NXT_MEM_CPU_PINNED = 2,
} NxtMemoryType;

typedef enum {
    NXT_BACKEND_STATE_UNINITIALIZED = 0,
    NXT_BACKEND_STATE_INITIALIZING  = 1,
    NXT_BACKEND_STATE_READY         = 2,
    NXT_BACKEND_STATE_ERROR         = 3,
    NXT_BACKEND_STATE_UNLOADING     = 4,
} NxtBackendState;

/* ─── Forward declarations ───────────────────────────────────────────── */

typedef struct NxtBackend       NxtBackend;
typedef struct NxtModel         NxtModel;
typedef struct NxtModelInstance  NxtModelInstance;
typedef struct NxtInferRequest  NxtInferRequest;
typedef struct NxtInferResponse NxtInferResponse;
typedef struct NxtBackendAPI    NxtBackendAPI;

/* ─── Tensor descriptor ──────────────────────────────────────────────── */

typedef struct {
    const char     *name;
    NxtDataType     dtype;
    int64_t        *shape;
    uint32_t        dims_count;
    uint64_t        byte_size;
    NxtMemoryType   memory_type;
    int32_t         memory_type_id;
} NxtTensor;

/* ─── Instance group configuration ────────────────────────────────────── */

typedef enum {
    NXT_INSTANCE_KIND_GPU = 0,
    NXT_INSTANCE_KIND_CPU = 1,
} NxtInstanceKind;

typedef struct {
    uint32_t        count;
    NxtInstanceKind kind;
    int32_t        *gpus;
    uint32_t        gpus_count;
} NxtInstanceGroup;

/* ─── Model configuration ────────────────────────────────────────────── */

typedef enum {
    NXT_VERSION_LATEST   = 0,
    NXT_VERSION_ALL      = 1,
    NXT_VERSION_SPECIFIC = 2,
} NxtVersionPolicy;

typedef struct {
    NxtVersionPolicy policy;
    uint32_t         num_versions;   /* for LATEST */
    int32_t         *versions;       /* for SPECIFIC */
    uint32_t         versions_count;
} NxtModelVersionConfig;

typedef struct {
    float  preferred_batch_size_ratio;
    double max_queue_delay_ms;
    bool   preserve_ordering;
    int32_t priority_levels;
} NxtDynamicBatchingConfig;

typedef struct {
    char                    *name;
    char                    *backend_name;
    char                    *model_path;
    int32_t                  max_batch_size;
    uint32_t                 input_count;
    NxtTensor              **inputs;
    uint32_t                 output_count;
    NxtTensor              **outputs;
    uint32_t                 instance_group_count;
    NxtInstanceGroup        *instance_groups;
    NxtModelVersionConfig    version_config;
    NxtDynamicBatchingConfig dynamic_batching;
} NxtModelConfig;

/* ─── Backend API function table ─────────────────────────────────────── */

typedef int (*NxtBackendInitFn)(NxtBackend *backend);
typedef int (*NxtBackendRunFn)(NxtBackend *backend, void *input, void *output);
typedef int (*NxtBackendFiniFn)(NxtBackend *backend);
typedef int (*NxtModelInitFn)(NxtModel *model);
typedef int (*NxtModelFiniFn)(NxtModel *model);
typedef int (*NxtModelInstanceInitFn)(NxtModelInstance *instance);
typedef int (*NxtModelInstanceFiniFn)(NxtModelInstance *instance);
typedef int (*NxtModelInstanceExecFn)(NxtModelInstance *instance,
                                       NxtInferRequest  **requests,
                                       uint32_t           request_count);

struct NxtBackendAPI {
    NxtBackendInitFn         backend_init;
    NxtBackendRunFn          backend_run;
    NxtBackendFiniFn         backend_fini;
    NxtModelInitFn           model_init;
    NxtModelFiniFn           model_fini;
    NxtModelInstanceInitFn   instance_init;
    NxtModelInstanceFiniFn   instance_fini;
    NxtModelInstanceExecFn   instance_exec;
};

/* ─── Backend handle ─────────────────────────────────────────────────── */

struct NxtBackend {
    char            *name;
    char            *so_path;
    void            *dl_handle;
    NxtBackendAPI    api;
    NxtBackendState  state;
    int32_t          priority;
    uint32_t         model_count;
    NxtModel        *models;
    void            *backend_state;
};

/* ─── Model handle ───────────────────────────────────────────────────── */

struct NxtModel {
    char               *name;
    int32_t             version;
    NxtModelConfig      config;
    NxtBackend         *backend;
    NxtModelInstance   *instances;
    uint32_t            instance_count;
    NxtBackendState     state;
    void               *backend_state;
};

/* ─── Model instance handle ──────────────────────────────────────────── */

struct NxtModelInstance {
    NxtModel         *model;
    char             *name;
    NxtInstanceKind   kind;
    int32_t           device_id;
    void             *instance_state;
    uint64_t          exec_count;
    double            total_exec_ms;
};

/* ─── Inference request ──────────────────────────────────────────────── */

typedef struct NxtInferRequest {
    const char  *request_id;
    uint32_t     batch_size;
    uint32_t     input_count;
    NxtTensor   *inputs;
    uint32_t     requested_output_count;
    char       **requested_output_names;
    uint64_t     enqueue_time_ns;
    int32_t      priority;
} NxtInferRequest;

/* ─── Inference response ─────────────────────────────────────────────── */

struct NxtInferResponse {
    char        *request_id;
    int          error_code;
    char        *error_message;
    uint32_t     output_count;
    NxtTensor   *outputs;
};

/* ─── Backend Manager API ────────────────────────────────────────────── */

int nxt_backend_manager_init(const char *backend_dir);
int nxt_backend_manager_fini(void);

int nxt_backend_register(const char *so_path);
int nxt_backend_unregister(const char *name);

int nxt_backend_load(const char *so_path, const char *name, int32_t priority);
int nxt_backend_unload(const char *name);
int nxt_backend_run(const char *name, void *input, void *output);

NxtBackend *nxt_backend_find(const char *name);
uint32_t    nxt_backend_count(void);
NxtBackend *nxt_backend_list(void);
int         nxt_backend_discover(void);

int nxt_model_create(const char *name, const NxtModelConfig *config);
int nxt_model_destroy(NxtModel *model);

NxtModel *nxt_model_find(const char *name, int32_t version);
uint32_t  nxt_model_count(void);

int nxt_model_set_version_policy(const char *name, const NxtModelVersionConfig *config);

/* ─── Scheduler API ──────────────────────────────────────────────────── */

typedef enum {
    NXT_SCHED_DYNAMIC  = 0,
    NXT_SCHED_SEQUENCE = 1,
    NXT_SCHED_ENSEMBLE = 2,
} NxtSchedulerPolicy;

typedef struct {
    NxtSchedulerPolicy  policy;
    uint32_t            max_preferred_batch_size;
    double              max_queue_delay_ms;
    bool                preserve_ordering;
    int32_t             priority_levels;
    uint32_t            max_queue_size;
} NxtSchedulerConfig;

int nxt_scheduler_init(const NxtSchedulerConfig *config);
int nxt_scheduler_fini(void);

int nxt_scheduler_enqueue(NxtInferRequest *request);
int nxt_scheduler_poll(void);

uint64_t nxt_scheduler_completed_count(void);
uint64_t nxt_scheduler_queued_count(void);
double   nxt_scheduler_avg_batch_size(void);

/* ─── Continuous Batching Scheduler (advanced API) ───────────────────── */

/* Opaque handle for a scheduled request within the continuous batching engine */
typedef struct SchedRequest SchedRequest;

/* Enqueue a pre-tokenized request with explicit token IDs */
int nxt_scheduler_enqueue_tokenized(const char *request_id,
                                     int32_t priority,
                                     const int32_t *token_ids,
                                     uint32_t  token_count,
                                     uint32_t  max_output_tokens);

/* Execute one scheduling step: returns scheduled requests and their token budgets.
 * Returns 0 if work was scheduled, 1 if idle, -1 on error.
 * Caller must free *out_scheduled with free(). */
int nxt_scheduler_schedule_step(SchedRequest ***out_scheduled,
                                 uint32_t *out_count,
                                 uint32_t *out_token_budget_per_req);

/* Mark a request as fully completed (removes from running queue) */
void nxt_scheduler_complete_request(const char *request_id);

/* Query running/waiting/preempted counts */
uint32_t nxt_scheduler_running_count(void);
uint32_t nxt_scheduler_waiting_count(void);
uint64_t nxt_scheduler_preempted_count(void);

/* Configure scheduling strategy */
void               nxt_scheduler_set_policy(NxtSchedulerPolicy policy);
NxtSchedulerPolicy nxt_scheduler_get_policy(void);
void               nxt_scheduler_set_token_budget(uint32_t budget);
uint32_t           nxt_scheduler_get_token_budget(void);

/* ─── Model Loader API ────────────────────────────────────────────────── */

struct Gpt2Config;  /* forward declaration */

/* Detect the model architecture from a GGUF file header.
 * Returns a string like "llama", "mistral", "qwen2", "gpt2", or NULL on error. */
const char *nxt_model_loader_detect_arch(const char *path);

/* Load only the model configuration from a file (without loading weights).
 * Returns true on success. */
bool nxt_model_loader_load_config(const char *path, struct Gpt2Config *cfg_out);

/* ─── Health check API ───────────────────────────────────────────────── */

bool nxt_server_is_live(void);
bool nxt_server_is_ready(void);
bool nxt_model_is_ready(const char *name, int32_t version);

typedef struct {
    const char *name;
    int32_t     version;
    uint64_t    inference_count;
    uint64_t    execution_count;
    double      avg_latency_ms;
    double      queue_latency_ms;
} NxtModelStats;

int nxt_model_stats(const char *name, int32_t version, NxtModelStats *stats);

/* ─── Memory/utility helpers for backend implementors ───────────────── */

NxtInferResponse *nxt_response_alloc(const char *request_id);
void              nxt_response_free(NxtInferResponse *response);
int               nxt_response_set_output(NxtInferResponse *response,
                                           const char *name, NxtDataType dtype,
                                           const int64_t *shape, uint32_t dims,
                                           const void *data, uint64_t byte_size);

const char *nxt_datatype_str(NxtDataType dtype);
const char *nxt_memory_type_str(NxtMemoryType mem);
const char *nxt_backend_state_str(NxtBackendState state);

#ifdef __cplusplus
}
#endif

#endif /* XLLM_BACKEND_H */
