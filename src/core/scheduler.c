/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Continuous Batching Scheduler — request queue, dynamic batching,
 * and preemptive scheduling inspired by vLLM's scheduler design.
 */

#include "backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ── Internal request states ─────────────────────────────────────────────── */

typedef enum {
    REQ_WAITING   = 0,
    REQ_RUNNING   = 1,
    REQ_PREEMPTED = 2,
    REQ_FINISHED  = 3,
} SchedReqStatus;

/* ── Tokens per sequence tracking ────────────────────────────────────────── */

struct SchedRequest {
    char            *request_id;
    SchedReqStatus   status;
    int32_t          priority;
    uint64_t         arrival_time_us;
    uint32_t         prompt_length;        /* total prompt tokens */
    uint32_t         max_output_tokens;    /* max tokens to generate */
    uint32_t         num_computed_tokens;  /* tokens already through KV-cache */
    uint32_t         num_output_tokens;    /* output tokens generated so far */
    uint32_t         num_preemptions;      /* times this request was preempted */
    bool             is_prefill_chunk;     /* true if this is a partial prefill */
    NxtInferRequest  *infer_request;       /* the original request */
};

/* ── Scheduler state ─────────────────────────────────────────────────────── */

typedef struct {
    SchedRequest      **requests;          /* all requests (hash by id) */
    uint32_t             request_capacity;
    uint32_t             request_count;

    SchedRequest      **running;           /* currently active requests */
    uint32_t             running_count;
    uint32_t             running_capacity;

    SchedRequest      **waiting;           /* queued requests (FCFS/priority) */
    uint32_t             waiting_count;
    uint32_t             waiting_capacity;
    uint32_t             waiting_head;     /* ring buffer head */

    uint32_t             max_num_seqs;
    uint32_t             max_num_scheduled_tokens;  /* token_budget per step */
    bool                 enable_chunked_prefill;
    uint32_t             long_prefill_token_threshold;
    NxtSchedulerPolicy   policy;
    bool                 scheduler_reserve_full_isl;

    /* Stats */
    uint64_t             completed_count;
    uint64_t             preempted_count;
    uint64_t             total_token_count;
    uint64_t             step_count;
} SchedState;

static SchedState g_sched;
static bool        g_sched_initialized = false;

#define MAX_REQUESTS 2048
#define MAX_RUNNING   128

/* ── Internal helpers ────────────────────────────────────────────────────── */

static uint64_t nxt_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + (uint64_t)ts.tv_nsec / 1000;
}

static SchedRequest *sched_find_request(const char *request_id) {
    for (uint32_t i = 0; i < g_sched.request_count; i++) {
        if (strcmp(g_sched.requests[i]->request_id, request_id) == 0)
            return g_sched.requests[i];
    }
    return NULL;
}

static int sched_request_compare_priority(const void *a, const void *b) {
    const SchedRequest *ra = *(const SchedRequest **)a;
    const SchedRequest *rb = *(const SchedRequest **)b;
    if (ra->priority != rb->priority)
        return ra->priority - rb->priority;  /* lower = higher priority */
    if (ra->arrival_time_us != rb->arrival_time_us)
        return (ra->arrival_time_us > rb->arrival_time_us) ? 1 : -1;
    return 0;
}

static void sched_sort_waiting_by_priority(void) {
    if (g_sched.waiting_count > 1) {
        qsort(g_sched.waiting + g_sched.waiting_head,
              g_sched.waiting_count - g_sched.waiting_head,
              sizeof(SchedRequest *), sched_request_compare_priority);
        /* Wrap-around portion */
        if (g_sched.waiting_head > 0) {
            qsort(g_sched.waiting, g_sched.waiting_head,
                  sizeof(SchedRequest *), sched_request_compare_priority);
        }
    }
}

static void sched_remove_from_running(uint32_t idx) {
    if (idx < g_sched.running_count - 1) {
        memmove(&g_sched.running[idx], &g_sched.running[idx + 1],
                (g_sched.running_count - idx - 1) * sizeof(SchedRequest *));
    }
    g_sched.running_count--;
}

static void sched_preempt_request(SchedRequest *req) {
    req->status = REQ_PREEMPTED;
    req->num_computed_tokens = 0;
    req->num_preemptions++;
    g_sched.preempted_count++;

    /* Re-insert into waiting queue at the front */
    if (g_sched.waiting_count >= g_sched.waiting_capacity) return;
    if (g_sched.waiting_head == 0)
        g_sched.waiting_head = g_sched.waiting_capacity;
    g_sched.waiting_head--;
    g_sched.waiting[g_sched.waiting_head] = req;
    g_sched.waiting_count++;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int nxt_scheduler_init(const NxtSchedulerConfig *config) {
    if (g_sched_initialized) nxt_scheduler_fini();

    memset(&g_sched, 0, sizeof(g_sched));
    g_sched.request_capacity  = MAX_REQUESTS;
    g_sched.running_capacity  = MAX_RUNNING;
    g_sched.waiting_capacity  = MAX_REQUESTS;

    g_sched.requests = calloc(g_sched.request_capacity, sizeof(SchedRequest *));
    g_sched.running  = calloc(g_sched.running_capacity, sizeof(SchedRequest *));
    g_sched.waiting  = calloc(g_sched.waiting_capacity, sizeof(SchedRequest *));
    if (!g_sched.requests || !g_sched.running || !g_sched.waiting) {
        free(g_sched.requests);
        free(g_sched.running);
        free(g_sched.waiting);
        return -1;
    }

    if (config) {
        g_sched.max_num_seqs                = config->max_preferred_batch_size;
        g_sched.max_num_scheduled_tokens    = config->max_queue_size > 0 ? config->max_queue_size : 4096;
        g_sched.enable_chunked_prefill      = (config->policy != NXT_SCHED_SEQUENCE);
        g_sched.long_prefill_token_threshold = 0;
        g_sched.policy                      = config->policy;
        g_sched.scheduler_reserve_full_isl  = config->preserve_ordering;
    } else {
        g_sched.max_num_seqs                = 32;
        g_sched.max_num_scheduled_tokens    = 2048;
        g_sched.enable_chunked_prefill      = true;
        g_sched.policy                      = NXT_SCHED_DYNAMIC;
    }

    g_sched_initialized = true;
    return 0;
}

int nxt_scheduler_fini(void) {
    if (!g_sched_initialized) return 0;
    for (uint32_t i = 0; i < g_sched.request_count; i++)
        free(g_sched.requests[i]);
    free(g_sched.requests);
    free(g_sched.running);
    free(g_sched.waiting);
    memset(&g_sched, 0, sizeof(g_sched));
    g_sched_initialized = false;
    return 0;
}

int nxt_scheduler_enqueue(NxtInferRequest *request) {
    if (!g_sched_initialized || !request) return -1;
    if (g_sched.request_count >= g_sched.request_capacity) return -1;
    if (sched_find_request(request->request_id)) return -1; /* duplicate */

    SchedRequest *sreq = calloc(1, sizeof(SchedRequest));
    if (!sreq) return -1;
    sreq->request_id       = strdup(request->request_id);
    sreq->status           = REQ_WAITING;
    sreq->priority         = request->priority;
    sreq->arrival_time_us  = nxt_time_us();
    sreq->prompt_length    = 0;  /* will be set when tokenized */
    sreq->max_output_tokens = 256;
    sreq->infer_request    = request;

    g_sched.requests[g_sched.request_count] = sreq;
    g_sched.request_count++;

    if (g_sched.waiting_count >= g_sched.waiting_capacity) {
        free(sreq->request_id);
        free(sreq);
        g_sched.request_count--;
        return -1;
    }

    uint32_t tail = (g_sched.waiting_head + g_sched.waiting_count) % g_sched.waiting_capacity;
    g_sched.waiting[tail] = sreq;
    g_sched.waiting_count++;

    if (g_sched.policy == NXT_SCHED_DYNAMIC) {
        sched_sort_waiting_by_priority();
    }

    return 0;
}

int nxt_scheduler_enqueue_tokenized(const char *request_id,
                                     int32_t priority,
                                     const int32_t *token_ids,
                                     uint32_t  token_count,
                                     uint32_t  max_output_tokens) {
    if (!g_sched_initialized || !request_id || !token_ids || token_count == 0) return -1;
    if (g_sched.request_count >= g_sched.request_capacity) return -1;
    if (sched_find_request(request_id)) return -1;

    SchedRequest *sreq = calloc(1, sizeof(SchedRequest));
    if (!sreq) return -1;
    sreq->request_id        = strdup(request_id);
    sreq->status            = REQ_WAITING;
    sreq->priority          = priority;
    sreq->arrival_time_us   = nxt_time_us();
    sreq->prompt_length     = token_count;
    sreq->max_output_tokens = max_output_tokens;
    sreq->infer_request     = NULL;

    g_sched.requests[g_sched.request_count] = sreq;
    g_sched.request_count++;

    if (g_sched.waiting_count >= g_sched.waiting_capacity) {
        free(sreq->request_id);
        free(sreq);
        g_sched.request_count--;
        return -1;
    }

    uint32_t tail = (g_sched.waiting_head + g_sched.waiting_count) % g_sched.waiting_capacity;
    g_sched.waiting[tail] = sreq;
    g_sched.waiting_count++;

    if (g_sched.policy == NXT_SCHED_DYNAMIC) {
        sched_sort_waiting_by_priority();
    }

    return 0;
}

/*
 * nxt_scheduler_schedule_step — one scheduling iteration.
 *
 * Algorithm (simplified from vLLM):
 *   1. Schedule RUNNING requests first (maintain continuity)
 *   2. Schedule WAITING requests (new + resumed preempted)
 *   3. If allocation fails, preempt lowest-priority running request
 *   4. Return the set of scheduled requests with token budgets
 *
 * Returns the number of requests scheduled (0 = no work to do).
 */
int nxt_scheduler_schedule_step(SchedRequest ***out_scheduled,
                                 uint32_t *out_count,
                                 uint32_t *out_token_budget_per_req) {
    if (!g_sched_initialized || !out_scheduled || !out_count) return -1;

    *out_scheduled = NULL;
    *out_count = 0;

    uint32_t token_budget = g_sched.max_num_scheduled_tokens;
    uint32_t scheduled_count = 0;
    uint32_t scheduled_capacity = 64;

    SchedRequest **scheduled = calloc(scheduled_capacity, sizeof(SchedRequest *));
    if (!scheduled) return -1;

    /* ── Step 1: Schedule RUNNING requests ───────────────────────────── */

    uint32_t req_idx = 0;
    while (req_idx < g_sched.running_count && token_budget > 0) {
        SchedRequest *req = g_sched.running[req_idx];

        /* Calculate tokens needed for this request */
        uint32_t num_new_tokens;
        if (req->num_computed_tokens == 0) {
            /* Prefill phase */
            num_new_tokens = req->prompt_length;
            if (g_sched.enable_chunked_prefill) {
                if (g_sched.long_prefill_token_threshold > 0 &&
                    num_new_tokens > g_sched.long_prefill_token_threshold) {
                    num_new_tokens = g_sched.long_prefill_token_threshold;
                }
                if (num_new_tokens > token_budget) {
                    num_new_tokens = token_budget;
                }
            } else if (num_new_tokens > token_budget) {
                /* Can't fit — preempt */
                sched_preempt_request(req);
                sched_remove_from_running(req_idx);
                continue;
            }
        } else {
            /* Decode phase — 1 token at a time */
            num_new_tokens = 1;
        }

        if (num_new_tokens == 0) {
            req_idx++;
            continue;
        }

        num_new_tokens = (num_new_tokens < token_budget) ? num_new_tokens : token_budget;

        /* Allocate in scheduled array */
        if (scheduled_count >= scheduled_capacity) {
            scheduled_capacity *= 2;
            SchedRequest **tmp = realloc(scheduled,
                                          scheduled_capacity * sizeof(SchedRequest *));
            if (!tmp) { free(scheduled); return -1; }
            scheduled = tmp;
        }

        scheduled[scheduled_count] = req;
        if (out_token_budget_per_req)
            out_token_budget_per_req[scheduled_count] = num_new_tokens;
        scheduled_count++;
        token_budget -= num_new_tokens;
        req_idx++;
    }

    /* ── Step 2: Schedule WAITING requests ────────────────────────────── */

    uint32_t waiting_processed = 0;
    while (waiting_processed < g_sched.waiting_count && token_budget > 0) {
        if (g_sched.running_count >= g_sched.max_num_seqs) break;

        uint32_t idx = (g_sched.waiting_head + waiting_processed) % g_sched.waiting_capacity;
        SchedRequest *req = g_sched.waiting[idx];
        waiting_processed++;

        if (!req) continue;
        if (req->status == REQ_FINISHED) continue;

        /* Calculate tokens for this new/resumed request */
        uint32_t num_new_tokens = req->prompt_length - req->num_computed_tokens;
        if (num_new_tokens == 0) num_new_tokens = 1; /* decode-only request */

        if (g_sched.enable_chunked_prefill) {
            if (g_sched.long_prefill_token_threshold > 0 &&
                num_new_tokens > g_sched.long_prefill_token_threshold) {
                num_new_tokens = g_sched.long_prefill_token_threshold;
            }
            if (num_new_tokens > token_budget) {
                num_new_tokens = token_budget;
            }
        } else if (num_new_tokens > token_budget) {
            /* FCFS/SEQUENCE: only admit if full prefill fits */
            if (g_sched.scheduler_reserve_full_isl) break;
            num_new_tokens = token_budget;
        }

        if (num_new_tokens == 0) continue;

        /* Allocate in scheduled array */
        if (scheduled_count >= scheduled_capacity) {
            scheduled_capacity *= 2;
            SchedRequest **tmp = realloc(scheduled,
                                          scheduled_capacity * sizeof(SchedRequest *));
            if (!tmp) { free(scheduled); return -1; }
            scheduled = tmp;
        }

        scheduled[scheduled_count] = req;
        if (out_token_budget_per_req)
            out_token_budget_per_req[scheduled_count] = num_new_tokens;
        scheduled_count++;
        token_budget -= num_new_tokens;

        /* Move from waiting to running */
        g_sched.running[g_sched.running_count] = req;
        g_sched.running_count++;
        req->status = REQ_RUNNING;

        /* Mark prefill chunk status */
        req->is_prefill_chunk = ((req->num_computed_tokens + num_new_tokens) < req->prompt_length);
    }

    /* Remove processed items from waiting queue */
    if (waiting_processed > 0) {
        g_sched.waiting_head = (g_sched.waiting_head + waiting_processed) % g_sched.waiting_capacity;
        g_sched.waiting_count -= waiting_processed;
    }

    /* ── Step 3: Update state for scheduled requests ──────────────────── */

    for (uint32_t i = 0; i < scheduled_count; i++) {
        SchedRequest *req = scheduled[i];
        uint32_t tokens = out_token_budget_per_req ? out_token_budget_per_req[i] : 0;
        req->num_computed_tokens += tokens;
        g_sched.total_token_count += tokens;

        /* Check for completion */
        if (req->num_computed_tokens >= req->prompt_length + req->num_output_tokens) {
            if (req->num_output_tokens >= req->max_output_tokens) {
                req->status = REQ_FINISHED;
                g_sched.completed_count++;
            }
        }
    }

    /* Remove finished requests from running */
    for (uint32_t i = 0; i < g_sched.running_count; ) {
        if (g_sched.running[i]->status == REQ_FINISHED) {
            sched_remove_from_running(i);
        } else {
            i++;
        }
    }

    g_sched.step_count++;
    *out_scheduled = scheduled;
    *out_count = scheduled_count;
    return (scheduled_count > 0) ? 0 : 1;  /* 0=work done, 1=idle */
}

void nxt_scheduler_complete_request(const char *request_id) {
    SchedRequest *req = sched_find_request(request_id);
    if (!req) return;
    req->status = REQ_FINISHED;
    g_sched.completed_count++;

    /* Remove from running */
    for (uint32_t i = 0; i < g_sched.running_count; i++) {
        if (g_sched.running[i] == req) {
            sched_remove_from_running(i);
            break;
        }
    }
}

/* ── Backend Manager integration ────────────────────────────────────────── */

int nxt_scheduler_poll(void) {
    if (!g_sched_initialized) return -1;

    SchedRequest **scheduled = NULL;
    uint32_t count = 0;
    uint32_t *token_budgets = calloc(MAX_RUNNING, sizeof(uint32_t));

    int rc = nxt_scheduler_schedule_step(&scheduled, &count, token_budgets);
    if (rc < 0) {
        free(token_budgets);
        return -1;
    }
    if (count == 0) {
        free(scheduled);
        free(token_budgets);
        return 0;
    }

    /* Execute scheduled requests via backend */
    for (uint32_t i = 0; i < count; i++) {
        SchedRequest *req = scheduled[i];
        if (!req->infer_request) continue;

        NxtModel *model = nxt_model_find("default", 0);
        if (model && model->backend && model->instances &&
            model->backend->api.instance_exec) {
            model->backend->api.instance_exec(model->instances,
                                               &req->infer_request, 1);
        }
    }

    free(scheduled);
    free(token_budgets);
    return 0;
}

/* ── Query helpers ───────────────────────────────────────────────────────── */

uint64_t nxt_scheduler_completed_count(void) { return g_sched.completed_count; }
uint64_t nxt_scheduler_queued_count(void)    { return g_sched.waiting_count; }

double nxt_scheduler_avg_batch_size(void) {
    if (g_sched.step_count == 0) return 0.0;
    return (double)g_sched.total_token_count / (double)g_sched.step_count;
}

uint32_t nxt_scheduler_running_count(void)  { return g_sched.running_count; }
uint32_t nxt_scheduler_waiting_count(void)  { return g_sched.waiting_count; }
uint64_t nxt_scheduler_preempted_count(void) { return g_sched.preempted_count; }

/* ── Strategy configuration ──────────────────────────────────────────────── */

void nxt_scheduler_set_policy(NxtSchedulerPolicy policy) {
    g_sched.policy = policy;
}

NxtSchedulerPolicy nxt_scheduler_get_policy(void) {
    return g_sched.policy;
}

void nxt_scheduler_set_token_budget(uint32_t budget) {
    g_sched.max_num_scheduled_tokens = budget;
}

uint32_t nxt_scheduler_get_token_budget(void) {
    return g_sched.max_num_scheduled_tokens;
}
