/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Triton-style model version policy tests: LATEST, ALL, SPECIFIC.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "backend.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { tests_run++; printf("  RUN  %s ... ", name); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg)  do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ── Test: Version Policy LATEST ──────────────────────────────────────── */
static void test_version_policy_latest(void) {
    TEST("version_policy_latest");
    NxtModelVersionConfig config = {
        .policy = NXT_VERSION_LATEST,
        .num_versions = 3
    };
    ASSERT(config.policy == NXT_VERSION_LATEST, "policy should be LATEST");
    ASSERT(config.num_versions == 3, "num_versions should be 3");

    int rc = nxt_backend_manager_init(".");
    ASSERT(rc == 0, "init failed");

    /* Set version policy on non-existent model should fail */
    rc = nxt_model_set_version_policy("nonexistent", &config);
    ASSERT(rc == -1, "set_version_policy on nonexistent model should fail");

    nxt_backend_manager_fini();
    PASS();
}

/* ── Test: Version Policy ALL ─────────────────────────────────────────── */
static void test_version_policy_all(void) {
    TEST("version_policy_all");
    NxtModelVersionConfig config = {
        .policy = NXT_VERSION_ALL,
        .num_versions = 0
    };
    ASSERT(config.policy == NXT_VERSION_ALL, "policy should be ALL");

    nxt_backend_manager_init(".");
    /* find with version=-1 (ALL) should work */
    ASSERT(nxt_model_find("any_model", -1) == NULL, "no model registered yet");
    nxt_backend_manager_fini();
    PASS();
}

/* ── Test: Version Policy SPECIFIC ────────────────────────────────────── */
static void test_version_policy_specific(void) {
    TEST("version_policy_specific");
    int32_t specific_versions[] = {1, 3, 5};
    NxtModelVersionConfig config = {
        .policy = NXT_VERSION_SPECIFIC,
        .versions = specific_versions,
        .versions_count = 3
    };
    ASSERT(config.policy == NXT_VERSION_SPECIFIC, "policy should be SPECIFIC");
    ASSERT(config.versions_count == 3, "versions_count should be 3");
    ASSERT(config.versions[0] == 1, "versions[0] should be 1");
    ASSERT(config.versions[1] == 3, "versions[1] should be 3");
    ASSERT(config.versions[2] == 5, "versions[2] should be 5");

    nxt_backend_manager_init(".");
    nxt_backend_manager_fini();
    PASS();
}

/* ── Test: Model Config Structure ─────────────────────────────────────── */
static void test_model_config_structure(void) {
    TEST("model_config_structure");
    NxtModelConfig config;
    memset(&config, 0, sizeof(config));

    config.name = strdup("test_model");
    config.backend_name = strdup("onnxruntime");
    config.model_path = strdup("/models/test_model/1");
    config.max_batch_size = 32;

    /* Input tensor */
    NxtTensor *input = calloc(1, sizeof(NxtTensor));
    input->name = "input_ids";
    input->dtype = NXT_TYPE_INT64;
    int64_t shape[] = {-1, 512};
    input->shape = malloc(2 * sizeof(int64_t));
    memcpy(input->shape, shape, 2 * sizeof(int64_t));
    input->dims_count = 2;
    config.inputs = calloc(1, sizeof(NxtTensor*));
    config.inputs[0] = input;
    config.input_count = 1;

    /* Output tensor */
    NxtTensor *output = calloc(1, sizeof(NxtTensor));
    output->name = "logits";
    output->dtype = NXT_TYPE_FP32;
    int64_t out_shape[] = {-1, 512, 32000};
    output->shape = malloc(3 * sizeof(int64_t));
    memcpy(output->shape, out_shape, 3 * sizeof(int64_t));
    output->dims_count = 3;
    config.outputs = calloc(1, sizeof(NxtTensor*));
    config.outputs[0] = output;
    config.output_count = 1;

    /* Instance groups */
    config.instance_groups = calloc(1, sizeof(NxtInstanceGroup));
    config.instance_groups[0].count = 2;
    config.instance_groups[0].kind = NXT_INSTANCE_KIND_GPU;
    int32_t gpus[] = {0, 1};
    config.instance_groups[0].gpus = malloc(2 * sizeof(int32_t));
    memcpy(config.instance_groups[0].gpus, gpus, 2 * sizeof(int32_t));
    config.instance_groups[0].gpus_count = 2;
    config.instance_group_count = 1;

    /* Version config */
    config.version_config.policy = NXT_VERSION_LATEST;
    config.version_config.num_versions = 5;

    /* Dynamic batching */
    config.dynamic_batching.preferred_batch_size_ratio = 4.0f;
    config.dynamic_batching.max_queue_delay_ms = 50.0;
    config.dynamic_batching.preserve_ordering = true;
    config.dynamic_batching.priority_levels = 2;

    /* Verify fields */
    ASSERT(strcmp(config.name, "test_model") == 0, "name mismatch");
    ASSERT(strcmp(config.backend_name, "onnxruntime") == 0, "backend_name mismatch");
    ASSERT(config.max_batch_size == 32, "max_batch_size mismatch");
    ASSERT(config.input_count == 1, "input_count mismatch");
    ASSERT(config.output_count == 1, "output_count mismatch");
    ASSERT(config.instance_group_count == 1, "instance_group_count mismatch");
    ASSERT(config.instance_groups[0].count == 2, "instance count mismatch");
    ASSERT(config.instance_groups[0].kind == NXT_INSTANCE_KIND_GPU, "instance kind mismatch");
    ASSERT(config.version_config.policy == NXT_VERSION_LATEST, "version policy mismatch");
    ASSERT(config.version_config.num_versions == 5, "num_versions mismatch");

    /* Cleanup */
    free(config.name);
    free(config.backend_name);
    free(config.model_path);
    free(input->shape);
    free(input);
    free(config.inputs);
    free(output->shape);
    free(output);
    free(config.outputs);
    free(config.instance_groups[0].gpus);
    free(config.instance_groups);

    PASS();
}

/* ── Test: Instance Kind Enum ─────────────────────────────────────────── */
static void test_instance_kind_enum(void) {
    TEST("instance_kind_enum");
    ASSERT(NXT_INSTANCE_KIND_GPU == 0, "GPU kind should be 0");
    ASSERT(NXT_INSTANCE_KIND_CPU == 1, "CPU kind should be 1");
    PASS();
}

/* ── Test: Scheduler Policy Enum ──────────────────────────────────────── */
static void test_scheduler_policy_enum(void) {
    TEST("scheduler_policy_enum");
    ASSERT(NXT_SCHED_DYNAMIC == 0, "DYNAMIC should be 0");
    ASSERT(NXT_SCHED_SEQUENCE == 1, "SEQUENCE should be 1");
    ASSERT(NXT_SCHED_ENSEMBLE == 2, "ENSEMBLE should be 2");
    PASS();
}

/* ── Test: Version Find with Different Versions ───────────────────────── */
static void test_version_find_semantics(void) {
    TEST("version_find_semantics");
    nxt_backend_manager_init(".");

    /* version=0 means LATEST */
    ASSERT(nxt_model_find("test", 0) == NULL, "no model with version 0");

    /* version=-1 means ALL */
    ASSERT(nxt_model_find("test", -1) == NULL, "no model with version -1");

    /* version=3 means SPECIFIC */
    ASSERT(nxt_model_find("test", 3) == NULL, "no model with version 3");

    ASSERT(nxt_model_count() == 0, "model count should be 0");

    nxt_backend_manager_fini();
    PASS();
}

/* ── Test: Dynamic Batching Config ────────────────────────────────────── */
static void test_dynamic_batching_config(void) {
    TEST("dynamic_batching_config");
    NxtDynamicBatchingConfig cfg = {
        .preferred_batch_size_ratio = 2.5f,
        .max_queue_delay_ms = 75.0,
        .preserve_ordering = false,
        .priority_levels = 3
    };
    ASSERT(cfg.preferred_batch_size_ratio == 2.5f, "batch size ratio mismatch");
    ASSERT(cfg.max_queue_delay_ms == 75.0, "queue delay mismatch");
    ASSERT(cfg.preserve_ordering == false, "preserve_ordering mismatch");
    ASSERT(cfg.priority_levels == 3, "priority_levels mismatch");
    PASS();
}

/* ── Test: Model Version Config Edge Cases ────────────────────────────── */
static void test_version_config_edge_cases(void) {
    TEST("version_config_edge_cases");

    /* LATEST with 0 versions: invalid but structurally valid */
    NxtModelVersionConfig v1 = { .policy = NXT_VERSION_LATEST, .num_versions = 0 };
    ASSERT(v1.policy == NXT_VERSION_LATEST, "LATEST policy");

    /* ALL with num_versions 0 */
    NxtModelVersionConfig v2 = { .policy = NXT_VERSION_ALL, .num_versions = 0 };
    ASSERT(v2.policy == NXT_VERSION_ALL, "ALL policy");

    /* SPECIFIC with empty versions list */
    NxtModelVersionConfig v3 = {
        .policy = NXT_VERSION_SPECIFIC,
        .versions = NULL,
        .versions_count = 0
    };
    ASSERT(v3.policy == NXT_VERSION_SPECIFIC, "SPECIFIC policy");
    ASSERT(v3.versions_count == 0, "zero versions");

    PASS();
}

/* ── Test Runner ──────────────────────────────────────────────────────── */
int main(void) {
    printf("=== xLLM Model Version Test Suite ===\n\n");

    test_version_policy_latest();
    test_version_policy_all();
    test_version_policy_specific();
    test_model_config_structure();
    test_instance_kind_enum();
    test_scheduler_policy_enum();
    test_version_find_semantics();
    test_dynamic_batching_config();
    test_version_config_edge_cases();

    printf("\n=== Results: %d run, %d passed, %d failed ===\n",
           tests_run, tests_passed, tests_failed);
    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
