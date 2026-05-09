/*
 * xLLM — Next-Generation LLM Inference Engine
 * Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This header must not be removed. All derivative works must retain this notice.
 *
 * Activation Kernels — Adapted for xLLM
 * Original source: vLLM csrc/activation_kernels.cu
 * Copyright (c) 2023, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#include "operator_api.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// ── SiLU (Sigmoid Linear Unit) ─────────────────────────────────────────
template <typename T>
__device__ __forceinline__ T silu(const T& x) {
    // x * sigmoid(x)
    return (T)(((float)x) / (1.0f + expf((float)-x)));
}

// ── GELU (none approximation) ──────────────────────────────────────────
template <typename T>
__device__ __forceinline__ T gelu(const T& x) {
    const float f = (float)x;
    constexpr float ALPHA = M_SQRT1_2;
    return (T)(f * 0.5f * (1.0f + erff(f * ALPHA)));
}

// ── GELU (tanh approximation) ──────────────────────────────────────────
template <typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
    const float f = (float)x;
    constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float KAPPA = 0.044715f;
    float x_cube = f * f * f;
    float inner = BETA * (f + KAPPA * x_cube);
    return (T)(0.5f * f * (1.0f + tanhf(inner)));
}

// ── SiLU + Mul Gate Kernel ─────────────────────────────────────────────
// Computes: out = silu(gate) * up  where input = [gate|up] interleaved
template <typename scalar_t, bool ACT_FIRST>
__global__ void silu_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2*d]
    const int d) {

    const scalar_t* x_ptr = input + blockIdx.x * 2 * d;
    const scalar_t* y_ptr = x_ptr + d;
    scalar_t* out_ptr = out + blockIdx.x * d;

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        const scalar_t x = x_ptr[idx];
        const scalar_t y = y_ptr[idx];
        if (ACT_FIRST) {
            out_ptr[idx] = silu(x) * y;
        } else {
            out_ptr[idx] = x * silu(y);
        }
    }
}

// ── GELU + Mul Gate Kernel ─────────────────────────────────────────────
template <typename scalar_t>
__global__ void gelu_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2*d]
    const int d) {

    const scalar_t* x_ptr = input + blockIdx.x * 2 * d;
    const scalar_t* y_ptr = x_ptr + d;
    scalar_t* out_ptr = out + blockIdx.x * d;

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        out_ptr[idx] = gelu(x_ptr[idx]) * y_ptr[idx];
    }
}

// ── GELU Tanh + Mul Gate Kernel ────────────────────────────────────────
template <typename scalar_t>
__global__ void gelu_tanh_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2*d]
    const int d) {

    const scalar_t* x_ptr = input + blockIdx.x * 2 * d;
    const scalar_t* y_ptr = x_ptr + d;
    scalar_t* out_ptr = out + blockIdx.x * d;

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        out_ptr[idx] = gelu_tanh(x_ptr[idx]) * y_ptr[idx];
    }
}

// ── Element-wise GELU ──────────────────────────────────────────────────
template <typename scalar_t>
__global__ void gelu_elementwise_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {

    const scalar_t* in_ptr = input + blockIdx.x * d;
    scalar_t* out_ptr = out + blockIdx.x * d;

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        out_ptr[idx] = gelu(in_ptr[idx]);
    }
}

// ── Element-wise SiLU ──────────────────────────────────────────────────
template <typename scalar_t>
__global__ void silu_elementwise_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {

    const scalar_t* in_ptr = input + blockIdx.x * d;
    scalar_t* out_ptr = out + blockIdx.x * d;

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        out_ptr[idx] = silu(in_ptr[idx]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Host-side launchers
// ═══════════════════════════════════════════════════════════════════════

void nxt_silu_and_mul(void* out, const void* input,
                      int num_tokens, int d, int dtype_size,
                      cudaStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(MIN(d, 1024));

    if (dtype_size == 2) {  // half
        silu_and_mul_kernel<half, true>
            <<<grid, block, 0, stream>>>(
                (half*)out, (const half*)input, d);
    } else {  // float
        silu_and_mul_kernel<float, true>
            <<<grid, block, 0, stream>>>(
                (float*)out, (const float*)input, d);
    }
}

void nxt_mul_and_silu(void* out, const void* input,
                      int num_tokens, int d, int dtype_size,
                      cudaStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(MIN(d, 1024));

    if (dtype_size == 2) {
        silu_and_mul_kernel<half, false>
            <<<grid, block, 0, stream>>>(
                (half*)out, (const half*)input, d);
    } else {
        silu_and_mul_kernel<float, false>
            <<<grid, block, 0, stream>>>(
                (float*)out, (const float*)input, d);
    }
}

void nxt_gelu_and_mul(void* out, const void* input,
                      int num_tokens, int d, int dtype_size,
                      cudaStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(MIN(d, 1024));

    if (dtype_size == 2) {
        gelu_and_mul_kernel<half>
            <<<grid, block, 0, stream>>>(
                (half*)out, (const half*)input, d);
    } else {
        gelu_and_mul_kernel<float>
            <<<grid, block, 0, stream>>>(
                (float*)out, (const float*)input, d);
    }
}

void nxt_gelu_tanh_and_mul(void* out, const void* input,
                           int num_tokens, int d, int dtype_size,
                           cudaStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(MIN(d, 1024));

    if (dtype_size == 2) {
        gelu_tanh_and_mul_kernel<half>
            <<<grid, block, 0, stream>>>(
                (half*)out, (const half*)input, d);
    } else {
        gelu_tanh_and_mul_kernel<float>
            <<<grid, block, 0, stream>>>(
                (float*)out, (const float*)input, d);
    }
}

void nxt_gelu_elementwise(void* out, const void* input,
                          int num_tokens, int d, int dtype_size,
                          cudaStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(MIN(d, 1024));

    if (dtype_size == 2) {
        gelu_elementwise_kernel<half>
            <<<grid, block, 0, stream>>>(
                (half*)out, (const half*)input, d);
    } else {
        gelu_elementwise_kernel<float>
            <<<grid, block, 0, stream>>>(
                (float*)out, (const float*)input, d);
    }
}

void nxt_silu_elementwise(void* out, const void* input,
                          int num_tokens, int d, int dtype_size,
                          cudaStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(MIN(d, 1024));

    if (dtype_size == 2) {
        silu_elementwise_kernel<half>
            <<<grid, block, 0, stream>>>(
                (half*)out, (const half*)input, d);
    } else {
        silu_elementwise_kernel<float>
            <<<grid, block, 0, stream>>>(
                (float*)out, (const float*)input, d);
    }
}
