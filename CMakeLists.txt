<!--
  xLLMs — Next-Generation LLM Inference Engine
  Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
  SPDX-License-Identifier: Apache-2.0
-->
option(BUILD_CUDA_BACKEND "Build CUDA backend (requires CUDA Toolkit)" OFF)
option(USE_FLASHINFER "Enable FlashInfer-style attention adapter (requires BUILD_CUDA_BACKEND)" OFF)
option(USE_TENSORRT "Enable TensorRT-LLM integration (requires BUILD_CUDA_BACKEND)" OFF)

cmake_minimum_required(VERSION 3.16)
project(xLLM VERSION 0.5.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")
link_libraries(dl)

# ── Detect CUDA ───────────────────────────────────────────────────────
set(CUDA_FOUND OFF)
set(CUDA_ENABLED OFF)

if(BUILD_CUDA_BACKEND)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90")

    set(CUDA_FOUND TRUE CACHE BOOL "CUDA found" FORCE)
    set(CUDA_ENABLED ON)
    message(STATUS "CUDA backend: ON (${CUDAToolkit_VERSION})")
else()
    message(STATUS "BUILD_CUDA_BACKEND=OFF — CPU-only build")
endif()

set(XLLM_BACKEND_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/xllm/backends"
    CACHE STRING "Directory where backend .so files are installed")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xllm_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/xllm_config.h
    @ONLY
)

# ── Library target: xllm_core ────────────────────────────────────────
add_library(xllm_core STATIC
    src/core/memory_manager.c
    src/core/lru_k.c
    src/core/prefix_sharing.c
    src/core/backend_manager.c
    src/core/scheduler.c
    src/core/model_loader.c
)

target_include_directories(xllm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

find_package(Threads REQUIRED)
target_link_libraries(xllm_core PUBLIC Threads::Threads)

# ── Operator library (CUDA) ────────────────────────────────────────────
if(CUDA_ENABLED)
    add_library(xllm_operators STATIC
        operators/page_attention.cu
        operators/page_attention_v2.cu
        operators/activation_kernels.cu
    )

    target_include_directories(xllm_operators PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )

    target_link_libraries(xllm_operators PUBLIC
        CUDA::cudart
        xllm_core
    )

    # ── CUDA backend shared library (dlopen'd by backend_manager) ──────────
    add_library(xllm_cuda_backend SHARED
        backends/cuda_backend.c
    )

    target_include_directories(xllm_cuda_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_BINARY_DIR}/generated
    )

    target_link_libraries(xllm_cuda_backend PRIVATE
        xllm_operators
        xllm_core
        CUDA::cudart
    )

    set_target_properties(xllm_cuda_backend PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/backends"
    )

    target_compile_definitions(xllm_cuda_backend PRIVATE USE_CUDA)

    install(TARGETS xllm_cuda_backend
        LIBRARY DESTINATION ${XLLM_BACKEND_INSTALL_DIR}
    )
endif()

# ── Test executables ────────────────────────────────────────────────────
enable_testing()

add_executable(test_page_pool
    tests/test_page_pool.c
)
target_link_libraries(test_page_pool xllm_core)
add_test(NAME page_pool_tests COMMAND test_page_pool)

add_executable(test_prefix_sharing
    tests/test_prefix_sharing.c
)
target_link_libraries(test_prefix_sharing xllm_core)
add_test(NAME prefix_sharing_tests COMMAND test_prefix_sharing)

add_executable(test_scheduler
    tests/test_scheduler.c
)
target_link_libraries(test_scheduler xllm_core Threads::Threads m)
add_test(NAME scheduler_tests COMMAND test_scheduler)

add_executable(test_model_loader
    tests/test_model_loader.c
)
target_link_libraries(test_model_loader xllm_core m)
add_test(NAME model_loader_tests COMMAND test_model_loader)

add_executable(test_backend
    tests/test_backend.c
)
target_link_libraries(test_backend xllm_core Threads::Threads m dl)
add_test(NAME backend_tests COMMAND test_backend)

add_executable(test_model_version
    tests/test_model_version.c
)
target_link_libraries(test_model_version xllm_core Threads::Threads m dl)
add_test(NAME model_version_tests COMMAND test_model_version)

if(CUDA_ENABLED)
    math(EXPR XLLM_CUDART_VERSION
        "${CUDAToolkit_VERSION_MAJOR} * 1000 + ${CUDAToolkit_VERSION_MINOR} * 10")

    add_executable(test_attention
        tests/test_attention.c
    )
    target_include_directories(test_attention PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_definitions(test_attention PRIVATE
        CUDART_VERSION=${XLLM_CUDART_VERSION}
    )
    target_link_libraries(test_attention
        xllm_operators
        CUDA::cudart
    )
    add_test(NAME attention_tests COMMAND test_attention)
endif()

# ── Inference library ───────────────────────────────────────────────────
set(INFERENCE_SOURCES
    src/inference/transformer_block.c
    src/inference/tokenizer.c
    src/inference/decode_loop.c
)

add_library(xllm_inference STATIC ${INFERENCE_SOURCES})

target_include_directories(xllm_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_link_libraries(xllm_inference PUBLIC
    xllm_core
    m
)

# ── FlashInfer-style adapter (optional) ──────────────────────────────────
if(USE_FLASHINFER)
    if(NOT CUDA_ENABLED)
        message(FATAL_ERROR "USE_FLASHINFER requires BUILD_CUDA_BACKEND=ON")
    endif()

    add_library(xllm_flash_adapter STATIC
        operators/flashinfer_adapter.cu
    )

    target_include_directories(xllm_flash_adapter PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )

    target_link_libraries(xllm_flash_adapter PUBLIC
        CUDA::cudart
        xllm_operators
    )

    set_target_properties(xllm_flash_adapter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_definitions(xllm_flash_adapter PRIVATE
        USE_FLASHINFER=1
    )

    add_executable(test_attention_bench
        tests/test_attention_bench.c
    )
    target_include_directories(test_attention_bench PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_definitions(test_attention_bench PRIVATE
        CUDART_VERSION=${XLLM_CUDART_VERSION}
    )
    target_link_libraries(test_attention_bench
        xllm_flash_adapter
        xllm_operators
        CUDA::cudart
    )
    add_test(NAME attention_bench_tests COMMAND test_attention_bench)
endif()

# ── TensorRT-LLM adapter (optional) ────────────────────────────────────
if(USE_TENSORRT)
    if(NOT CUDA_ENABLED)
        message(FATAL_ERROR "USE_TENSORRT requires BUILD_CUDA_BACKEND=ON")
    endif()

    find_package(TensorRT REQUIRED)

    add_library(xllm_tensorrt_adapter STATIC
        src/adapter/tensorrt_adapter.c
    )

    target_include_directories(xllm_tensorrt_adapter PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${TensorRT_INCLUDE_DIRS}
    )

    target_link_libraries(xllm_tensorrt_adapter PUBLIC
        xllm_inference
        xllm_operators
        xllm_core
        CUDA::cudart
        ${TensorRT_LIBRARIES}
    )

    target_compile_definitions(xllm_tensorrt_adapter PUBLIC
        XLLM_HAS_TENSORRT=1
    )

    add_executable(test_tensorrt_integration
        tests/test_tensorrt_integration.c
    )
    target_include_directories(test_tensorrt_integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )
    target_compile_definitions(test_tensorrt_integration PRIVATE
        XLLM_HAS_TENSORRT=1
    )
    target_link_libraries(test_tensorrt_integration
        xllm_tensorrt_adapter
    )
    add_test(NAME tensorrt_integration_tests COMMAND test_tensorrt_integration)
else()
    add_executable(test_tensorrt_integration
        tests/test_tensorrt_integration.c
        src/adapter/tensorrt_adapter.c
    )
    target_include_directories(test_tensorrt_integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )
    target_link_libraries(test_tensorrt_integration
        xllm_core
        m
    )
    add_test(NAME tensorrt_integration_tests COMMAND test_tensorrt_integration)
endif()

# ── Examples ────────────────────────────────────────────────────────────
add_executable(run_gpt2 examples/run_gpt2.c)
target_link_libraries(run_gpt2 xllm_inference m)

add_executable(load_model examples/load_model.c)
target_link_libraries(load_model xllm_core m)

add_executable(text_gen examples/text_gen.c)
target_link_libraries(text_gen xllm_core xllm_inference m)
