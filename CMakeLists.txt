option(USE_CUDA "Enable CUDA support" OFF)
option(USE_FLASHINFER "Enable FlashInfer-style attention adapter (requires USE_CUDA)" OFF)
option(USE_TENSORRT "Enable TensorRT-LLM integration (requires USE_CUDA)" OFF)

if(USE_CUDA)
    cmake_minimum_required(VERSION 3.16)
    project(nxtLLM VERSION 0.5.0 LANGUAGES C CXX CUDA)
else()
    cmake_minimum_required(VERSION 3.16)
    project(nxtLLM VERSION 0.5.0 LANGUAGES C CXX)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")

# ── Find CUDA ───────────────────────────────────────────────────────
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ── Library target: nxtllm_core ──────────────────────────────────────
add_library(nxtllm_core STATIC
    src/core/memory_manager.c
    src/core/lru_k.c
    src/core/prefix_sharing.c
)

target_include_directories(nxtllm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(Threads REQUIRED)
target_link_libraries(nxtllm_core PUBLIC Threads::Threads)

# ── Operator library (CUDA) ──────────────────────────────────────────
if(USE_CUDA)
    add_library(nxtllm_operators STATIC
        operators/page_attention.cu
        operators/page_attention_v2.cu
        operators/activation_kernels.cu
    )

    target_include_directories(nxtllm_operators PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )

    target_link_libraries(nxtllm_operators PUBLIC
        CUDA::cudart
        nxtllm_core
    )

    set_target_properties(nxtllm_operators PROPERTIES
        CUDA_ARCHITECTURES "75;80;86;89;90"
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ── Test executables ──────────────────────────────────────────────────
enable_testing()

add_executable(test_page_pool
    tests/test_page_pool.c
)
target_link_libraries(test_page_pool nxtllm_core)
add_test(NAME page_pool_tests COMMAND test_page_pool)

add_executable(test_prefix_sharing
    tests/test_prefix_sharing.c
)
target_link_libraries(test_prefix_sharing nxtllm_core)
add_test(NAME prefix_sharing_tests COMMAND test_prefix_sharing)

if(USE_CUDA)
    math(EXPR NXTLLM_CUDART_VERSION
        "${CUDAToolkit_VERSION_MAJOR} * 1000 + ${CUDAToolkit_VERSION_MINOR} * 10")

    add_executable(test_attention
        tests/test_attention.c
    )
    target_include_directories(test_attention PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_definitions(test_attention PRIVATE
        CUDART_VERSION=${NXTLLM_CUDART_VERSION}
    )
    target_link_libraries(test_attention
        nxtllm_operators
        CUDA::cudart
    )
    add_test(NAME attention_tests COMMAND test_attention)
endif()

# ── Inference library ─────────────────────────────────────────────────
set(INFERENCE_SOURCES
    src/inference/transformer_block.c
    src/inference/tokenizer.c
    src/inference/decode_loop.c
)

add_library(nxtllm_inference STATIC ${INFERENCE_SOURCES})

target_include_directories(nxtllm_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_link_libraries(nxtllm_inference PUBLIC
    nxtllm_core
    m
)

# ── FlashInfer-style adapter (optional) ────────────────────────────────
if(USE_FLASHINFER)
    if(NOT USE_CUDA)
        message(FATAL_ERROR "USE_FLASHINFER requires USE_CUDA=ON")
    endif()

    add_library(nxtllm_flash_adapter STATIC
        operators/flashinfer_adapter.cu
    )

    target_include_directories(nxtllm_flash_adapter PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )

    target_link_libraries(nxtllm_flash_adapter PUBLIC
        CUDA::cudart
        nxtllm_operators
    )

    set_target_properties(nxtllm_flash_adapter PROPERTIES
        CUDA_ARCHITECTURES "80;86;89;90"
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_definitions(nxtllm_flash_adapter PRIVATE
        USE_FLASHINFER=1
    )

    add_executable(test_attention_bench
        tests/test_attention_bench.c
    )
    target_include_directories(test_attention_bench PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_definitions(test_attention_bench PRIVATE
        CUDART_VERSION=${NXTLLM_CUDART_VERSION}
    )
    target_link_libraries(test_attention_bench
        nxtllm_flash_adapter
        nxtllm_operators
        CUDA::cudart
    )
    add_test(NAME attention_bench_tests COMMAND test_attention_bench)
endif()

# ── TensorRT-LLM adapter (optional) ────────────────────────────────────
if(USE_TENSORRT)
    if(NOT USE_CUDA)
        message(FATAL_ERROR "USE_TENSORRT requires USE_CUDA=ON")
    endif()

    find_package(TensorRT REQUIRED)

    # TensorRT adapter library (stub implementation when no TRT found)
    add_library(nxtllm_tensorrt_adapter STATIC
        src/adapter/tensorrt_adapter.c
    )

    target_include_directories(nxtllm_tensorrt_adapter PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${TensorRT_INCLUDE_DIRS}
    )

    target_link_libraries(nxtllm_tensorrt_adapter PUBLIC
        nxtllm_inference
        nxtllm_operators
        nxtllm_core
        CUDA::cudart
        ${TensorRT_LIBRARIES}
    )

    target_compile_definitions(nxtllm_tensorrt_adapter PUBLIC
        NXTLLM_HAS_TENSORRT=1
    )

    # TensorRT integration test
    add_executable(test_tensorrt_integration
        tests/test_tensorrt_integration.c
    )

    target_include_directories(test_tensorrt_integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )

    target_compile_definitions(test_tensorrt_integration PRIVATE
        NXTLLM_HAS_TENSORRT=1
    )

    target_link_libraries(test_tensorrt_integration
        nxtllm_tensorrt_adapter
    )

    add_test(NAME tensorrt_integration_tests COMMAND test_tensorrt_integration)
else()
    # Stub mode: compile adapter stub directly with test (no TensorRT needed)
    add_executable(test_tensorrt_integration
        tests/test_tensorrt_integration.c
        src/adapter/tensorrt_adapter.c
    )

    target_include_directories(test_tensorrt_integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )

    target_link_libraries(test_tensorrt_integration
        nxtllm_core
        m
    )

    add_test(NAME tensorrt_integration_tests COMMAND test_tensorrt_integration)
endif()

# ── Example: run_gpt2 ─────────────────────────────────────────────────
add_executable(run_gpt2 examples/run_gpt2.c)

target_link_libraries(run_gpt2
    nxtllm_inference
    m
)
