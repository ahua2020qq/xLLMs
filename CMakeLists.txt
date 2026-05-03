option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
    cmake_minimum_required(VERSION 3.16)
    project(nxtLLM VERSION 0.3.0 LANGUAGES C CXX CUDA)
else()
    cmake_minimum_required(VERSION 3.16)
    project(nxtLLM VERSION 0.3.0 LANGUAGES C CXX)
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

# ── Operator library (CUDA) ──────────────────────────────────────────
if(USE_CUDA)
    add_library(nxtllm_operators STATIC
        operators/page_attention.cu
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

# ── Example: run_gpt2 ─────────────────────────────────────────────────
add_executable(run_gpt2 examples/run_gpt2.c)

target_link_libraries(run_gpt2
    nxtllm_inference
    m
)
