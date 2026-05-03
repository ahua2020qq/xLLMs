cmake_minimum_required(VERSION 3.16)
project(nxtLLM VERSION 0.2.0 LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")

# ── Find CUDA ───────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Library target: nxtllm_core ──────────────────────────────────────
add_library(nxtllm_core STATIC
    src/core/memory_manager.c
    src/core/lru_k.c
)

target_include_directories(nxtllm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# ── Operator library (CUDA) ──────────────────────────────────────────
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

# ── Test executables ──────────────────────────────────────────────────
enable_testing()

add_executable(test_page_pool
    tests/test_page_pool.c
)
target_link_libraries(test_page_pool nxtllm_core)
add_test(NAME page_pool_tests COMMAND test_page_pool)

add_executable(test_attention
    tests/test_attention.c
)
target_link_libraries(test_attention nxtllm_operators)
add_test(NAME attention_tests COMMAND test_attention)
