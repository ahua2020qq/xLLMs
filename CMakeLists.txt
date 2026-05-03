cmake_minimum_required(VERSION 3.16)
project(nxtLLM VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")

# ── Library target: nxtllm_core ──────────────────────────────────────
add_library(nxtllm_core STATIC
    src/core/memory_manager.c
    src/core/lru_k.c
)

target_include_directories(nxtllm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# ── Test executable ──────────────────────────────────────────────────
enable_testing()

add_executable(test_page_pool
    tests/test_page_pool.c
)

target_link_libraries(test_page_pool nxtllm_core)

add_test(NAME page_pool_tests COMMAND test_page_pool)
