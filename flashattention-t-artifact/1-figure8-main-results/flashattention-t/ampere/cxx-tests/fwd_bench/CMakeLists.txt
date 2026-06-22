cmake_minimum_required(VERSION 3.21)

set(VENV_NAME ".venv" CACHE STRING "Name of the virtual environment")
set(PROJ_NAME "time_test")
set(CMAKE_UTILS_DIR "${CMAKE_CURRENT_LIST_DIR}/../cmake")

############################################################
############### DO NOT EDIT BELOW THIS LINE ################

# include the prologue file BEFORE calling project() !!!
include(${CMAKE_UTILS_DIR}/cmakePrologue.cmake)
include(${CMAKE_UTILS_DIR}/utilFuncs.cmake)
project(${PROJ_NAME} LANGUAGES CXX CUDA)
#
# flashVenv.cmake provides "add_single_source_executable" function
#
include(${CMAKE_UTILS_DIR}/flashVenv.cmake)

############### DO NOT EDIT ABOVE THIS LINE ################
############################################################

### csv2

include(FetchContent)
FetchContent_Declare(
    csv2
    GIT_REPOSITORY https://github.com/p-ranav/csv2.git
    GIT_TAG v0.1
    # GIT_CONFIG "http.proxy=127.0.0.1:7890" "https.proxy=127.0.0.1:7890"
)
FetchContent_MakeAvailable(csv2)

function(link_to_csv2 TARGET_NAME)
  target_include_directories(${TARGET_NAME} PRIVATE ${csv2_SOURCE_DIR}/include)
  target_link_libraries(${TARGET_NAME} PRIVATE csv2::csv2)
endfunction()

### argparse

FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.1
    # GIT_CONFIG "http.proxy=127.0.0.1:7890" "https.proxy=127.0.0.1:7890"
)
FetchContent_MakeAvailable(argparse)

function(link_to_argparse TARGET_NAME)
  target_include_directories(${TARGET_NAME} PRIVATE ${argparse_SOURCE_DIR}/include)
  target_link_libraries(${TARGET_NAME} PRIVATE argparse::argparse)
endfunction()


add_single_source_executable(fp16-fwd-bench-orig.cu)
link_to_csv2(fp16-fwd-bench-orig)
link_to_argparse(fp16-fwd-bench-orig)


add_executable(
  fp16-fwd-bench
  fp16-fwd-bench.cu
  benches/hdim64-noncausal.cu
  benches/hdim64-causal.cu
  benches/hdim128-noncausal.cu
  benches/hdim128-causal.cu
)
target_compile_options(fp16-fwd-bench PRIVATE --expt-relaxed-constexpr)
target_compile_options(fp16-fwd-bench PRIVATE --expt-extended-lambda)
target_compile_options(fp16-fwd-bench PRIVATE --use_fast_math)
target_include_directories(fp16-fwd-bench PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(fp16-fwd-bench PRIVATE ${TORCH_LIBRARIES})
target_include_directories(fp16-fwd-bench PRIVATE ${CUTLASS_INCLUDE_DIR})
target_include_directories(fp16-fwd-bench PRIVATE ${CSRC_FLASH_ATTN_SRC_DIR})
target_include_directories(fp16-fwd-bench PRIVATE ${CUSTOM_API_DIR})
link_to_csv2(fp16-fwd-bench)
link_to_argparse(fp16-fwd-bench)
