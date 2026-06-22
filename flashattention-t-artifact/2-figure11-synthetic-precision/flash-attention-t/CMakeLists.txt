cmake_minimum_required(VERSION 3.26)


# check if ccache exists
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  # use ccache
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
endif()

# check if mold exists
find_program(MOLD_FOUND mold)
if(MOLD_FOUND)
  # use mold linker
  set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=mold")
  set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=mold")
endif()


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# fucking stupid pytorch relies on its custom VAR to specify CUDA architectures
# and it completely ignores CMAKE_CUDA_ARCHITECTURES

# the "TORCH_TORCH_CUDA_ARCH_LIST" is meant to choose CUDA arch like the below
# ``TORCH_CUDA_ARCH_LIST="6.1 8.6" python build_my_extension.py``
# ``TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python build_my_extension.py``

# we implement our own mapping logic

# set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures to build for" FORCE)
# set(TORCH_CUDA_ARCH_LIST "8.0")

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "Please set CMAKE_CUDA_ARCHITECTURES")
endif()

# check the CMAKE_CUDA_ARCHITECTURES is not a list
if (CMAKE_CUDA_ARCHITECTURES MATCHES ";")
  message(FATAL_ERROR "Please set CMAKE_CUDA_ARCHITECTURES as a single value")
endif()

# convert CMAKE_CUDA_ARCHITECTURES to TORCH_CUDA_ARCH_LIST
set(TORCH_CUDA_ARCH_LIST "")
if (CMAKE_CUDA_ARCHITECTURES STREQUAL "80")
  set(TORCH_CUDA_ARCH_LIST "8.0")
elseif (CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
  set(TORCH_CUDA_ARCH_LIST "8.6")
elseif (CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  set(TORCH_CUDA_ARCH_LIST "8.9")
elseif (CMAKE_CUDA_ARCHITECTURES STREQUAL "90")
  set(TORCH_CUDA_ARCH_LIST "9.0")
elseif (CMAKE_CUDA_ARCHITECTURES STREQUAL "90a")
  set(TORCH_CUDA_ARCH_LIST "9.0a")
else()
  message(FATAL_ERROR "Unsupported CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

project(flash-attn-cmake LANGUAGES CXX CUDA)

set(VENV_NAME ".venv" CACHE STRING "Name of the virtual environment")

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${VENV_NAME}/lib/python3.11/site-packages/torch/share/cmake/Torch")
find_package(Torch CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${VENV_NAME}/lib/python3.11/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 CONFIG REQUIRED)
set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include)

set(CSRC_FLASH_ATTN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn)
set(CSRC_FLASH_ATTN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src)

file(GLOB_RECURSE CSRC_FLASH_ATTN_CUDA_SOURCES ${CSRC_FLASH_ATTN_SRC_DIR}/*.cu)
message(STATUS "CUDA sources: ${CSRC_FLASH_ATTN_CUDA_SOURCES}")

function(setup_individual_cuda_target target_name)
  target_compile_options(${target_name} PRIVATE --expt-relaxed-constexpr)
  target_compile_options(${target_name} PRIVATE --expt-extended-lambda)
  target_compile_options(${target_name} PRIVATE --use_fast_math)
  target_include_directories(${target_name} PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(${target_name} PRIVATE ${CUTLASS_INCLUDE_DIR})
  target_include_directories(${target_name} PRIVATE ${CSRC_FLASH_ATTN_SRC_DIR})
endfunction()

list(APPEND CU_FILE_INDIVIDUAL_TARGETS)

foreach(cuda_source ${CSRC_FLASH_ATTN_CUDA_SOURCES})

  # workaround #1
  # ignore any file that contains "224" inside their name
  if (${cuda_source} MATCHES "224")
      message(STATUS "Ignoring file: ${cuda_source}")
      continue()
  endif()

  get_filename_component(cuda_source_name ${cuda_source} NAME)
  string(REPLACE ".cu" "" target_name ${cuda_source_name})
  message(STATUS "CUDA source: ${cuda_source_name}")
  message(STATUS "CUDA target: ${target_name}")
  add_library(${target_name} SHARED ${cuda_source})
  setup_individual_cuda_target(${target_name})
  list(APPEND CU_FILE_INDIVIDUAL_TARGETS ${target_name})
endforeach()

set(MODULE_NAME flash_attn_2_cuda)

pybind11_add_module(${MODULE_NAME} ${CSRC_FLASH_ATTN_DIR}/flash_api.cpp)
target_link_libraries(${MODULE_NAME} PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(${MODULE_NAME} PUBLIC ${CU_FILE_INDIVIDUAL_TARGETS})
target_include_directories(${MODULE_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(${MODULE_NAME} PRIVATE ${CUTLASS_INCLUDE_DIR})
target_include_directories(${MODULE_NAME} PRIVATE ${CSRC_FLASH_ATTN_SRC_DIR})

target_link_libraries(${MODULE_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/${VENV_NAME}/lib/python3.11/site-packages/torch/lib/libtorch_python.so")
target_compile_definitions(${MODULE_NAME} PRIVATE TORCH_EXTENSION_NAME=${MODULE_NAME})