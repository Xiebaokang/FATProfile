locate_file_dir(${VENV_NAME} VENV_CONTAINING_DIR)
locate_file_dir(csrc FLASH_ATTN_ROOT_DIR)
locate_file_dir(custom_api CUSTOM_API_CONTAINING_DIR)
message(STATUS "Found ${VENV_NAME} in ${VENV_CONTAINING_DIR}")

list(APPEND PYTHON_VER_CANDIDATES 3.11 3.12)
foreach(PYTHON_VER ${PYTHON_VER_CANDIDATES})
  set(VENV_PYTHON_LIB_DIR ${VENV_CONTAINING_DIR}/${VENV_NAME}/lib/python${PYTHON_VER})
  if (EXISTS ${VENV_PYTHON_LIB_DIR})
    set(PYTHON_LIB_DIR ${VENV_PYTHON_LIB_DIR})
    message(STATUS "Found Python lib dir: ${PYTHON_LIB_DIR}")
  endif()
endforeach()
set(USE_SYSTEM_NVTX ON)
set(Torch_DIR "${PYTHON_LIB_DIR}/site-packages/torch/share/cmake/Torch")
find_package(Torch CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

set(CUTLASS_INCLUDE_DIR ${FLASH_ATTN_ROOT_DIR}/csrc/cutlass/include)
set(CSRC_FLASH_ATTN_DIR ${FLASH_ATTN_ROOT_DIR}/csrc/flash_attn)
set(CSRC_FLASH_ATTN_SRC_DIR ${FLASH_ATTN_ROOT_DIR}/csrc/flash_attn/src)

set(CUSTOM_API_DIR ${CUSTOM_API_CONTAINING_DIR}/custom_api)

assert_dir_exists(${CUTLASS_INCLUDE_DIR})
assert_dir_exists(${CSRC_FLASH_ATTN_DIR})
assert_dir_exists(${CSRC_FLASH_ATTN_SRC_DIR})


# DGX MIG detection logic
set(IS_DGX_MIG OFF)

enable_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
  # check if GPU is MIG of DGX
  execute_process(
    COMMAND nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader
    OUTPUT_VARIABLE MIG_MODE
    OUTPUT_STRIP_TRAILING_WHITESPACE  
  )
  if (MIG_MODE STREQUAL "Enabled")
    message(STATUS "MIG mode is enabled, compiling on DGX MIG A100")
    set(IS_DGX_MIG ON)
  endif()
else()
  message(FATAL_ERROR "CUDA compiler not found")
endif()

function(add_single_source_executable SOURCEFILE)
  # target name is the source file name without extension
  get_filename_component(TARGET_NAME ${SOURCEFILE} NAME_WE)
  add_executable(${TARGET_NAME} ${SOURCEFILE})
  target_compile_options(${TARGET_NAME} PRIVATE --expt-relaxed-constexpr)
  target_compile_options(${TARGET_NAME} PRIVATE --expt-extended-lambda)
  target_compile_options(${TARGET_NAME} PRIVATE --use_fast_math)
  target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(${TARGET_NAME} PRIVATE ${CUTLASS_INCLUDE_DIR})
  target_include_directories(${TARGET_NAME} PRIVATE ${CSRC_FLASH_ATTN_SRC_DIR})
  target_include_directories(${TARGET_NAME} PRIVATE ${CUSTOM_API_DIR})

  # define a macro IS_DGX_MIG=1 if IS_DGX_MIG is ON
  # We do NOT need to write "-DIS_DGX_MIG=1" in the following cmake command
  if (IS_DGX_MIG)
    target_compile_definitions(${TARGET_NAME} PRIVATE IS_DGX_MIG=1)
    #also suppress warning on macro re-definition 
  else()
    target_compile_definitions(${TARGET_NAME} PRIVATE IS_DGX_MIG=0)
  endif()

endfunction()
  