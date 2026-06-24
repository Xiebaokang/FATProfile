# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-src")
  file(MAKE_DIRECTORY "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-src")
endif()
file(MAKE_DIRECTORY
  "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-build"
  "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-subbuild/csv2-populate-prefix"
  "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-subbuild/csv2-populate-prefix/tmp"
  "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-subbuild/csv2-populate-prefix/src/csv2-populate-stamp"
  "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-subbuild/csv2-populate-prefix/src"
  "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-subbuild/csv2-populate-prefix/src/csv2-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-subbuild/csv2-populate-prefix/src/csv2-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/xiebaokang/projects/cuda/test/microbenchmarks/FATProfile/flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench/build/_deps/csv2-subbuild/csv2-populate-prefix/src/csv2-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
