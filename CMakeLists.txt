
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

cmake_policy(VERSION 3.3)

project(HIP_CL
        VERSION 0.1
        DESCRIPTION "Hip implementation over OpenCL"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11 )
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()


if(UNIX AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  message("Using Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
    message(FATAL_ERROR "This project requires Clang >= 8.0")
  endif()
else()
  message(FATAL_ERROR "This project must be compiled with Clang")
endif()

set(FORBIT_IMPLICIT_FUNCTIONS "-Werror=implicit-function-declaration;-Werror=incompatible-pointer-types")
add_compile_options(${FORBIT_IMPLICIT_FUNCTIONS})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(DEFAULT_BUILD_TYPE "Debug")
else()
  set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
endif()


set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(PTHREAD_LIBRARY Threads::Threads)

add_subdirectory(lib)

option(SAVE_TEMPS "Save temporary compilation products" OFF)

option(VERBOSE "Verbose compilation" OFF)

if(SAVE_TEMPS)
  add_compile_options("--save-temps")
endif()

if(VERBOSE)
  add_compile_options("-v")
endif()

if(DEBUG)
  add_compile_options("-D_GLIBCXX_DEBUG")
endif()

add_compile_options("-x" "hip")
add_compile_options("--target=x86_64-linux-gnu")

add_compile_options("--hip-device-lib-path=${CMAKE_BINARY_DIR}" "--hip-device-lib=kernellib.bc")

include_directories("${CMAKE_SOURCE_DIR}/lib")

add_subdirectory(samples)
