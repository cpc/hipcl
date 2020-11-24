# LLVM requires CMake >= 3.4.3
cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
cmake_policy(VERSION 3.4)

set(HIPCL_VERSION_MAJOR 0)
set(HIPCL_VERSION_MINOR 9)
set(HIPCL_VERSION_PATCH 0)
set(HIPCL_VERSION_FULL "${HIPCL_VERSION_MAJOR}.${HIPCL_VERSION_MINOR}.${HIPCL_VERSION_PATCH}")

set(LIB_BUILD_VERSION "${HIPCL_VERSION_FULL}")
set(LIB_API_VERSION "${HIPCL_VERSION_MAJOR}")

project(HIP_CL
        VERSION "${HIPCL_VERSION_FULL}"
        DESCRIPTION "Hip implementation over OpenCL"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_UBSAN "Enable UBSanitizer" OFF)
option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
set(SANITIZER_LIBS)

if(ENABLE_ASAN)
  if("${CMAKE_C_COMPILER_VERSION}" VERSION_LESS "6.0.0")
    list(APPEND SANITIZER_OPTIONS "-fsanitize=address")
  else()
    list(APPEND SANITIZER_OPTIONS "-fsanitize=address" "-fsanitize-recover=address")
  endif()
  list(APPEND SANITIZER_LIBS "asan")
endif()

if(ENABLE_LSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=leak")
  list(APPEND SANITIZER_LIBS "lsan")
endif()

if(ENABLE_TSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=thread")
  list(APPEND SANITIZER_LIBS "tsan")
endif()

if(ENABLE_UBSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=undefined")
  list(APPEND SANITIZER_LIBS "ubsan")
endif()

if(SANITIZER_OPTIONS)
  list(APPEND SANITIZER_OPTIONS "-fno-omit-frame-pointer")
  string(REPLACE ";" " " SANITIZER_OPTIONS_STR "${SANITIZER_OPTIONS}")
endif()

enable_testing()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/opt/hipcl" CACHE PATH "Install path prefix" FORCE)
endif()

message(STATUS "HipCL will be installed to: ${CMAKE_INSTALL_PREFIX}")

set(HIPCL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib" CACHE PATH "where libhipcl will be installed")
set(HIPCL_LLVM_DIR "${CMAKE_INSTALL_PREFIX}/lib/llvm" CACHE PATH "where hip LLVM passes will be installed")
set(HIPCL_DATA_DIR "${CMAKE_INSTALL_PREFIX}/share" CACHE PATH "where kernel library bitcode will be installed")
set(HIPCL_INC_DIR "${CMAKE_INSTALL_PREFIX}/include" CACHE PATH "where kernel hipcl headers will be installed")
set(HIPCL_BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin" CACHE PATH "where hipcl_config will be installed")
set(HIPCL_SAMPLE_BINDIR "${CMAKE_INSTALL_PREFIX}/bin/samples" CACHE PATH "where samples will be installed")
set(HIPCL_CMAKE_DIR "${HIPCL_LIB_DIR}/hipcl/cmake" CACHE PATH "where cmake files will be installed")

if(UNIX AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
    message(FATAL_ERROR "This project requires Clang >= 8.0")
  endif()
else()
  message(FATAL_ERROR "This project must be compiled with Clang")
endif()

message(STATUS "Using Clang: ${CMAKE_CXX_COMPILER}")
set(HIPCL_COMPILER "${CMAKE_CXX_COMPILER}")

get_filename_component(CLANG_BIN_PATH "${CMAKE_CXX_COMPILER}" DIRECTORY)

if(NOT DEFINED LLVM_LINK)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-link")
    set(LLVM_LINK "${CLANG_BIN_PATH}/llvm-link" CACHE PATH "llvm-link")
  else()
    message(FATAL_ERROR "Can't find llvm-link at ${CLANG_BIN_PATH}. Please provide CMake argument -DLLVM_LINK=<path/to/llvm-link>")
  endif()
endif()

message(STATUS "Using llvm-link: ${LLVM_LINK}")

if(NOT DEFINED LLVM_SPIRV)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-spirv")
    set(LLVM_SPIRV "${CLANG_BIN_PATH}/llvm-spirv" CACHE PATH "llvm-spirv")
  else()
    message(FATAL_ERROR "Can't find llvm-spirv at ${CLANG_BIN_PATH}. Please copy llvm-spirv to ${CLANG_BIN_PATH}, Clang expects it there!")
  endif()
endif()

message(STATUS "Using llvm-spirv: ${LLVM_SPIRV}")

if(NOT DEFINED LLVM_CONFIG)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-config")
    set(LLVM_CONFIG "${CLANG_BIN_PATH}/llvm-config" CACHE PATH "llvm-config")
  else()
    message(FATAL_ERROR "Can't find llvm-config at ${CLANG_BIN_PATH}. Please provide CMake argument -DLLVM_CONFIG=<path/to/llvm-config>")
  endif()
endif()

message(STATUS "Using llvm-config: ${LLVM_CONFIG}")


if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(DEFAULT_BUILD_TYPE "Debug")
else()
  set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMake build type not set, setting to ${DEFAULT_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "CMake build type" FORCE)
endif()

message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
set(PTHREAD_LIBRARY Threads::Threads)

##########################################

add_subdirectory(lib)

# hipcl_config
add_subdirectory(bin)

# LLVM pass
add_subdirectory(llvm_passes)

##########################################

get_target_property(HIPCL_CFLAGS_PRIV "hipcl" COMPILE_OPTIONS)
message(STATUS "hipcl private compile options: ${HIPCL_CFLAGS_PRIV}")
get_target_property(HIPCL_IDIR_PRIV "hipcl" INCLUDE_DIRECTORIES)
message(STATUS "hipcl private include directories: ${HIPCL_IDIR_PRIV}")


get_target_property(HIPCL_CFLAGS_IFC "hipcl" INTERFACE_COMPILE_OPTIONS)
message(STATUS "hipcl public compile options: ${HIPCL_CFLAGS_IFC}")
get_target_property(HIPCL_IDIR_IFC "hipcl" INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "hipcl public include directories: ${HIPCL_IDIR_IFC}")

string(REPLACE "${CMAKE_INSTALL_PREFIX}" "-I${CMAKE_INSTALL_PREFIX}" HIPCL_IDIR_IFC "${HIPCL_IDIR_IFC}")
# HIPCL_CXX_OPTIONS is what's reported by hipcl-config.
string(REPLACE ";" " " HIPCL_CXX_OPTIONS "${HIPCL_CFLAGS_IFC};${HIPCL_IDIR_IFC}")

# The hipcl "COMPILE_OPTIONS" target properties contain some generator expressions;
# we have to get rid of those for hipcl_config.h
# this is the way Boost does it (until there's a better way)
# https://gitlab.kitware.com/cmake/cmake/issues/17984
string(REPLACE "$<BUILD_INTERFACE:" "$<0:" HIPCL_CXX_OPTIONS "${HIPCL_CXX_OPTIONS}")
string(REPLACE "$<INSTALL_INTERFACE:" "$<1:" HIPCL_CXX_OPTIONS "${HIPCL_CXX_OPTIONS}")
message(STATUS "HIPCL_CXX_OPTIONS: ${HIPCL_CXX_OPTIONS}")

##########################################

configure_file(
  "${CMAKE_SOURCE_DIR}/hipcl_config.h.in"
  "${CMAKE_BINARY_DIR}/hipcl_config.h.gen"
  @ONLY)

file(GENERATE
     OUTPUT "${CMAKE_BINARY_DIR}/hipcl_config.h"
     INPUT "${CMAKE_BINARY_DIR}/hipcl_config.h.gen")

install(FILES
  "${CMAKE_SOURCE_DIR}/cmake/FindHIP.cmake"
  "${CMAKE_BINARY_DIR}/hipConfigVersion.cmake"
  "${CMAKE_BINARY_DIR}/hip-config.cmake"
  DESTINATION "${HIPCL_CMAKE_DIR}")

##########################################

add_subdirectory(samples)
