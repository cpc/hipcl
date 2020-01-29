###############################################################################
# FindHIP.cmake
###############################################################################

###############################################################################
# SET: Variable defaults
###############################################################################
# User defined flags
set(HIP_HIPCC_FLAGS "" CACHE STRING "Semicolon delimited flags for HIPCC")
set(HIP_HCC_FLAGS "" CACHE STRING "Semicolon delimited flags for HCC")
set(HIP_NVCC_FLAGS "" CACHE STRING "Semicolon delimted flags for NVCC")
set(HIP_HIPCL_FLAGS "" CACHE STRING "Semicolon delimted flags for HIPCL")
mark_as_advanced(HIP_HIPCC_FLAGS HIP_HCC_FLAGS HIP_NVCC_FLAGS HIP_HIPCL_FLAGS)

set(_hip_configuration_types ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} Debug MinSizeRel Release RelWithDebInfo)
list(REMOVE_DUPLICATES _hip_configuration_types)

foreach(config ${_hip_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(HIP_HIPCC_FLAGS_${config_upper} "" CACHE STRING "Semicolon delimited flags for HIPCC")
    set(HIP_HCC_FLAGS_${config_upper} "" CACHE STRING "Semicolon delimited flags for HCC")
    set(HIP_NVCC_FLAGS_${config_upper} "" CACHE STRING "Semicolon delimited flags for NVCC")
    set(HIP_HIPCL_FLAGS_${config_upper} "" CACHE STRING "Semicolon delimted flags for HIPCL")
    mark_as_advanced(HIP_HIPCC_FLAGS_${config_upper} HIP_HCC_FLAGS_${config_upper} HIP_NVCC_FLAGS_${config_upper} HIP_HIPCL_FLAGS_${config_upper})
endforeach()

option(HIP_HOST_COMPILATION_CPP "Host code compilation mode" ON)
option(HIP_VERBOSE_BUILD "Print out the commands run while compiling the HIP source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can be forced on with this option." OFF)
mark_as_advanced(HIP_HOST_COMPILATION_CPP)

###############################################################################
# Set HIP CMAKE Flags
###############################################################################

# Copy the invocation styles from CXX to HIP
set(CMAKE_HIP_ARCHIVE_CREATE ${CMAKE_CXX_ARCHIVE_CREATE})
set(CMAKE_HIP_ARCHIVE_APPEND ${CMAKE_CXX_ARCHIVE_APPEND})
set(CMAKE_HIP_ARCHIVE_FINISH ${CMAKE_CXX_ARCHIVE_FINISH})
set(CMAKE_SHARED_LIBRARY_SONAME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG})
set(CMAKE_SHARED_LIBRARY_CREATE_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS})
set(CMAKE_SHARED_LIBRARY_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_CXX_FLAGS})
#set(CMAKE_SHARED_LIBRARY_LINK_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS})
set(CMAKE_SHARED_LIBRARY_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
set(CMAKE_SHARED_LIBRARY_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
set(CMAKE_SHARED_LIBRARY_LINK_STATIC_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_STATIC_CXX_FLAGS})
set(CMAKE_SHARED_LIBRARY_LINK_DYNAMIC_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_DYNAMIC_CXX_FLAGS})

# Set the CMake Flags to use the HCC Compilier.
set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CMAKE_HIP_CREATE_SHARED_MODULE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <LINK_LIBRARIES> -shared" )
set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

###############################################################################
# FIND: HIP and associated helper binaries
###############################################################################
# HIP is supported on Linux only
if(UNIX AND NOT APPLE AND NOT CYGWIN)
    # Search for HIP installation
    if(NOT HIP_ROOT_DIR)
        # Search in user specified path first
        find_path(
            HIP_ROOT_DIR
            NAMES hipcl_config
            PATHS
            ENV HIP_PATH
            PATH_SUFFIXES bin
            DOC "HIPCL installed location"
            NO_DEFAULT_PATH
            )
        # Now search in default path
        find_path(
            HIP_ROOT_DIR
            NAMES hipcl_config
            PATHS
            /opt/hipcl
            PATH_SUFFIXES bin
            DOC "HIPCL installed location"
            )

        # Check if we found HIP installation
        if(HIP_ROOT_DIR)
            # If so, fix the path
            string(REGEX REPLACE "[/\\\\]?bin[64]*[/\\\\]?$" "" HIP_ROOT_DIR ${HIP_ROOT_DIR})
            # And push it back to the cache
            set(HIP_ROOT_DIR ${HIP_ROOT_DIR} CACHE PATH "HIP installed location" FORCE)
        endif()
        if(NOT EXISTS ${HIP_ROOT_DIR})
            if(HIP_FIND_REQUIRED)
                message(FATAL_ERROR "Specify HIP_ROOT_DIR")
            elseif(NOT HIP_FIND_QUIETLY)
                message("HIP_ROOT_DIR not found or specified")
            endif()
        endif()
    endif()

    # Find HIPCC (clang++) executable
    find_program(
        HIP_HIPCC_EXECUTABLE
        NAMES clang++
        PATHS
        "${HIP_ROOT_DIR}/llvm"
        ENV HIP_PATH
        /opt/hipcl/llvm
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
        )
    mark_as_advanced(HIP_HIPCC_EXECUTABLE)

    # Find HIPCL_CONFIG executable
    find_program(
        HIP_HIPCONFIG_EXECUTABLE
        NAMES hipcl_config
        PATHS
        "${HIP_ROOT_DIR}"
        ENV HIP_PATH
        /opt/hipcl
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
        )
    mark_as_advanced(HIP_HIPCONFIG_EXECUTABLE)

    if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_VERSION)
        # Compute the version
        execute_process(
            COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --version
            OUTPUT_VARIABLE _hip_version
            ERROR_VARIABLE _hip_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
            )
        if(NOT _hip_error)
            set(HIP_VERSION ${_hip_version} CACHE STRING "Version of HIP as computed from hipcc")
        else()
            set(HIP_VERSION "0.0.0" CACHE STRING "Version of HIP as computed by FindHIP()")
        endif()
        mark_as_advanced(HIP_VERSION)
    endif()
    if(HIP_VERSION)
        string(REPLACE "." ";" _hip_version_list "${HIP_VERSION}")
        list(GET _hip_version_list 0 HIP_VERSION_MAJOR)
        list(GET _hip_version_list 1 HIP_VERSION_MINOR)
        list(GET _hip_version_list 2 HIP_VERSION_PATCH)
        set(HIP_VERSION_STRING "${HIP_VERSION}")
    endif()

    if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_PLATFORM)
        message(STATUS "#### Running ${HIP_HIPCONFIG_EXECUTABLE} to find platform!")
        # Compute the platform
        execute_process(
            COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --platform
            OUTPUT_VARIABLE _hip_platform
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        set(HIP_PLATFORM ${_hip_platform} CACHE STRING "HIP platform as computed by hipconfig")
        mark_as_advanced(HIP_PLATFORM)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    HIP
    REQUIRED_VARS
    HIP_ROOT_DIR
    HIP_HIPCC_EXECUTABLE
    HIP_HIPCONFIG_EXECUTABLE
    HIP_PLATFORM
    VERSION_VAR HIP_VERSION
    )


include(hip-targets)

# vim: ts=4:sw=4:expandtab:smartindent
