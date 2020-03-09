add_subdirectory(bitcode)

###################################################################################

find_package(OpenCL 2.0 REQUIRED)

set(HIPCL_SOURCES hipcl.cc backend.cc log.cc spirv.cc)

set_source_files_properties(${HIPCL_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(hipcl SHARED ${HIPCL_SOURCES})

set_target_properties(hipcl PROPERTIES
                      CXX_STANDARD_REQUIRED ON
                      SOVERSION "${LIB_API_VERSION}"
                      VERSION "${LIB_BUILD_VERSION}")

target_link_libraries(hipcl ${SANITIZER_LIBS} ${PTHREAD_LIBRARY} ${OpenCL_LIBRARIES})

add_dependencies("hipcl" "kernellib_bc")

if(DEBUG)
    target_compile_definitions(hipcl PRIVATE "_GLIBCXX_DEBUG")
endif()

if(LOGLEVEL)
  set(VALID_LEVELS "DEBUG;INFO;WARN;ERROR;CRITICAL;OFF")
  if(LOGLEVEL IN_LIST VALID_LEVELS)
    target_compile_definitions(hipcl PRIVATE "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LOGLEVEL}")
  else()
    message(WARNING "Unknown loglevel: ${LOGLEVEL}, ignoring")
  endif()
endif()

target_compile_options(hipcl PRIVATE "-Wno-unused-parameter")

target_compile_options(hipcl INTERFACE "-x" "hip")

if(SANITIZER_OPTIONS)
  target_compile_options(hipcl INTERFACE ${SANITIZER_OPTIONS})
  target_compile_options(hipcl PRIVATE ${SANITIZER_OPTIONS})
endif()

target_compile_options(hipcl INTERFACE
        "$<INSTALL_INTERFACE:--hip-device-lib-path=${HIPCL_DATA_DIR}>"
        "$<BUILD_INTERFACE:--hip-device-lib-path=${CMAKE_BINARY_DIR}>"
        "--hip-device-lib=kernellib.bc")

# for LLVM passes
target_compile_options(hipcl INTERFACE
        "$<INSTALL_INTERFACE:--hip-llvm-pass-path=${HIPCL_LLVM_DIR}>"
        "$<BUILD_INTERFACE:--hip-llvm-pass-path=${CMAKE_BINARY_DIR}/llvm_passes>")

target_include_directories(hipcl
    PUBLIC
        "$<INSTALL_INTERFACE:${HIPCL_INC_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
    PRIVATE
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/include"
)

install(TARGETS hipcl
        EXPORT "hip-targets"
        LIBRARY DESTINATION "${HIPCL_LIB_DIR}"
        ARCHIVE DESTINATION "${HIPCL_LIB_DIR}"
        )

install(EXPORT "hip-targets"
        NAMESPACE "hip::"
        DESTINATION "${HIPCL_CMAKE_DIR}")

install(FILES
        "${CMAKE_SOURCE_DIR}/include/hip/hipcl.hh"
        "${CMAKE_SOURCE_DIR}/include/hip/hipcl_mathlib.hh"
        "${CMAKE_SOURCE_DIR}/include/hip/hip_fatbin.h"
        "${CMAKE_SOURCE_DIR}/include/hip/hip_fp16.h"
        "${CMAKE_SOURCE_DIR}/include/hip/hip_runtime.h"
        "${CMAKE_SOURCE_DIR}/include/hip/hip_vector_types.h"
  DESTINATION "${HIPCL_INC_DIR}/hip")

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/hipcl-config.cmake.in"
  "${CMAKE_BINARY_DIR}/hip-config.cmake"
  INSTALL_DESTINATION "${HIPCL_CMAKE_DIR}"
  PATH_VARS HIPCL_INC_DIR HIPCL_LIB_DIR HIPCL_BIN_DIR HIPCL_COMPILER
  )

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/hipConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
  )
