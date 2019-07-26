add_subdirectory(bitcode)

if(LOGLEVEL)
  set(VALID_LEVELS "DEBUG;INFO;WARN;ERROR;CRITICAL;OFF")
  if(LOGLEVEL IN_LIST VALID_LEVELS)
    add_compile_options("-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LOGLEVEL}")
  else()
    message(WARNING "Unknown loglevel ${LOGLEVEL}")
  endif()
endif()

include_directories("${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(OpenCL 2.0 REQUIRED)

set(HIPCL_SOURCES hipcl.cc backend.cc log.cc spirv.cc)

set_source_files_properties(${HIPCL_SOURCES} PROPERTIES LANGUAGE CXX )

add_library("hipcl" SHARED ${HIPCL_SOURCES})

target_link_libraries("hipcl" ${PTHREAD_LIBRARY} ${OpenCL_LIBRARIES})

set_target_properties("hipcl" PROPERTIES CXX_STANDARD_REQUIRED ON)

install(TARGETS "hipcl"
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/hipcl.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipcl_mathlib.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/hip_fatbin.h"
	DESTINATION "include")
