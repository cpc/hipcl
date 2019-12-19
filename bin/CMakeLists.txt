
set_source_files_properties(hipcl_config.cc PROPERTIES LANGUAGE CXX )

add_executable(hipcl_config hipcl_config.cc)

set_target_properties(hipcl_config PROPERTIES CXX_STANDARD_REQUIRED ON)

target_include_directories(hipcl_config PRIVATE "${CMAKE_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/cxxopts/include")

target_link_libraries(hipcl_config ${PTHREAD_LIBRARY})

install(TARGETS "hipcl_config"
        RUNTIME DESTINATION "${HIPCL_BIN_DIR}")
