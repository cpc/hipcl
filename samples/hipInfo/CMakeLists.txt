set(SOURCE hipInfo.cpp)

set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)

add_executable(hipInfo ${SOURCE})

set_target_properties(hipInfo PROPERTIES CXX_STANDARD_REQUIRED ON)

add_dependencies(hipInfo "kernellib_bc")

target_link_libraries(hipInfo "${SANITIZER_LIBS}" "hipcl")

install(TARGETS hipInfo
        RUNTIME DESTINATION "${HIPCL_BIN_DIR}")
