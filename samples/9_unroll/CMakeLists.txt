# unroll

set(SOURCES unroll.cpp)
set_source_files_properties( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("unroll" ${SOURCES})
add_dependencies("unroll" "kernellib_bc")

target_link_libraries("unroll" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("unroll" PROPERTIES CXX_STANDARD_REQUIRED ON)


install(TARGETS "unroll"
        RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")
