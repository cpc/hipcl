
# stream

set(SOURCES stream.cpp)
set_source_files_properties( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("stream" ${SOURCES})
add_dependencies("stream" "kernellib_bc")

target_link_libraries("stream" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("stream" PROPERTIES CXX_STANDARD_REQUIRED ON)

install(TARGETS "stream"
        RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")
