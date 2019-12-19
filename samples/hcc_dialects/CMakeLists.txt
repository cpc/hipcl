# vadd_hip

set(SOURCES vadd_hip.cpp)
set_source_files_properties( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("vadd_hip" ${SOURCES})
add_dependencies("vadd_hip" "kernellib_bc")

target_link_libraries("vadd_hip" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("vadd_hip" PROPERTIES CXX_STANDARD_REQUIRED ON)


install(TARGETS "vadd_hip"
        RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")
