set(HIPMATH_SOURCES hipmath.cc)
set_source_files_properties( ${HIPMATH_SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("hipmath" ${HIPMATH_SOURCES})
add_dependencies("hipmath" "kernellib_bc")

target_link_libraries("hipmath" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("hipmath" PROPERTIES CXX_STANDARD_REQUIRED ON)


install(TARGETS "hipmath"
        RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

add_test(NAME "hipmath"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hipmath"
         PASS_REGULAR_EXPRESSION "PASSED")
