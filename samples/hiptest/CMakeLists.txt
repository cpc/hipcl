# hiptest

set(HIPTEST_SOURCES hiptest.cc)
set_source_files_properties( ${HIPTEST_SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("hiptest" ${HIPTEST_SOURCES})
add_dependencies("hiptest" "kernellib_bc")

target_link_libraries("hiptest" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("hiptest" PROPERTIES CXX_STANDARD_REQUIRED ON)

install(TARGETS "hiptest"
        RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

add_test(NAME "hiptest"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hiptest"
         PASS_REGULAR_EXPRESSION "PASSED")
