# MatrixTranspose

set(SOURCES MatrixTranspose.cpp)
set_source_files_properties( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("MatrixTranspose" ${SOURCES})
add_dependencies("MatrixTranspose" "kernellib_bc")

target_link_libraries("MatrixTranspose" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("MatrixTranspose" PROPERTIES
                      CXX_STANDARD_REQUIRED ON
                      )

install(TARGETS "MatrixTranspose"
        RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

add_test(NAME "matrix_transpose"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/MatrixTranspose"
         PASS_REGULAR_EXPRESSION "PASSED")
