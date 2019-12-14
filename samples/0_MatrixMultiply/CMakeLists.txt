# MatrixMultiply

set(SOURCES MatrixMultiply.cpp)
set_source_files_properties( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("MatrixMultiply" ${SOURCES})
add_dependencies("MatrixMultiply" "kernellib_bc")

target_link_libraries("MatrixMultiply" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("MatrixMultiply" PROPERTIES
                      CXX_STANDARD_REQUIRED ON
                      )

install(TARGETS "MatrixMultiply"
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

add_test(NAME "matrix_multiply"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/MatrixMultiply"
         PASS_REGULAR_EXPRESSION "PASSED")
