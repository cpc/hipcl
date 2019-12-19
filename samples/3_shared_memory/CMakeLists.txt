# sharedMemory

set(SOURCES sharedMemory.cpp)
set_source_files_properties( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("sharedMemory" ${SOURCES})
add_dependencies("sharedMemory" "kernellib_bc")

target_link_libraries("sharedMemory" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("sharedMemory" PROPERTIES CXX_STANDARD_REQUIRED ON)

install(TARGETS "sharedMemory"
        RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

add_test(NAME "sharedMemory_matmul"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/sharedMemory"
         PASS_REGULAR_EXPRESSION "PASSED")
