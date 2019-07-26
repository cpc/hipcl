# hipEvent

set(SOURCES hipEvent.cpp)
set_source_files_properties( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("hipEvent" ${SOURCES})
add_dependencies("hipEvent" "kernellib_bc")

target_link_libraries("hipEvent" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("hipEvent" PROPERTIES CXX_STANDARD_REQUIRED ON)

install(TARGETS "hipEvent"
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

add_test(NAME "hipEvent"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hipEvent"
         PASS_REGULAR_EXPRESSION "PASSED")
