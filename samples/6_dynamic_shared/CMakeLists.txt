# dynamic_shared

set(SOURCES dynamic_shared.cpp)
set_source_files_properties( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("dynamic_shared" ${SOURCES})
add_dependencies("dynamic_shared" "kernellib_bc")

target_link_libraries("dynamic_shared" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("dynamic_shared" PROPERTIES CXX_STANDARD_REQUIRED ON)


install(TARGETS "dynamic_shared"
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
