# bit_extract

set(SOURCES bit_extract.cpp)
set_source_files_properties( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_executable("bit_extract" ${SOURCES})
add_dependencies("bit_extract" "kernellib_bc")

target_link_libraries("bit_extract" ${PTHREAD_LIBRARY} "hipcl" ${OpenCL_LIBRARIES})

set_target_properties("bit_extract" PROPERTIES CXX_STANDARD_REQUIRED ON)


install(TARGETS "bit_extract"
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
