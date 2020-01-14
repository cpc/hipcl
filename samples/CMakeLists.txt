
option(SAVE_TEMPS "Save temporary compilation products" OFF)
option(VERBOSE "Verbose compilation" OFF)

if(SAVE_TEMPS)
  add_compile_options("--save-temps")
endif()

if(VERBOSE)
  add_compile_options("-v")
endif()

# ARGN = sources
function(add_hipcl_test EXEC_NAME TEST_NAME TEST_PASS)

    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCES})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    add_dependencies("${EXEC_NAME}" "kernellib_bc")

    target_link_libraries("${EXEC_NAME}" "hipcl")

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

    add_test(NAME "${TEST_NAME}"
             COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME}"
             PASS_REGULAR_EXPRESSION "${TEST_PASS}")

endfunction()

set(SAMPLES
    hipmath
    hiptest
    bit_extract
    hcc_dialects
    0_MatrixTranspose
    0_MatrixMultiply
    1_hipEvent
    3_shared_memory
#    6_dynamic_shared
#    7_streams
#    9_unroll
)

foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()
