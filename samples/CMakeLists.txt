
option(SAVE_TEMPS "Save temporary compilation products" OFF)
option(VERBOSE "Verbose compilation" OFF)

if(SAVE_TEMPS)
  add_compile_options("--save-temps")
endif()

if(VERBOSE)
  add_compile_options("-v")
endif()

# ARGN = test args
function(add_hipcl_test EXEC_NAME TEST_NAME TEST_PASS SOURCE)

    set(TEST_EXEC_ARGS ${ARGN})
    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCE})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" "${SANITIZER_LIBS}" "hipcl")

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

    add_test(NAME "${TEST_NAME}"
             COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME}" ${TEST_EXEC_ARGS}
             )

    set_tests_properties("${TEST_NAME}" PROPERTIES
             PASS_REGULAR_EXPRESSION "${TEST_PASS}")


endfunction()


# ARGN = sources
function(add_hipcl_binary EXEC_NAME)

    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCES})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" "${SANITIZER_LIBS}" "hipcl")

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

endfunction()

# ARGN = sources
function(add_hipcl_binary_device_link EXEC_NAME)
    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCES})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" "${SANITIZER_LIBS}" "hipcl")

    target_compile_options("${EXEC_NAME}" PRIVATE "-fgpu-rdc")

    target_link_options("${EXEC_NAME}" PRIVATE
        "-fgpu-rdc"
	"--hip-link"
	"$<INSTALL_INTERFACE:--hip-llvm-pass-path=${HIPCL_LLVM_DIR}>"
	"$<BUILD_INTERFACE:--hip-llvm-pass-path=${CMAKE_BINARY_DIR}/llvm_passes>"
	"$<INSTALL_INTERFACE:--hip-device-lib-path=${HIPCL_DATA_DIR}>"
	"$<BUILD_INTERFACE:--hip-device-lib-path=${CMAKE_BINARY_DIR}>"
	"--hip-device-lib=kernellib.bc")

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

endfunction()


set(SAMPLES
    hipmath
    hiptest
    bit_extract
    hcc_dialects
    fp16
    0_MatrixTranspose
    0_MatrixMultiply
    1_hipEvent
    2_vecadd
    3_shared_memory
    4_shfl
    5_2dshfl
    6_dynamic_shared
    hipInfo
#    7_streams
#    9_unroll
    10_memcpy3D
    hipSymbol
    hipDeviceLink
)

foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()

add_subdirectory(hip-cuda)
