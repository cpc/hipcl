
option(SAVE_TEMPS "Save temporary compilation products" OFF)
option(VERBOSE "Verbose compilation" OFF)

if(SAVE_TEMPS)
  add_compile_options("--save-temps")
endif()

if(VERBOSE)
  add_compile_options("-v")
endif()


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
