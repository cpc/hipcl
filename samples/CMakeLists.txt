set(SAMPLES
    hipmath
    hiptest
    bit_extract
    hcc_dialects
    0_MatrixTranspose
    1_hipEvent
    3_shared_memory
#    6_dynamic_shared
#    7_streams
#    9_unroll
)

foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()
