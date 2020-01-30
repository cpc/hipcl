set(SAMPLES
    BinomialOption
    BitonicSort
    DCT
    dwtHaar1D
    FastWalshTransform
    FloydWarshall
# requires dynamic memory:
#   Histogram
# is broken:
#   PrefixSum
    RecursiveGaussian
    SimpleConvolution
)

foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()
