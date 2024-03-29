# NOTE: As of December 2021 the code of HIPCL was integrated to a new [CHIP-SPV](https://github.com/CHIP-SPV/chip-spv) project and will be continued there. This README and the HIPCL repo is left here for a legacy reference. #


# HIPCL library #
--------------------

### What is HIP? ###
[Heterogeneous-compute Interface for Portability](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_faq.md), or HIP, is a C++ runtime API and kernel language that allows developers to write code that runs on both AMD and NVidia GPUs. CUDA applications can be converted to HIP in a largely automated fashion.

### What is HIPCL ###

HIPCL is a library that allows applications using the HIP API to be run on devices which support OpenCL and SPIR-V, thus providing a portability path from CUDA to OpenCL. HIPCL was created by [Customized Parallel Computing](http://tuni.fi/cpc) group of [Tampere University](http://tuni.fi/en), Finland.





## Building HIPCL ##
--------------------

There are a few extra install/usage options documented in 'doc' directory.

HIPCL has some prerequisites to build:
 * LLVM + patched Clang
 * LLVM-SPIRV translator tool from Khronos
 * An OpenCL implementation with (at least partial) 2.x support;
   HIPCL requires Shared Virtual Memory and clCreateProgramWithIL() support

### Clang + LLVM ###

You'll need to build a patched Clang that can compile HIP source code to ELF+SPIR-V fat binaries.

Download LLVM + Clang:

    git clone https://github.com/llvm-mirror/llvm.git
    cd llvm
    git checkout -b release_80 origin/release_80
    cd tools
    git clone https://github.com/cpc/hipcl-clang.git clang
    cd clang
    git checkout -b release_80 origin/release_80

Build+install LLVM/Clang:

    cmake -DCMAKE_INSTALL_PREFIX=<llvm_install_dir> [other cmake flags] llvm-git-directory
    make
    sudo make install

### LLVM-SPIRV Translator ###

download, build+install the LLVM-SPIRV translator:

    git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
    cd SPIRV-LLVM-Translator
    git checkout -b release_80 origin/llvm_release_80
    mkdir build; cd build
    cmake -DLLVM_DIR=<llvm_install_dir>/lib/cmake/llvm ..
    make llvm-spirv
    sudo cp tools/llvm-spirv/llvm-spirv <llvm_install_dir>/bin/

### Known supported OpenCL implementations ###

At least Intel's "NEO" OpenCL implementation supports 2.x and SPIR-V on Intel GPUs.

It's also possible to use a sufficiently recent (2019/07+) [POCL](http://code.portablecl.org),
but it must be built with LLVM-SPIRV support:

    git clone https://github.com/pocl/pocl.git
    cd pocl
    mkdir build; cd build
    cmake -DCMAKE_INSTALL_PREFIX=/usr \
          -DWITH_LLVM_CONFIG=<llvm_install_dir>/bin/llvm-config \
          -DLLVM_SPIRV=<llvm_install_dir>/bin/llvm-spirv \
          ..
    make
    sudo make install

The last step (`sudo make install`) is optional - it's possible to use Pocl from build directory
(by exporting some env variables: `POCL_BULDING=1` and `OCL_ICD_VENDORS=<pocl-build-dir>/ocl-vendors`).
Note that -DCMAKE_INSTALL_PREFIX=/usr implies system-wide installation.
See https://github.com/pocl/pocl/blob/master/doc/sphinx/source/install.rst for details.

Whatever you end up using, make sure that clinfo lists your chosen OpenCL implementation.

### Build HIPCL library ###

build+install the HIPCL library:

    git clone https://github.com/cpc/hipcl.git
    cd hipcl
    mkdir build ; cd build;
    cmake -DCMAKE_INSTALL_PREFIX=<hipcl_install_dir> \
          -DCMAKE_CXX_COMPILER=<llvm_install_dir>/bin/clang++ \
          -DCMAKE_C_COMPILER=<llvm_install_dir>/bin/clang \
          ..
    make

`CMAKE_INSTALL_PREFIX` defaults to `/opt/hipcl`.
The samples directory contains some examples; these can be run from build directory, individually or via ctest.

`make install` will create `<hipcl_install_dir>/{lib/libhipcl.so, share/kernellib.bc, include/hip}` and
copy the examples to `<hipcl_install_dir>/bin/samples` directory.

Note that CMake removes RPATH at `make install` time, which means that the samples installed into
`<hipcl_install_dir>/bin` will look for `libhipcl.so` in the default system library paths (/usr/lib and such).

### Using HIPCL library ###

HIPCL provides a CMake export target named `hip::hipcl`. Using it from CMake is therefore straightforward:

    find_package(HIP REQUIRED CONFIG PATHS "${HIPCL_INSTALL_PREFIX}")
    target_link_libraries(your-executable hip::hipcl)

This will automatically add all required flags. Note that you must compile your project with CMAKE_CXX_COMPILER set to the Clang built in the first step.

For using outside CMake, there is a `${HIPCL_INSTALL_PREFIX}/bin/hipcl_config` binary which prints the required flags. Manually you can build using this command:

    <llvm_install_dir>/bin/clang++ -pthread -fPIE -O2 -g -std=c++11 `hipcl_config -C` -o binary source.cc -Wl,-rpath,<hipcl_install_prefix>/lib -L<hipcl_install_prefix>/lib -lhipcl

To see what compilation commands are actually run, and get the intermediate files (including the SPIR-V), add ``-v --save-temps`` to the compilation flags. Intermediate files will be saved into the current working directory. The SPIR-V that ends up embedded in the ELF binary is in a file named "a.out-hip-spir64-unknown-unknown-sm_20".


### CUDA conversion example ###

To convert a CUDA source to HIP source, use the hipify-clang tool from AMD's HIP repository:
https://github.com/ROCm-Developer-Tools/HIP/tree/master/hipify-clang

Usage:

    hipify-clang [hipify args] -- [clang cuda args]

E.g.

    ./hipify-clang -inplace -print-stats example.cu -- -x cuda --cuda-path=/usr/local/cuda-8.0 -I /usr/local/cuda-8.0/samples/common/inc

This should produce a source with CUDA API translated to HIP API calls. To build a HIPCL executable from this source, see above `Using HIPCL library`.

## Frequently encountered issues ##
------------------------

* `clEnqueueSVMMemCopy() failed with error -5` - this appears to be a driver bug on Intel GPUs, occurs when one tries to memcpy from read-only data stored in ELF to SVM memory. SVMMemCopy from other sources (stack / heap) works without issues.

* programs may take a long time to start. This is because there Clang inserts startup hooks which register SPIR-V binaries; HIPCL at this point compiles each, and for each program built, creates all kernels. This can take a long time on some implementations.

* HIPCL reports the global memory size from OpenCL as available memory, but unlike CUDA, it's not possible to allocate all of that memory in a single block; HIPCL is limited by CL_DEVICE_MAX_MEM_ALLOC_SIZE.

## Known HIPCL-Clang issues ##
------------------------

* Using HIP_DYNAMIC_SHARED() macro outside a function scope is not yet supported. Doing so will likely result in error: Assertion `FuncSet.size() <= 1 && "more than one function uses dynamic mem variable!"' failed.`

* There are unfortunately still some unresolved compiler bugs present in the HIPCL patches to Clang, so compilation may fail, especially when HIPCL is compiled with -O0 flag.

## Known libhipcl issues  ##
--------------------

Some of these are simply not yet implemented, some are missing because they would require an OpenCL extension.

### Device Side / Math Library ###

OpenCL Extension required:

 * __fsqrt_rd and various intrinsics for add/sub/div/mul with predefined rounding mode
   (currently these are mapped to OpenCL variants with default rounding mode)
 * __shfl and friends are only available on Intel via cl_intel_subgroups extension.

### Host runtime API ###

Implemented with caveats:

* hipEventElapsedTime() can return imprecise values

* hipModuleLaunchKernel accepts the "extra" argument, but the size of pointed to memory
  (HIP_LAUNCH_PARAM_BUFFER_SIZE) must be exactly the sum of sizes of individual
  arguments - no padding is allowed. Otherwise it's impossible to figure out how to set
  the OpenCL kernel arguments.

* A certain amount of Device properties are impossible to get via OpenCL API. Values
  reported by HIPCL are completely made up.

Not implemented and/or require extension to OpenCL:

* hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
* hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image,...);

  This API is not possible to implement with SPIR-V binaries,
  because there is no size parameter (only a void* pointer),
  and SPIR-V binaries don't have their size embedded.
  It might be possible to implement with disassembled
  text format of SPIR-V.

* hipSetDeviceFlags(unsigned flags)

  The flags change how the runtime waits for results (yield thread to OS
  or busy waiting / spinning)

* hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);
* hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
* hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
* hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);
* hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig);
* hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig);
* hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);
* hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config);
* hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig);
* hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);
* hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);
* hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags);
* hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId);
* hipError_t hipStreamQuery(hipStream_t stream)
* hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
                                hipModule_t hmod, const char* name);
* hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
* hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func);
* hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
* hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);
* hipError_t hipSetDeviceFlags(unsigned flags);

######  Symbol API Not Implemented  ######

* hipError_t hipMemcpyToSymbolAsync(void*, const void*, size_t, size_t,
                                  hipMemcpyKind, hipStream_t, const char*);
* hipError_t hipMemcpyFromSymbol(void*, const void*, size_t, size_t,
                               hipMemcpyKind, const char*);
* hipError_t hipMemcpyFromSymbolAsync(void*, const void*, size_t, size_t,
                                    hipMemcpyKind, hipStream_t, const char*);
* hipError_t hipGetSymbolAddress(void** devPtr, const void* symbolName);
* hipError_t hipGetSymbolSize(size_t* size, const void* symbolName);
* hipError_t hipMemcpyToSymbol(void*, const void*, size_t, size_t, hipMemcpyKind,
                             const char*);
* hipError_t hipMemcpyToSymbol(const void* symbolName, const void* src,
                             size_t sizeBytes, size_t offset __dparm(0),
                             hipMemcpyKind kind __dparm(hipMemcpyHostToDevice));

######  Peer2Peer Functions Are Not Implemented Yet ######

* hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
* hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
* hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);
* hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t sizeBytes);
* hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice, size_t sizeBytes, hipStream_t stream __dparm(0));

######  PROFILER Not implemented  ######

* hipError_t hipProfilerStart();
* hipError_t hipProfilerStop();

######  API CALLBACKs Not implemented  ######

* hipError_t hipRegisterApiCallback(uint32_t id, void* fun, void* arg);
* hipError_t hipRemoveApiCallback(uint32_t id);
* hipError_t hipRegisterActivityCallback(uint32_t id, void* fun, void* arg);
* hipError_t hipRemoveActivityCallback(uint32_t id);

######  TEXTURES not implemented  ######
