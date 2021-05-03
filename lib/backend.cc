#include <algorithm>
#include <cassert>
#include <fstream>

#include "backend.hh"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#define FIND_QUEUE(stream)                                                     \
  ClQueue *Queue = findQueue(stream);                                          \
  if (Queue == nullptr)                                                        \
    return hipErrorInvalidResourceHandle;

#define FIND_QUEUE_LOCKED(stream)                                              \
  std::lock_guard<std::mutex> Lock(ContextMutex);                              \
  ClQueue *Queue = findQueue(stream);                                          \
  if (Queue == nullptr)                                                        \
    return hipErrorInvalidResourceHandle;

size_t NumDevices = 0;

static std::vector<ClDevice *> OpenCLDevices INIT_PRIORITY(120);
static std::vector<cl::Platform> Platforms INIT_PRIORITY(120);

/********************************/

bool ClEvent::updateFinishStatus() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  if (Status != EVENT_STATUS_RECORDING)
    return false;

  int Stat = Event->getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
  if (Stat <= CL_COMPLETE) {
    Status = EVENT_STATUS_RECORDED;
    return true;
  }
  return false;
}

bool ClEvent::recordStream(hipStream_t S, cl_event E) {
  std::lock_guard<std::mutex> Lock(EventMutex);

  Stream = S;
  Status = EVENT_STATUS_RECORDING;

  if (Event != nullptr) {
    cl_uint refc = Event->getInfo<CL_EVENT_REFERENCE_COUNT>();
    logDebug("removing old event, refc: {}\n", refc);

    delete Event;
  }

  Event = new cl::Event(E, true);
  return true;
}

bool ClEvent::wait() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  if (Status != EVENT_STATUS_RECORDING)
    return false;

  Event->wait();
  Status = EVENT_STATUS_RECORDED;
  return true;
}

uint64_t ClEvent::getFinishTime() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  int err;
  uint64_t ret = Event->getProfilingInfo<CL_PROFILING_COMMAND_END>(&err);
  assert(err == CL_SUCCESS);
  return ret;
}

/********************************/

static int setLocalSize(size_t shared, OCLFuncInfo *FuncInfo,
                        cl_kernel kernel) {

  int err = CL_SUCCESS;

  if (shared > 0) {
    logDebug("setLocalMemSize to {}\n", shared);
    size_t LastArgIdx = FuncInfo->ArgTypeInfo.size() - 1;
    if (FuncInfo->ArgTypeInfo[LastArgIdx].space != OCLSpace::Local) {
      // this can happen if for example the llvm optimizes away
      // the dynamic local variable
      logWarn("Can't set the dynamic local size, "
              "because the kernel doesn't use any local memory.\n");
    } else {
      err = ::clSetKernelArg(kernel, LastArgIdx, shared, nullptr);
      if (err != CL_SUCCESS) {
        logError("clSetKernelArg() failed to set dynamic local size!\n");
      }
    }
  }

  return err;
}

bool ClKernel::setup(size_t Index, OpenCLFunctionInfoMap &FuncInfoMap) {
  int err = 0;
  Name = Kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&err);
  if (err != CL_SUCCESS) {
    logError("clGetKernelInfo(CL_KERNEL_FUNCTION_NAME) failed: {}\n", err);
    return false;
  }

  logDebug("Kernel {} is: {} \n", Index, Name);

  auto it = FuncInfoMap.find(Name);
  // for global support
  if (it == FuncInfoMap.end())
    return true;
  FuncInfo = it->second;

  // TODO attributes
  cl_uint NumArgs = Kernel.getInfo<CL_KERNEL_NUM_ARGS>(&err);
  if (err != CL_SUCCESS) {
    logError("clGetKernelInfo(CL_KERNEL_NUM_ARGS) failed: {}\n", err);
    return false;
  }
  assert(FuncInfo->ArgTypeInfo.size() == NumArgs);

  if (NumArgs > 0) {
    logDebug("Kernel {} numArgs: {} \n", Name, NumArgs);
    logDebug("  RET_TYPE: {} {} {}\n", FuncInfo->retTypeInfo.size,
             (unsigned)FuncInfo->retTypeInfo.space,
             (unsigned)FuncInfo->retTypeInfo.type);
    for (auto &argty : FuncInfo->ArgTypeInfo) {
      logDebug("  ARG: SIZE {} SPACE {} TYPE {}\n", argty.size,
               (unsigned)argty.space, (unsigned)argty.type);
      TotalArgSize += argty.size;
    }
  }
  return true;
}

int ClKernel::setAllArgs(void **args, size_t shared) {
  void *p;
  int err;

  for (cl_uint i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];

    if (ai.type == OCLType::Pointer) {
      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      p = *(void **)(args[i]);
      logDebug("setArg SVM {} to PTR {}\n", i, p);
      err = ::clSetKernelArgSVMPointer(Kernel(), i, p);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
        return err;
      }
    } else {
      logDebug("setArg {} SIZE {}\n", i, ai.size);
      err = ::clSetKernelArg(Kernel(), i, ai.size, args[i]);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArg failed with error {}\n", err);
        return err;
      }
    }
  }

  return setLocalSize(shared, FuncInfo, Kernel());
}

int ClKernel::setAllArgs(void *args, size_t size, size_t shared) {
  void *p = args;
  int err;

  for (cl_uint i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];

    if (ai.type == OCLType::Pointer) {
      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      void *pp = *(void **)p;
      logDebug("setArg SVM {} to PTR {}\n", i, pp);
      err = ::clSetKernelArgSVMPointer(Kernel(), i, pp);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
        return err;
      }
    } else {
      logDebug("setArg {} SIZE {}\n", i, ai.size);
      err = ::clSetKernelArg(Kernel(), i, ai.size, p);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArg failed with error {}\n", err);
        return err;
      }
    }

    p = (char *)p + ai.size;
  }

  return setLocalSize(shared, FuncInfo, Kernel());
}

/********************************/

bool ClProgram::setup(std::string &binary) {

  size_t numWords = binary.size() / 4;
  int32_t *bindata = new int32_t[numWords + 1];
  std::memcpy(bindata, binary.data(), binary.size());
  bool res = parseSPIR(bindata, numWords, FuncInfo);
  delete[] bindata;
  if (!res) {
    logError("SPIR-V parsing failed\n");
    return false;
  }

  int err;
  std::vector<char> binary_vec(binary.begin(), binary.end());
  Program = cl::Program(Context, binary_vec, false, &err);
  if (err != CL_SUCCESS) {
    logError("CreateProgramWithIL Failed: {}\n", err);
    return false;
  }

  std::string name = Device.getInfo<CL_DEVICE_NAME>();

  int build_failed = Program.compile("-x spir -cl-kernel-arg-info");

  std::string log = Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(Device, &err);
  if (err != CL_SUCCESS) {
    logError("clGetProgramBuildInfo() Failed: {}\n", err);
    return false;
  }
  logDebug("Program BUILD LOG for device {}:\n{}\n", name, log);
  if (build_failed != CL_SUCCESS) {
    logError("clBuildProgram() Failed: {}\n", build_failed);
    return false;
  }

  cl_program prg = Program();
  prg = clLinkProgram(Context(), 0, NULL,
                      symbolSupported() ? "-cl-take-global-address" : NULL, 1,
                      &prg, NULL, NULL, &build_failed);

  if (!prg) {
    logError("clLinkProgram() Failed: {}\n", build_failed);
    return false;
  }
  Program = prg;
  log = Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(Device, &err);
  if (err != CL_SUCCESS) {
    logError("clGetProgramBuildInfo() Failed: {}\n", err);
    return false;
  }
  logDebug("Program BUILD LOG for device {}:\n{}\n", name, log);
  if (build_failed != CL_SUCCESS) {
    logError("clLinkProgram() Failed: {}\n", build_failed);
    return false;
  }

  std::vector<cl::Kernel> kernels;
  err = Program.createKernels(&kernels);
  if (err != CL_SUCCESS) {
    logError("clCreateKernels() Failed: {}\n", err);
    return false;
  }
  logDebug("Kernels in program: {} \n", kernels.size());
  Kernels.resize(kernels.size());

  for (size_t i = 0; i < kernels.size(); ++i) {
    ClKernel *k = new ClKernel(Context, std::move(kernels[i]));
    if (k == nullptr)
      return false; // TODO memleak
    if (!k->setup(i, FuncInfo))
      return false;
    Kernels[i] = k;
  }
  return true;
}

ClProgram::~ClProgram() {
  for (hipFunction_t K : Kernels) {
    delete K;
  }
  Kernels.clear();

  std::set<OCLFuncInfo *> PtrsToDelete;
  for (auto &kv : FuncInfo)
    PtrsToDelete.insert(kv.second);
  for (auto &Ptr : PtrsToDelete)
    delete Ptr;
}

hipFunction_t ClProgram::getKernel(std::string &name) {
  for (hipFunction_t It : Kernels) {
    if (It->isNamed(name)) {
      return It;
    }
  }
  return nullptr;
}

hipFunction_t ClProgram::getKernel(const char *name) {
  std::string SearchName(name);
  return getKernel(SearchName);
}

bool ClProgram::getSymbolAddressSize(const void *name, hipDeviceptr_t *dptr,
                                     size_t *bytes) {
  cl_int err = clGetDeviceGlobalVariablePointerINTEL_ptr(
      Device(), Program(), (const char *)name, bytes, dptr);
  if (err != CL_SUCCESS)
    return false;
  else
    return true;
}

/********************************/

void *SVMemoryRegion::allocate(size_t size) {
  void *Ptr = ::clSVMAlloc(Context(), CL_MEM_READ_WRITE, size, SVM_ALIGNMENT);
  if (Ptr) {
    logDebug("clSVMAlloc allocated: {} / {}\n", Ptr, size);
    SvmAllocations.emplace(Ptr, size);
  } else
    logError("clSVMAlloc of {} bytes failed\n", size);
  return Ptr;
}

bool SVMemoryRegion::free(void *p, size_t *size) {
  auto I = SvmAllocations.find(p);
  if (I != SvmAllocations.end()) {
    void *Ptr = I->first;
    *size = I->second;
    logDebug("clSVMFree on: {}\n", Ptr);
    SvmAllocations.erase(I);
    ::clSVMFree(Context(), Ptr);
    return true;
  } else {
    logError("clSVMFree on unknown memory: {}\n", p);
    return false;
  }
}

bool SVMemoryRegion::hasPointer(const void *p) {
  logDebug("hasPointer on: {}\n", p);
  if (SvmAllocations.find((void *)p) != SvmAllocations.end())
    return true;
  if (GlobalPointers.find((void *)p) != GlobalPointers.end())
    return true;
  return false;
}

void SVMemoryRegion::addGlobal(void *ptr, size_t size) {
  GlobalPointers.emplace(ptr, size);
}

void SVMemoryRegion::removeGlobal(void *ptr) {
  auto it = GlobalPointers.find(ptr);
  if (it != GlobalPointers.end())
    GlobalPointers.erase(it);
}

bool SVMemoryRegion::pointerSize(void *ptr, size_t *size) {
  logDebug("pointerSize on: {}\n", ptr);
  auto I = SvmAllocations.find(ptr);
  if (I != SvmAllocations.end()) {
    *size = I->second;
    return true;
  }
  auto J = GlobalPointers.find(ptr);
  if (J != GlobalPointers.end()) {
    *size = I->second;
    return true;
  }
  return false;
}

bool SVMemoryRegion::pointerInfo(void *ptr, void **pbase, size_t *psize) {
  logDebug("pointerInfo on: {}\n", ptr);
  for (auto I : SvmAllocations) {
    if ((I.first <= ptr) && (ptr < ((const char *)I.first + I.second))) {
      if (pbase)
        *pbase = I.first;
      if (psize)
        *psize = I.second;
      return true;
    }
  }
  for (auto I : GlobalPointers) {
    if ((I.first <= ptr) && (ptr < ((const char *)I.first + I.second))) {
      if (pbase)
        *pbase = I.first;
      if (psize)
        *psize = I.second;
      return true;
    }
  }
  return false;
}

void SVMemoryRegion::clear() {
  for (auto I : SvmAllocations) {
    ::clSVMFree(Context(), I.first);
  }
  SvmAllocations.clear();
}

/***********************************************************************/

hipError_t ClQueue::memCopy(void *dst, const void *src, size_t size) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  logDebug("clSVMmemcpy {} -> {} / {} B\n", src, dst, size);
  cl_event ev = nullptr;
  int retval =
      ::clEnqueueSVMMemcpy(Queue(), CL_FALSE, dst, src, size, 0, nullptr, &ev);
  if (retval == CL_SUCCESS) {
    if (LastEvent != nullptr) {
      logDebug("memCopy: LastEvent == {}, will be: {}", (void *)LastEvent,
               (void *)ev);
      clReleaseEvent(LastEvent);
    } else
      logDebug("memCopy: LastEvent == NULL, will be: {}\n", (void *)ev);
    LastEvent = ev;
  } else {
    logError("clEnqueueSVMMemCopy() failed with error {}\n", retval);
  }
  return (retval == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;
}

hipError_t ClQueue::memFill(void *dst, size_t size, void *pattern,
                            size_t patt_size) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  logDebug("clSVMmemfill {} / {} B\n", dst, size);
  cl_event ev = nullptr;
  int retval = ::clEnqueueSVMMemFill(Queue(), dst, pattern, patt_size, size, 0,
                                     nullptr, &ev);
  if (retval == CL_SUCCESS) {
    if (LastEvent != nullptr) {
      logDebug("memFill: LastEvent == {}, will be: {}", (void *)LastEvent,
               (void *)ev);
      clReleaseEvent(LastEvent);
    } else
      logDebug("memFill: LastEvent == NULL, will be: {}\n", (void *)ev);
    LastEvent = ev;
  } else {
    logError("clEnqueueSVMMemFill() failed with error {}\n", retval);
  }

  return (retval == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;
}

bool ClQueue::finish() {
  int err = Queue.finish();
  if (err != CL_SUCCESS)
    logError("clFinish() failed with error {}\n", err);
  return err == CL_SUCCESS;
}

static void notifyOpenCLevent(cl_event event, cl_int status, void *data) {
  hipStreamCallbackData *Data = (hipStreamCallbackData *)data;
  Data->Callback(Data->Stream, Data->Status, Data->UserData);
  delete Data;
}

bool ClQueue::addCallback(hipStreamCallback_t callback, void *userData) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  int err;
  if (LastEvent == nullptr) {
    callback(this, hipSuccess, userData);
    return true;
  }

  hipStreamCallbackData *Data = new hipStreamCallbackData{};
  Data->Stream = this;
  Data->Callback = callback;
  Data->UserData = userData;
  Data->Status = hipSuccess;
  err = ::clSetEventCallback(LastEvent, CL_COMPLETE, notifyOpenCLevent, Data);
  if (err != CL_SUCCESS)
    logError("clSetEventCallback failed with error {}\n", err);
  return (err == CL_SUCCESS);
}

bool ClQueue::enqueueBarrierForEvent(hipEvent_t ProvidedEvent) {
  std::lock_guard<std::mutex> Lock(QueueMutex);
  // CUDA API cudaStreamWaitEvent:
  // event may be from a different device than stream.

  cl::Event MarkerEvent;
  logDebug("Queue is: {}\n", (void *)(Queue()));
  int err = Queue.enqueueMarkerWithWaitList(nullptr, &MarkerEvent);
  if (err != CL_SUCCESS)
    return false;

  cl::vector<cl::Event> Events = {MarkerEvent, ProvidedEvent->getEvent()};
  cl::Event barrier;
  err = Queue.enqueueBarrierWithWaitList(&Events, &barrier);
  if (err != CL_SUCCESS) {
    logError("clEnqueueBarrierWithWaitList failed with error {}\n", err);
    return false;
  }

  if (LastEvent)
    clReleaseEvent(LastEvent);
  LastEvent = barrier();

  return true;
}

bool ClQueue::recordEvent(hipEvent_t event) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  /* slightly tricky WRT refcounts.
   * if LastEvents != NULL, it should have refcount 1.
   * if NULL, enqueue a marker here;
   * libOpenCL will process it & decrease refc to 1;
   * we retain it here because d-tor is called at } and releases it.
   *
   * in both cases, event->recordStream should Retain */
  if (LastEvent == nullptr) {
    cl::Event MarkerEvent;
    int err = Queue.enqueueMarkerWithWaitList(nullptr, &MarkerEvent);
    if (err) {
      logError ("enqueueMarkerWithWaitList FAILED with {}\n", err);
      return false;
    } else {
      LastEvent = MarkerEvent();
      clRetainEvent(LastEvent);
    }
  }

  logDebug("record Event: {} on Queue: {}\n", (void *)(LastEvent),
           (void *)(Queue()));

  cl_uint refc1, refc2;
  int err =
      ::clGetEventInfo(LastEvent, CL_EVENT_REFERENCE_COUNT, 4, &refc1, NULL);
  assert(err == CL_SUCCESS);
  // can be >1 because recordEvent can be called >1 on the same event
  assert(refc1 >= 1);

  return event->recordStream(this, LastEvent);

  err = ::clGetEventInfo(LastEvent, CL_EVENT_REFERENCE_COUNT, 4, &refc2, NULL);
  assert(err == CL_SUCCESS);
  assert(refc2 >= 2);
  assert(refc2 == (refc1 + 1));
}

hipError_t ClQueue::launch(ClKernel *Kernel, ExecItem *Arguments) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  if (Arguments->setupAllArgs(Kernel) != CL_SUCCESS) {
    logError("Failed to set kernel arguments for launch! \n");
    return hipErrorLaunchFailure;
  }

  dim3 GridDim = Arguments->GridDim;
  dim3 BlockDim = Arguments->BlockDim;

  const cl::NDRange global(GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z);
  const cl::NDRange local(BlockDim.x, BlockDim.y, BlockDim.z);

  cl::Event ev;
  int err = Queue.enqueueNDRangeKernel(Kernel->get(), cl::NullRange, global,
                                       local, nullptr, &ev);

  if (err != CL_SUCCESS)
    logError("clEnqueueNDRangeKernel() failed with: {}\n", err);
  hipError_t retval = (err == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;

  if (retval == hipSuccess) {
    if (LastEvent != nullptr) {
      logDebug("Launch: LastEvent == {}, will be: {}", (void *)LastEvent,
               (void *)ev.get());
      clReleaseEvent(LastEvent);
    } else
      logDebug("launch: LastEvent == NULL, will be: {}\n", (void *)ev.get());
    LastEvent = ev.get();
    clRetainEvent(LastEvent);
  }

  delete Arguments;
  return retval;
}

hipError_t ClQueue::launch3(ClKernel *Kernel, dim3 grid, dim3 block) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  dim3 GridDim = grid;
  dim3 BlockDim = block;

  const cl::NDRange global(GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z);
  const cl::NDRange local(BlockDim.x, BlockDim.y, BlockDim.z);

  cl::Event ev;
  int err = Queue.enqueueNDRangeKernel(Kernel->get(), cl::NullRange, global,
                                       local, nullptr, &ev);

  if (err != CL_SUCCESS)
    logError("clEnqueueNDRangeKernel() failed with: {}\n", err);
  hipError_t retval = (err == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;

  if (retval == hipSuccess) {
    if (LastEvent != nullptr) {
      logDebug("Launch3: LastEvent == {}, will be: {}", (void *)LastEvent,
               (void *)ev.get());
      clReleaseEvent(LastEvent);
    } else
      logDebug("launch3: LastEvent == NULL, will be: {}\n", (void *)ev.get());
    LastEvent = ev.get();
    clRetainEvent(LastEvent);
  }

  return retval;
}

/***********************************************************************/

void ExecItem::setArg(const void *arg, size_t size, size_t offset) {
  if ((offset + size) > ArgData.size())
    ArgData.resize(offset + size + 1024);

  std::memcpy(ArgData.data() + offset, arg, size);
  logDebug("setArg on {} size {} offset {}\n", (void *)this, size, offset);
  OffsetsSizes.push_back(std::make_tuple(offset, size));
}

int ExecItem::setupAllArgs(ClKernel *kernel) {
  OCLFuncInfo *FuncInfo = kernel->getFuncInfo();
  size_t NumLocals = 0;
  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].space == OCLSpace::Local)
      ++NumLocals;
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert (NumLocals <= 1);

  if ((OffsetsSizes.size()+NumLocals) != FuncInfo->ArgTypeInfo.size()) {
      logError("Some arguments are still unset\n");
      return CL_INVALID_VALUE;
  }

  if (OffsetsSizes.size() == 0)
    return CL_SUCCESS;

  std::sort(OffsetsSizes.begin(), OffsetsSizes.end());
  if ((std::get<0>(OffsetsSizes[0]) != 0) ||
      (std::get<1>(OffsetsSizes[0]) == 0)) {
          logError("Invalid offset/size\n");
          return CL_INVALID_VALUE;
      }

  // check args are set
  if (OffsetsSizes.size() > 1) {
    for (size_t i = 1; i < OffsetsSizes.size(); ++i) {
      if ( (std::get<0>(OffsetsSizes[i]) == 0) ||
           (std::get<1>(OffsetsSizes[i]) == 0) ||
           (
           (std::get<0>(OffsetsSizes[i - 1]) + std::get<1>(OffsetsSizes[i - 1])) >
            std::get<0>(OffsetsSizes[i]))
           ) {
          logError("Invalid offset/size\n");
          return CL_INVALID_VALUE;
        }
    }
  }

  const unsigned char *start = ArgData.data();
  void *p;
  int err;
  for (cl_uint i = 0; i < OffsetsSizes.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];
    logDebug("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n", i,
             std::get<0>(OffsetsSizes[i]), std::get<1>(OffsetsSizes[i]),
             (unsigned)ai.type, (unsigned)ai.space, ai.size);

    if (ai.type == OCLType::Pointer) {

      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      assert(std::get<1>(OffsetsSizes[i]) == ai.size);
      p = *(void **)(start + std::get<0>(OffsetsSizes[i]));
      logDebug("setArg SVM {} to {}\n", i, p);
      err = ::clSetKernelArgSVMPointer(kernel->get().get(), i, p);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
        return err;
      }
    } else {
      size_t size = std::get<1>(OffsetsSizes[i]);
      size_t offs = std::get<0>(OffsetsSizes[i]);
      void* value = (void*)(start + offs);
      logDebug("setArg {} size {} offs {}\n", i, size, offs);
      err =
          ::clSetKernelArg(kernel->get().get(), i, size, value);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArg failed with error {}\n", err);
        return err;
      }
    }
  }

  return setLocalSize(SharedMem, FuncInfo, kernel->get().get());
}

/***********************************************************************/

/* errinfo is a pointer to an error string.
 * private_info and cb represent a pointer to binary data that is
 * returned by the OpenCL implementation that can be used
 * to log additional information helpful in debugging the error.
 * user_data is a pointer to user supplied data.
 */

static void intel_driver_cb(
    const char *errinfo,
    const void *private_info,
    size_t cb,
    void *user_data) {

    logDebug("INTEL DIAG: {}\n", errinfo);
}

ClContext::ClContext(ClDevice *D, unsigned f) {
  Device = D;
  Flags = f;
  int err;

  if (D->supportsIntelDiag()) {
    logDebug("creating context with Intel Debugging\n");
    cl_bitfield vl =
            CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL
            | CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL
            | CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL;
    cl_context_properties props[] = {
        CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL,
        (cl_context_properties)vl,
        0 };
    Context = cl::Context(D->getDevice(), props,
                          intel_driver_cb, this,
                          &err);
  } else {
    logDebug("creating context for dev: {}\n", D->getName());
    Context = cl::Context(D->getDevice(), NULL, NULL, NULL, &err);
  }
  assert(err == CL_SUCCESS);
  cl_platform_id pid = D->getDevice().getInfo<CL_DEVICE_PLATFORM>();
  clGetDeviceGlobalVariablePointerINTEL_ptr =
      (cl_int(*)(cl_device_id, cl_program, const char *, size_t *, void **))
          clGetExtensionFunctionAddressForPlatform(
              pid, "clGetDeviceGlobalVariablePointerINTEL");

  cl::CommandQueue CmdQueue(Context, Device->getDevice(),
                            CL_QUEUE_PROFILING_ENABLE, &err);
  assert(err == CL_SUCCESS);

  DefaultQueue = new ClQueue(CmdQueue, 0, 0);

  Memory.init(Context);
}

void ClContext::reset() {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  int err;

  while (!this->ExecStack.empty()) {
    ExecItem *Item = ExecStack.top();
    delete Item;
    this->ExecStack.pop();
  }

  this->Queues.clear();
  delete DefaultQueue;
  this->Memory.clear();

  cl::CommandQueue CmdQueue(Context, Device->getDevice(),
                            CL_QUEUE_PROFILING_ENABLE, &err);
  assert(err == CL_SUCCESS);

  DefaultQueue = new ClQueue(CmdQueue, 0, 0);
}

ClContext::~ClContext() {

  while (!this->ExecStack.empty()) {
    ExecItem *Item = ExecStack.top();
    delete Item;
    this->ExecStack.pop();
  }

  for (ClQueue *Q : Queues) {
    delete Q;
  }
  Queues.clear();
  delete DefaultQueue;
  Memory.clear();

  for (ClProgram *P : Programs) {
    delete P;
  }
  Programs.clear();

  for (auto It : BuiltinPrograms) {
    delete It.second;
  }
  BuiltinPrograms.clear();
}

hipStream_t ClContext::findQueue(hipStream_t stream) {
  if (stream == nullptr || stream == DefaultQueue)
    return DefaultQueue;

  auto I = Queues.find(stream);
  if (I == Queues.end())
    return nullptr;
  return *I;
}

ClEvent *ClContext::createEvent(unsigned flags) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  return new ClEvent(Context, flags);
}

void *ClContext::allocate(size_t size) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  if (!Device->reserveMem(size))
    return nullptr;

  void *retval = Memory.allocate(size);
  if (retval == nullptr)
    Device->releaseMem(size);
  return retval;
}

bool ClContext::free(void *p) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  size_t size;

  bool retval = Memory.free(p, &size);
  if (retval)
    Device->releaseMem(size);
  return retval;
}

bool ClContext::hasPointer(const void *p) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  return Memory.hasPointer(p);
}

bool ClContext::getPointerSize(void *ptr, size_t *size) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  return Memory.pointerSize(ptr, size);
}

bool ClContext::findPointerInfo(hipDeviceptr_t dptr, hipDeviceptr_t *pbase,
                                size_t *psize) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  return Memory.pointerInfo(dptr, pbase, psize);
}

hipError_t ClContext::memCopy(void *dst, const void *src, size_t size,
                              hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);

  if (Memory.hasPointer(dst) || Memory.hasPointer(src))
    return Queue->memCopy(dst, src, size);
  else
    return hipErrorInvalidDevicePointer;
}

hipError_t ClContext::memFill(void *dst, size_t size, void *pattern,
                              size_t pat_size, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);

  if (!Memory.hasPointer(dst))
    return hipErrorInvalidDevicePointer;

  return Queue->memFill(dst, size, pattern, pat_size);
}

hipError_t ClContext::recordEvent(hipStream_t stream, hipEvent_t event) {
  FIND_QUEUE_LOCKED(stream);

  return Queue->recordEvent(event) ? hipSuccess : hipErrorInvalidContext;
}

#define NANOSECS 1000000000

hipError_t ClContext::eventElapsedTime(float *ms, hipEvent_t start,
                                       hipEvent_t stop) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  assert(start->isFromContext(Context));
  assert(stop->isFromContext(Context));

  if (!start->isRecordingOrRecorded() || !stop->isRecordingOrRecorded())
    return hipErrorInvalidResourceHandle;

  start->updateFinishStatus();
  stop->updateFinishStatus();
  if (!start->isFinished() || !stop->isFinished())
    return hipErrorNotReady;

  uint64_t Started = start->getFinishTime();
  uint64_t Finished = stop->getFinishTime();

  logDebug("EventElapsedTime: STARTED {} / {} FINISHED {} / {} \n",
           (void *)start, Started, (void *)stop, Finished);

  // apparently fails for Intel NEO, god knows why
  // assert(Finished >= Started);
  uint64_t Elapsed;
  if (Finished < Started) {
    logWarn("Finished < Started\n");
    Elapsed = Started - Finished;
  } else
    Elapsed = Finished - Started;
  uint64_t MS = (Elapsed / NANOSECS)*1000;
  uint64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  *ms = (float)MS + FractInMS;
  return hipSuccess;
}

bool ClContext::createQueue(hipStream_t *stream, unsigned flags, int priority) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  int err;
  cl::CommandQueue NewQueue(Context, Device->getDevice(),
                            CL_QUEUE_PROFILING_ENABLE, &err);
  assert(err == CL_SUCCESS);

  hipStream_t Ptr = new ClQueue(NewQueue, flags, priority);
  Queues.insert(Ptr);
  *stream = Ptr;
  return true;
}

bool ClContext::releaseQueue(hipStream_t stream) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  auto I = Queues.find(stream);
  if (I == Queues.end())
    return false;
  hipStream_t QueuePtr = *I;
  delete QueuePtr;
  Queues.erase(I);
  return true;
}

bool ClContext::finishAll() {
  std::vector<cl::CommandQueue> Copies;
  {
    std::lock_guard<std::mutex> Lock(ContextMutex);
    for (hipStream_t I : Queues) {
      Copies.push_back(I->getQueue());
    }
    Copies.push_back(DefaultQueue->getQueue());
  }

  for (cl::CommandQueue &I : Copies) {
    int err = I.finish();
    if (err != CL_SUCCESS) {
      logError("clFinish() failed with error {}\n", err);
      return false;
    }
  }
  return true;
}

hipError_t ClContext::configureCall(dim3 grid, dim3 block, size_t shared,
                                    hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);

  ExecItem *NewItem = new ExecItem(grid, block, shared, Queue);
  ExecStack.push(NewItem);

  return hipSuccess;
}

hipError_t ClContext::setArg(const void *arg, size_t size, size_t offset) {
  // Can't do a size check here b/c we don't know the kernel yet
  std::lock_guard<std::mutex> Lock(ContextMutex);
  ExecStack.top()->setArg(arg, size, offset);
  return hipSuccess;
}

hipError_t ClContext::createProgramBuiltin(std::string *module,
                                           const void *HostFunction,
                                           std::string &FunctionName) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  logDebug("createProgramBuiltin: {}\n", FunctionName);

  auto it = ProgramsCache.find(module);
  if (it != ProgramsCache.end()) {
    BuiltinPrograms[HostFunction] = it->second;
    return hipSuccess;
  }

  ClProgram *p = new ClProgram(Context, Device->getDevice());
  if (p == nullptr)
    return hipErrorOutOfMemory;

  if (!p->setup(*module)) {
    logCritical("Failed to build program for '{}'", FunctionName);
    delete p;
    return hipErrorInitializationError;
  }
  ProgramsCache[module] = p;
  BuiltinPrograms[HostFunction] = p;
  return hipSuccess;
}

hipError_t ClContext::createProgramBuiltinVar(std::string *module,
                                              const void *HostVar,
                                              std::string &VarName) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  logDebug("createProgramBuiltinVar: {}\n", VarName);
  if (!symbolSupported()) {
    logError("createProgramBuiltinVar is not supported on this platform");
    return hipErrorNotSupported;
  }

  ClProgram *p;
  auto it = ProgramsCache.find(module);
  if (it != ProgramsCache.end()) {
    p = it->second;
  } else {
    p = new ClProgram(Context, Device->getDevice());
    if (p == nullptr)
      return hipErrorOutOfMemory;

    if (!p->setup(*module)) {
      logCritical("Failed to build program for '{}'", VarName);
      delete p;
      return hipErrorInitializationError;
    }
    ProgramsCache[module] = p;
  }
  hipDeviceptr_t dPtr;
  size_t sz;
  if (!it->second->getSymbolAddressSize(VarName.c_str(), &dPtr, &sz)) {
    logError("Symbol '{}' was not found in program", VarName);
  }
  GlobalVarsMap[VarName] = std::make_tuple(p, dPtr, sz);
  Memory.addGlobal(dPtr, sz);
  BuiltinVars[HostVar] = p;
  return hipSuccess;
}

hipError_t ClContext::destroyProgramBuiltin(std::string *module) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  auto it = ProgramsCache.find(module);
  if (it == ProgramsCache.end())
    return hipErrorUnknown;
  ClProgram *prog = it->second;
  ProgramsCache.erase(it);

  for (auto iter = BuiltinPrograms.begin(); iter != BuiltinPrograms.end();) {
    if (iter->second == prog) {
      iter = BuiltinPrograms.erase(iter);
    } else {
      ++iter;
    }
  }
  for (auto iter = BuiltinVars.begin(); iter != BuiltinVars.end();) {
    if (iter->second == prog) {
      iter = BuiltinVars.erase(iter);
    } else {
      ++iter;
    }
  }
  for (auto iter = GlobalVarsMap.begin(); iter != GlobalVarsMap.end();) {
    if (std::get<0>(iter->second) == prog) {
      Memory.removeGlobal(std::get<1>(iter->second));
      iter = GlobalVarsMap.erase(iter);
    } else {
      ++iter;
    }
  }
  delete prog;
  return hipSuccess;
}

hipError_t ClContext::launchHostFunc(const void *HostFunction) {

  std::string FunctionName;
  std::string *module;

  if (!Device->getModuleAndFName(HostFunction, FunctionName, &module)) {
    logCritical("can NOT find kernel with stub address {} for device {}\n",
                HostFunction, Device->getHipDeviceT());
    return hipErrorLaunchFailure;
  }

  std::lock_guard<std::mutex> Lock(ContextMutex);

  ClKernel *Kernel = nullptr;
  // TODO can this happen ?
  if (BuiltinPrograms.find(HostFunction) != BuiltinPrograms.end())
    Kernel = BuiltinPrograms[HostFunction]->getKernel(FunctionName);

  if (Kernel == nullptr) {
    logCritical("can NOT find kernel with stub address {} for device {}\n",
                HostFunction, Device->getHipDeviceT());
    return hipErrorLaunchFailure;
  }

  ExecItem *Arguments;
  Arguments = ExecStack.top();
  ExecStack.pop();

  return Arguments->launch(Kernel);
}

hipError_t ClContext::launchWithKernelParams(dim3 grid, dim3 block,
                                             size_t shared, hipStream_t stream,
                                             void **kernelParams,
                                             hipFunction_t kernel) {
  FIND_QUEUE_LOCKED(stream);

  if (!kernel->isFromContext(Context))
    return hipErrorLaunchFailure;

  int err = kernel->setAllArgs(kernelParams, shared);
  if (err != CL_SUCCESS) {
    logError("Failed to set kernel arguments for launch! \n");
    return hipErrorLaunchFailure;
  }

  return stream->launch3(kernel, grid, block);
}

hipError_t ClContext::launchWithExtraParams(dim3 grid, dim3 block,
                                            size_t shared, hipStream_t stream,
                                            void **extraParams,
                                            hipFunction_t kernel) {
  FIND_QUEUE_LOCKED(stream);

  if (!kernel->isFromContext(Context))
    return hipErrorLaunchFailure;

  void *args = nullptr;
  size_t size = 0;

  void **p = extraParams;
  while (*p && (*p != HIP_LAUNCH_PARAM_END)) {
    if (*p == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
      args = (void *)p[1];
      p += 2;
      continue;
    } else if (*p == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
      size = (size_t)p[1];
      p += 2;
      continue;
    } else {
      logError("Unknown parameter in extraParams: {}\n", *p);
      return hipErrorLaunchFailure;
    }
  }

  if (args == nullptr || size == 0) {
    logError("extraParams doesn't contain all required parameters\n");
    return hipErrorLaunchFailure;
  }

  // TODO This only accepts structs with no padding.
  if (size != kernel->getTotalArgSize()) {
    logError("extraParams doesn't have correct size\n");
    return hipErrorLaunchFailure;
  }

  int err = kernel->setAllArgs(args, size, shared);
  if (err != CL_SUCCESS) {
    logError("Failed to set kernel arguments for launch! \n");
    return hipErrorLaunchFailure;
  }
  return stream->launch3(kernel, grid, block);
}

ClProgram *ClContext::createProgram(std::string &binary) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  ClProgram *prog = new ClProgram(Context, Device->getDevice());
  if (prog == nullptr)
    return nullptr;

  if (!prog->setup(binary)) {
    delete prog;
    return nullptr;
  }

  Programs.emplace(prog);
  return prog;
}

hipError_t ClContext::destroyProgram(ClProgram *prog) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  auto it = Programs.find(prog);
  if (it == Programs.end())
    return hipErrorInvalidHandle;

  for (auto iter = GlobalVarsMap.begin(); iter != GlobalVarsMap.end();) {
    if (std::get<0>(iter->second) == prog) {
      Memory.removeGlobal(std::get<1>(iter->second));
      iter = GlobalVarsMap.erase(iter);
    } else {
      ++iter;
    }
  }

  Programs.erase(it);
  return hipSuccess;
}

bool ClContext::getSymbolAddressSize(const void *name, hipDeviceptr_t *dptr,
                                     size_t *bytes) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  auto it = GlobalVarsMap.find((const char *)name);
  if (it != GlobalVarsMap.end()) {
    *dptr = std::get<1>(it->second);
    *bytes = std::get<2>(it->second);
    return true;
  }
  for (auto it = Programs.begin(); it != Programs.end(); ++it) {
    if ((*it)->getSymbolAddressSize(name, dptr, bytes)) {
      GlobalVarsMap[(const char *)name] = std::make_tuple(*it, *dptr, *bytes);
      Memory.addGlobal(*dptr, *bytes);
      return true;
    }
  }
  return false;
}

/***********************************************************************/

void ClDevice::setupProperties(int index) {
  cl_int err;
  std::string Temp;
  cl::Device Dev = this->Device;

  Temp = Dev.getInfo<CL_DEVICE_NAME>(&err);
  strncpy(Properties.name, Temp.c_str(), 255);
  Properties.name[255] = 0;

  Properties.totalGlobalMem = Dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err);

  Properties.sharedMemPerBlock = Dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);

  Properties.maxThreadsPerBlock =
      Dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);

  std::vector<size_t> wi = Dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  Properties.maxThreadsDim[0] = wi[0];
  Properties.maxThreadsDim[1] = wi[1];
  Properties.maxThreadsDim[2] = wi[2];

  // Maximum configured clock frequency of the device in MHz.
  Properties.clockRate = 1000 * Dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  Properties.multiProcessorCount = Dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  Properties.l2CacheSize = Dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  Properties.totalConstMem = Dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up
  Properties.regsPerBlock = 64;

  // The minimum subgroup size on an intel GPU
  if (Dev.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
    std::vector<uint> sg = Dev.getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
    if (sg.begin() != sg.end())
      Properties.warpSize = *std::min_element(sg.begin(), sg.end());
  }
  Properties.maxGridSize[0] = Properties.maxGridSize[1] =
      Properties.maxGridSize[2] = 65536;
  Properties.memoryClockRate = 1000;
  Properties.memoryBusWidth = 256;
  Properties.major = 2;
  Properties.minor = 0;

  Properties.maxThreadsPerMultiProcessor = 10;

  Properties.computeMode = 0;
  Properties.arch = {};

  Temp = Dev.getInfo<CL_DEVICE_EXTENSIONS>();
  if (Temp.find("cl_khr_global_int32_base_atomics") != std::string::npos)
    Properties.arch.hasGlobalInt32Atomics = 1;
  else
    Properties.arch.hasGlobalInt32Atomics = 0;

  if (Temp.find("cl_khr_local_int32_base_atomics") != std::string::npos)
    Properties.arch.hasSharedInt32Atomics = 1;
  else
    Properties.arch.hasSharedInt32Atomics = 0;

  if (Temp.find("cl_khr_int64_base_atomics") != std::string::npos) {
    Properties.arch.hasGlobalInt64Atomics = 1;
    Properties.arch.hasSharedInt64Atomics = 1;
  }
  else {
    Properties.arch.hasGlobalInt64Atomics = 1;
    Properties.arch.hasSharedInt64Atomics = 1;
  }

  if (Temp.find("cl_khr_fp64") != std::string::npos) 
    Properties.arch.hasDoubles = 1;
  else
    Properties.arch.hasDoubles = 0;

  Properties.clockInstructionRate = 2465;
  Properties.concurrentKernels = 1;
  Properties.pciDomainID = 0;
  Properties.pciBusID = 0x10;
  Properties.pciDeviceID = 0x40 + index;
  Properties.isMultiGpuBoard = 0;
  Properties.canMapHostMemory = 1;
  Properties.gcnArch = 0;
  Properties.integrated = 0;
  Properties.maxSharedMemoryPerMultiProcessor = 0;
}

ClDevice::ClDevice(cl::Device d, cl::Platform p, hipDevice_t index) {
  Device = d;
  Platform = p;
  Index = index;
  SupportsIntelDiag = false;

  setupProperties(index);

  std::string extensions = d.getInfo<CL_DEVICE_EXTENSIONS>();
  if (extensions.find("cl_intel_driver_diag") != std::string::npos) {
      logDebug("Intel debug extension supported\n");
      SupportsIntelDiag = true;
  }

  TotalUsedMem = 0;
  MaxUsedMem = 0;
  GlobalMemSize = Properties.totalGlobalMem;
  PrimaryContext = nullptr;

  logDebug("Device {} is {}: name \"{}\" \n",
           index, (void *)this, Properties.name);
}

void ClDevice::setPrimaryCtx() {
  PrimaryContext = new ClContext(this, 0);
}

void ClDevice::reset() {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  // TODO should we remove all contexts ?
  PrimaryContext->reset();
  for (ClContext *C : Contexts) {
    C->reset();
  }
}

ClDevice::~ClDevice() {
  delete PrimaryContext;
  logInfo("Max used memory on device {}: {} MB\n", Properties.name, (MaxUsedMem >> 20));
  logDebug("Destroy device {}\n", Properties.name);
  for (ClContext *C : Contexts) {
    delete C;
  }
  Contexts.clear();
}

ClDevice::ClDevice(ClDevice &&rhs) {
  Index = rhs.Index;
  Properties = rhs.Properties;
  Attributes = std::move(rhs.Attributes);

  Device = std::move(rhs.Device);
  Platform = std::move(rhs.Platform);
  PrimaryContext = std::move(rhs.PrimaryContext);
  Contexts = std::move(rhs.Contexts);
  TotalUsedMem = rhs.TotalUsedMem;
  MaxUsedMem = rhs.MaxUsedMem;
  GlobalMemSize = rhs.GlobalMemSize;
}

bool ClDevice::reserveMem(size_t bytes) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  if (bytes <= (GlobalMemSize - TotalUsedMem)) {
    TotalUsedMem += bytes;
    if (TotalUsedMem > MaxUsedMem)
      MaxUsedMem = TotalUsedMem;
    logDebug("Currently used memory on dev {}: {} M\n", Index, (TotalUsedMem >> 20));
    return true;
  } else {
    logError("Can't allocate {} bytes of memory\n", bytes);
    return false;
  }
}

bool ClDevice::releaseMem(size_t bytes) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  if (TotalUsedMem >= bytes) {
    TotalUsedMem -= bytes;
    return true;
  } else {
    return false;
  }
}

int ClDevice::getAttr(int *pi, hipDeviceAttribute_t attr) {
  auto I = Attributes.find(attr);
  if (I != Attributes.end()) {
    *pi = I->second;
    return 0;
  } else {
    return 1;
  }
}

void ClDevice::copyProperties(hipDeviceProp_t *prop) {
  if (prop)
    std::memcpy(prop, &this->Properties, sizeof(hipDeviceProp_t));
}

bool ClDevice::addContext(ClContext *ctx) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  auto it = Contexts.find(ctx);
  if (it != Contexts.end())
    return false;
  Contexts.emplace(ctx);
  return true;
}

bool ClDevice::removeContext(ClContext *ctx) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  auto I = std::find(Contexts.begin(), Contexts.end(), ctx);
  if (I == Contexts.end())
    return false;

  Contexts.erase(I);
  delete ctx;
  // TODO:
  // As per CUDA docs , attempting to access ctx from those threads which has
  // this ctx as current, will result in the error
  // HIP_ERROR_CONTEXT_IS_DESTROYED.
  return true;
}

ClContext *ClDevice::newContext(unsigned int flags) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  ClContext *ctx = new ClContext(this, flags);
  if (ctx != nullptr)
    Contexts.emplace(ctx);
  return ctx;
}

void ClDevice::registerModule(std::string *module) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  Modules.push_back(module);
}

void ClDevice::unregisterModule(std::string *module) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  auto it = std::find(Modules.begin(), Modules.end(), module);
  if (it == Modules.end()) {
    logCritical("unregisterModule: couldn't find {}\n", (void *)module);
    return;
  } else
    Modules.erase(it);

  const void *HostFunction = nullptr;
  const void *HostVar = nullptr;
  std::map<const void *, std::string *>::iterator it2, e;

  it2 = HostPtrToModuleMap.begin();
  while (it2 != HostPtrToModuleMap.end()) {
    if (it2->second == module) {
      HostFunction = it2->first;
      HostPtrToNameMap.erase(HostFunction);
      it2 = HostPtrToModuleMap.erase(it2);
    } else {
      ++it2;
    }
  }

  it2 = HostVarPtrToModuleMap.begin();
  while (it2 != HostVarPtrToModuleMap.end()) {
    if (it2->second == module) {
      HostVar = it2->first;
      HostVarPtrToNameMap.erase(HostVar);
      it2 = HostVarPtrToModuleMap.erase(it2);
    } else {
      ++it2;
    }
  }
  PrimaryContext->destroyProgramBuiltin(module);
  for (ClContext *C : Contexts) {
    C->destroyProgramBuiltin(module);
  }
}

bool ClDevice::registerFunction(std::string *module, const void *HostFunction,
                                const char *FunctionName) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  auto it = std::find(Modules.begin(), Modules.end(), module);
  if (it == Modules.end()) {
    logError("Module PTR not FOUND: {}\n", (void *)module);
    return false;
  }

  HostPtrToModuleMap.emplace(std::make_pair(HostFunction, module));
  HostPtrToNameMap.emplace(std::make_pair(HostFunction, FunctionName));

  std::string temp(FunctionName);
  return (PrimaryContext->createProgramBuiltin(module, HostFunction, temp) ==
          hipSuccess);
}

bool ClDevice::registerVar(std::string *module, const void *HostVar,
                           const char *VarName) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  auto it = std::find(Modules.begin(), Modules.end(), module);
  if (it == Modules.end()) {
    logError("Module PTR not FOUND: {}\n", (void *)module);
    return false;
  }

  HostVarPtrToModuleMap.emplace(std::make_pair(HostVar, module));
  HostVarPtrToNameMap.emplace(std::make_pair(HostVar, VarName));

  std::string temp(VarName);
  return (PrimaryContext->createProgramBuiltinVar(module, HostVar, temp) ==
          hipSuccess);
}

bool ClDevice::getModuleAndFName(const void *HostFunction,
                                 std::string &FunctionName,
                                 std::string **module) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  auto it1 = HostPtrToModuleMap.find(HostFunction);
  auto it2 = HostPtrToNameMap.find(HostFunction);

  if ((it1 == HostPtrToModuleMap.end()) || (it2 == HostPtrToNameMap.end()))
    return false;

  FunctionName.assign(it2->second);
  *module = it1->second;
  return true;
}

bool ClDevice::getModuleAndVarName(const void *HostVar, std::string &VarName,
                                   std::string **module) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  auto it1 = HostVarPtrToModuleMap.find(HostVar);
  auto it2 = HostVarPtrToNameMap.find(HostVar);

  if ((it1 == HostVarPtrToModuleMap.end()) ||
      (it2 == HostVarPtrToNameMap.end()))
    return false;

  VarName.assign(it2->second);
  *module = it1->second;
  return true;
}
/***********************************************************************/

ClDevice &CLDeviceById(int deviceId) { return *OpenCLDevices.at(deviceId); }

class InvalidDeviceType : public std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

class InvalidPlatformOrDeviceNumber : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

static void InitializeOpenCLCallOnce() {

  cl_int err = cl::Platform::get(&Platforms);
  std::string ver;
  if (err != CL_SUCCESS)
    return;

  OpenCLDevices.clear();
  NumDevices = 0;
  std::vector<cl::Device> Devices;
  const char *selected_platform_str = std::getenv("HIPCL_PLATFORM");
  const char *selected_device_str = std::getenv("HIPCL_DEVICE");
  const char *selected_device_type_str = std::getenv("HIPCL_DEVICE_TYPE");
  int selected_platform = -1;
  int selected_device = -1;
  cl_bitfield selected_dev_type = 0;
  try {
    if (selected_platform_str) {
      selected_platform = std::stoi(selected_platform_str);
      if ((selected_platform < 0) || (selected_platform >= Platforms.size()))
        throw InvalidPlatformOrDeviceNumber(
            "HIPCL_PLATFORM: platform number out of range");
    }

    if (selected_device_str) {
      selected_device = std::stoi(selected_device_str);
      Devices.clear();
      if (selected_platform < 0)
        selected_platform = 0;
      err =
          Platforms[selected_platform].getDevices(CL_DEVICE_TYPE_ALL, &Devices);
      if (err != CL_SUCCESS)
        throw InvalidPlatformOrDeviceNumber(
            "HIPCL_DEVICE: can't get devices for platform");
      if ((selected_device < 0) || (selected_device >= Devices.size()))
        throw InvalidPlatformOrDeviceNumber(
            "HIPCL_DEVICE: device number out of range");
    }

    if (selected_device_type_str) {
      std::string s(selected_device_type_str);
      if (s == "all")
        selected_dev_type = CL_DEVICE_TYPE_ALL;
      else if (s == "cpu")
        selected_dev_type = CL_DEVICE_TYPE_CPU;
      else if (s == "gpu")
        selected_dev_type = CL_DEVICE_TYPE_GPU;
      else if (s == "default")
        selected_dev_type = CL_DEVICE_TYPE_DEFAULT;
      else if (s == "accel")
        selected_dev_type = CL_DEVICE_TYPE_ACCELERATOR;
      else
        throw InvalidDeviceType(
            "Unknown value provided for HIPCL_DEVICE_TYPE\n");
    }
  } catch (const InvalidDeviceType &e) {
    logCritical("{}\n", e.what());
    return;
  } catch (const InvalidPlatformOrDeviceNumber &e) {
    logCritical("{}\n", e.what());
    return;
  } catch (const std::invalid_argument &e) {
    logCritical(
        "Could not convert HIPCL_PLATFORM or HIPCL_DEVICES to a number\n");
    return;
  } catch (const std::out_of_range &e) {
    logCritical("HIPCL_PLATFORM or HIPCL_DEVICES is out of range\n");
    return;
  }

  if (selected_dev_type == 0)
    selected_dev_type = CL_DEVICE_TYPE_ALL;
  for (auto Platform : Platforms) {
    Devices.clear();
    err = Platform.getDevices(selected_dev_type, &Devices);
    if (err != CL_SUCCESS)
      continue;
    if (Devices.size() == 0)
      continue;
    if (selected_platform >= 0 && (Platforms[selected_platform] != Platform))
      continue;

    for (cl::Device &Dev : Devices) {
      ver.clear();
      if (selected_device >= 0 && (Devices[selected_device] != Dev))
        continue;
      ver = Dev.getInfo<CL_DEVICE_IL_VERSION>(&err);
      if ((err == CL_SUCCESS) && (ver.rfind("SPIR-V_1.", 0) == 0)) {
        ClDevice *temp = new ClDevice(Dev, Platform, NumDevices);
        temp->setPrimaryCtx();
        OpenCLDevices.emplace_back(temp);
        ++NumDevices;
      }
    }
  }

  logDebug("DEVICES {}", NumDevices);
  assert(NumDevices == OpenCLDevices.size());
}

void InitializeOpenCL() {
  static std::once_flag OpenClInitialized;
  std::call_once(OpenClInitialized, InitializeOpenCLCallOnce);
}

static void UnInitializeOpenCLCallOnce() {
  logDebug("DEVICES UNINITALIZE \n");

  for (ClDevice *d : OpenCLDevices) {
    delete d;
  }

  for (auto Platform : Platforms) {
    Platform.unloadCompiler();
  }

  // spdlog::details::os::sleep_for_millis(18000);
}

void UnInitializeOpenCL() {
  static std::once_flag OpenClUnInitialized;
  std::call_once(OpenClUnInitialized, UnInitializeOpenCLCallOnce);
}

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
