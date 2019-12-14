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

bool ClEvent::isFinished() {
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

bool ClKernel::setup(size_t Index, OpenCLFunctionInfoMap &FuncInfoMap) {
  int err = 0;
  Name = Kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&err);
  if (err != CL_SUCCESS) {
    logError("clGetKernelInfo(CL_KERNEL_FUNCTION_NAME) failed: {}\n", err);
    return false;
  }

  logDebug("Kernel {} is: {} \n", Index, Name);

  auto it = FuncInfoMap.find(Name);
  assert(it != FuncInfoMap.end());
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
      logDebug("  ARG_TYPE: {} {} {}\n", argty.size, (unsigned)argty.space,
               (unsigned)argty.type);
      TotalArgSize += argty.size;
    }
  }
  return true;
}

int ClKernel::setAllArgs(void **args) {
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
      logDebug("ERR {}\n", err);
      if (err != CL_SUCCESS)
        return err;
    } else {
      logDebug("setArg {} SIZE {}\n", i, ai.size);
      err = ::clSetKernelArg(Kernel(), i, ai.size, args[i]);
      logDebug("ERR {}\n", err);
      if (err != CL_SUCCESS)
        return err;
    }
  }
  return 0;
}

int ClKernel::setAllArgs(void *args, size_t size) {
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
      logDebug("ERR {}\n", err);
      if (err != CL_SUCCESS)
        return err;

    } else {
      logDebug("setArg {} SIZE {}\n", i, ai.size);
      err = ::clSetKernelArg(Kernel(), i, ai.size, p);
      logDebug("ERR {}\n", err);
      if (err != CL_SUCCESS)
        return err;
    }

    p = (char *)p + ai.size;
  }
  return 0;
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

  int build_failed = Program.build("-x spir -cl-kernel-arg-info");

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
  return (SvmAllocations.find((void *)p) != SvmAllocations.end());
}

bool SVMemoryRegion::pointerSize(void *ptr, size_t *size) {
  logDebug("pointerSize on: {}\n", ptr);
  auto I = SvmAllocations.find(ptr);
  if (I != SvmAllocations.end()) {
    *size = I->second;
    return true;
  } else {
    return false;
  }
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
  }
  return (retval == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;
}

bool ClQueue::finish() { return Queue.finish() == CL_SUCCESS; }

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
  if (err != CL_SUCCESS)
    return false;

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
    Queue.enqueueMarkerWithWaitList(nullptr, &MarkerEvent);
    LastEvent = MarkerEvent();
    clRetainEvent(LastEvent);
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

  if (Arguments->setupAllArgs(Kernel) != CL_SUCCESS)
    return hipErrorLaunchFailure;

  dim3 GridDim = Arguments->GridDim;
  dim3 BlockDim = Arguments->BlockDim;

  const cl::NDRange global(GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z);
  const cl::NDRange local(BlockDim.x, BlockDim.y, BlockDim.z);

  cl::Event ev;
  int err = Queue.enqueueNDRangeKernel(Kernel->get(), cl::NullRange, global,
                                       local, nullptr, &ev);

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
  assert(OffsetsSizes.size() == FuncInfo->ArgTypeInfo.size());

  std::sort(OffsetsSizes.begin(), OffsetsSizes.end());
  assert(std::get<0>(OffsetsSizes[0]) == 0);
  // check args are set
  if (OffsetsSizes.size() > 1) {
    for (size_t i = 1; i < OffsetsSizes.size(); ++i) {
      if (std::get<0>(OffsetsSizes[i - 1]) + std::get<1>(OffsetsSizes[i - 1]) !=
          std::get<0>(OffsetsSizes[i]))
        return CL_INVALID_VALUE;
    }
  }

  const unsigned char *start = ArgData.data();
  void *p;
  int err;
  for (cl_uint i = 0; i < OffsetsSizes.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];
    if (ai.type == OCLType::Pointer) {
      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      assert(std::get<1>(OffsetsSizes[i]) == ai.size);
      p = *(void **)(start + std::get<0>(OffsetsSizes[i]));
      logDebug("setArg SVM {} to {}\n", i, p);
      err = ::clSetKernelArgSVMPointer(kernel->get().get(), i, p);
      logDebug("ERR {}\n", err);
      if (err != CL_SUCCESS)
        return err;
    } else {
      logDebug("setArg {}\n", i);
      err =
          ::clSetKernelArg(kernel->get().get(), i, std::get<1>(OffsetsSizes[i]),
                           (start + std::get<0>(OffsetsSizes[i])));
      logDebug("ERR {}\n", err);
      if (err != CL_SUCCESS)
        return err;
    }
  }
  return CL_SUCCESS;
}

/***********************************************************************/

ClContext::ClContext(ClDevice *D, unsigned f) {
  Device = D;
  Flags = f;
  int err;
  Context = cl::Context(D->getDevice(), NULL, NULL, NULL, &err);
  assert(err == CL_SUCCESS);

  cl::CommandQueue CmdQueue(Context, Device->getDevice(),
                            CL_QUEUE_PROFILING_ENABLE, &err);
  assert(err == CL_SUCCESS);

  DefaultQueue = new ClQueue(CmdQueue, 0, 0);

  Memory.init(Context);
}

void ClContext::reset() {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  while (!this->ExecStack.empty()) {
    ExecItem *Item = ExecStack.top();
    delete Item;
    this->ExecStack.pop();
  }

  this->Queues.clear();
  delete DefaultQueue;
  this->Memory.clear();

  cl::CommandQueue CmdQueue(Context, Device->getDevice(),
                            CL_QUEUE_PROFILING_ENABLE);
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

bool ClContext::eventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  assert(start->isFromContext(Context));
  assert(stop->isFromContext(Context));

  if (!stop->wasRecorded())
    return false;

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
  uint64_t S = Elapsed / NANOSECS;
  uint64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  *ms = (float)S + FractInMS;
  return true;
}

bool ClContext::createQueue(hipStream_t *stream, unsigned flags, int priority) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  cl::CommandQueue NewQueue(Context, Device->getDevice(),
                            CL_QUEUE_PROFILING_ENABLE);
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
  }

  for (cl::CommandQueue &I : Copies) {
    I.finish();
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

  ClProgram *p = new ClProgram(Context, Device->getDevice());
  if (p == nullptr)
    return hipErrorOutOfMemory;

  if (!p->setup(*module)) {
    logCritical("Failed to build program for '{}'", FunctionName);
    delete p;
    return hipErrorInitializationError;
  }

  BuiltinPrograms[HostFunction] = p;
  return hipSuccess;
}

hipError_t ClContext::destroyProgramBuiltin(const void *HostFunction) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  auto it = BuiltinPrograms.find(HostFunction);
  if (it == BuiltinPrograms.end())
    return hipErrorUnknown;
  delete it->second;
  BuiltinPrograms.erase(it);
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

  // TODO can this happen ?
  assert(BuiltinPrograms.find(HostFunction) != BuiltinPrograms.end());

  ClKernel *Kernel = BuiltinPrograms[HostFunction]->getKernel(FunctionName);
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

  int err = kernel->setAllArgs(kernelParams);
  if (err != CL_SUCCESS)
    return hipErrorLaunchFailure;

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

  int err = kernel->setAllArgs(args, size);
  if (err != CL_SUCCESS)
    return hipErrorLaunchFailure;

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

  Programs.erase(it);
  return hipSuccess;
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

  Properties.clockRate = Dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  Properties.multiProcessorCount = Dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  Properties.l2CacheSize = Dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  Properties.totalConstMem = Dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up
  Properties.regsPerBlock = 64;
  Properties.warpSize = 64;
  Properties.maxGridSize[0] = Properties.maxGridSize[1] =
      Properties.maxGridSize[2] = 65536;
  Properties.memoryClockRate = 1000;
  Properties.memoryBusWidth = 256;
  Properties.major = 2;
  Properties.minor = 0;
  Properties.maxThreadsPerMultiProcessor = 10;

  Properties.computeMode = 0;
  Properties.arch = {};
  Properties.arch.hasGlobalInt32Atomics = 1;
  Properties.arch.hasSharedInt32Atomics = 1;
  Properties.arch.hasGlobalInt64Atomics = 1;
  Properties.arch.hasSharedInt64Atomics = 1;

  Properties.clockInstructionRate = 2465;
  Properties.concurrentKernels = 1;
  Properties.pciDomainID = 0;
  Properties.pciBusID = 0x10;
  Properties.pciDeviceID = 0x40 + index;
  Properties.isMultiGpuBoard = 0;
  Properties.canMapHostMemory = 1;
  Properties.gcnArch = 0;
  Properties.integrated = 0;
}

ClDevice::ClDevice(cl::Device d, cl::Platform p, hipDevice_t index) {
  Device = d;
  Platform = p;
  Index = index;

  setupProperties(index);

  TotalUsedMem = 0;
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
  GlobalMemSize = rhs.GlobalMemSize;
}

bool ClDevice::reserveMem(size_t bytes) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  if (bytes <= (GlobalMemSize - TotalUsedMem)) {
    TotalUsedMem += bytes;
    return true;
  } else {
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
  Modules.erase(it);

  const void *HostFunction = nullptr;
  std::map<const void *, std::string *>::iterator it2, e;
  for (it2 = HostPtrToModuleMap.begin(), e = HostPtrToModuleMap.end(); it2 != e;
       ++it2) {
    if (it2->second == module) {
      HostFunction = it2->first;
      break;
    }
  }

  assert(HostFunction);
  HostPtrToModuleMap.erase(it2);
  auto it3 = HostPtrToNameMap.find(HostFunction);
  HostPtrToNameMap.erase(it3);

  PrimaryContext->destroyProgramBuiltin(HostFunction);

  for (ClContext *C : Contexts) {
    C->destroyProgramBuiltin(HostFunction);
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

/***********************************************************************/

ClDevice &CLDeviceById(int deviceId) { return *OpenCLDevices.at(deviceId); }

static void InitializeOpenCLCallOnce() {

  cl_int err = cl::Platform::get(&Platforms);
  std::string ver;
  if (err != CL_SUCCESS)
    return;

  OpenCLDevices.clear();
  NumDevices = 0;
  std::vector<cl::Device> Devices;

  for (auto Platform : Platforms) {
    Devices.clear();
    err = Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);
    if (err != CL_SUCCESS)
      continue;
    if (Devices.size() == 0)
      continue;

    for (cl::Device &Dev : Devices) {
      ver.clear();
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
