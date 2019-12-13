
#include <list>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#include <CL/cl2.hpp>

#include "hipcl.hh"

/************************************************************/

#if !defined(SPDLOG_ACTIVE_LEVEL)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#endif

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

void setupSpdlog();

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
template <typename... Args>
void logDebug(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::debug(fmt, std::forward<const Args>(args)...);
}
#else
#define logDebug(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
template <typename... Args>
void logInfo(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::info(fmt, std::forward<const Args>(args)...);
}
#else
#define logInfo(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
template <typename... Args>
void logWarn(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::warn(fmt, std::forward<const Args>(args)...);
}
#else
#define logWarn(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
template <typename... Args>
void logError(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::error(fmt, std::forward<const Args>(args)...);
}
#else
#define logError(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL
template <typename... Args>
void logCritical(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::critical(fmt, std::forward<const Args>(args)...);
}
#else
#define logCritical(...) void(0)
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
/************************************************************/

#include "common.hh"

#ifdef __GNUC__
#define INIT_PRIORITY(x) __attribute__((init_priority(x)))
#else
#define INIT_PRIORITY(x)
#endif

#define SVM_ALIGNMENT 128

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class ClEvent {
  std::mutex EventMutex;
  cl::Event Event;
  hipStream_t Stream;
  event_status_e Status;
  unsigned Flags;
  cl::Context Context;

public:
  ClEvent(cl::Context &c, unsigned flags)
      : Event(), Stream(nullptr), Status(EVENT_STATUS_INIT), Flags(flags),
        Context(c) {}
  ~ClEvent() {}

  uint64_t getFinishTime();
  cl::Event &getEvent() { return Event; }
  bool isFromContext(cl::Context &Other) { return (Context == Other); }
  bool wasRecorded() const { return (Status == EVENT_STATUS_RECORDED); }
  bool recordStream(hipStream_t S, cl::Event &E);
  bool wait();
  bool isFinished();
};

typedef std::map<const void *, std::vector<hipFunction_t>> hipFunctionMap;

class ExecItem;

class ClKernel {
  std::mutex KernelMutex;
  cl::Kernel Kernel;
  std::string Name;
  OCLFuncInfo *FuncInfo;
  size_t TotalArgSize;
  cl::Context Context;
  //  hipFuncAttributes attributes;

public:
  ClKernel(cl::Context &C, cl::Kernel &&K)
      : Kernel(K), Name(), FuncInfo(nullptr), TotalArgSize(0), Context(C) {}
  ~ClKernel() {}
  bool setup(size_t Index, OpenCLFunctionInfoMap &FuncInfoMap);

  bool isNamed(const std::string &arg) const { return Name == arg; }
  bool isFromContext(const cl::Context &arg) const { return Context == arg; }

  int setAllArgs(void **args);
  int setAllArgs(void *args, size_t size);
  size_t getTotalArgSize() const { return TotalArgSize; }

  hipError_t launch(ExecItem *ei);
  hipError_t launch3(cl::CommandQueue &Queue, dim3 GridDim, dim3 BlockDim);
};

/********************************/

class ClProgram {
  cl::Program Program;
  cl::Context Context;
  cl::Device Device;
  std::vector<hipFunction_t> Kernels;
  OpenCLFunctionInfoMap FuncInfo;

public:
  ClProgram(cl::Context &C, cl::Device &D) : Program(), Context(C), Device(D) {}
  ~ClProgram();

  bool setup(std::string &binary);
  hipFunction_t getKernel(const char *name);
  hipFunction_t getKernel(std::string &name);
};

struct hipStreamCallbackData {
  hipStream_t Stream;
  hipError_t Status;
  void *UserData;
  hipStreamCallback_t Callback;
};

class SVMemoryRegion {
  // ContextMutex should be enough

  std::map<void *, size_t> SvmAllocations;
  cl::Context Context;

public:
  void init(cl::Context &C) { Context = C; }
  SVMemoryRegion &operator=(SVMemoryRegion &&rhs) {
    SvmAllocations = std::move(rhs.SvmAllocations);
    Context = std::move(rhs.Context);
    return *this;
  }

  void *allocate(size_t size);
  bool free(void *p, size_t *size);
  bool hasPointer(const void *p);
  bool pointerSize(void *ptr, size_t *size);
  bool pointerInfo(void *ptr, void **pbase, size_t *psize);
  int memCopy(void *dst, const void *src, size_t size, cl::CommandQueue &queue);
  int memFill(void *dst, size_t size, void *pattern, size_t patt_size,
              cl::CommandQueue &queue);
  void clear();
};

class ExecItem {

  dim3 GridDim;
  dim3 BlockDim;
  size_t SharedMem;
  cl::CommandQueue Queue;
  std::vector<uint8_t> ArgData;
  std::vector<std::tuple<size_t, size_t>> OffsetsSizes;

public:
  ExecItem(dim3 grid, dim3 block, size_t shared, cl::CommandQueue q)
      : GridDim(grid), BlockDim(block), SharedMem(shared), Queue(q) {}

  void setArg(const void *arg, size_t size, size_t offset);
  int setupAndLaunch(cl::Kernel &kernel, OCLFuncInfo *FuncInfo);
};

class ClQueue {
  cl::CommandQueue Queue;
  cl::Context Context;
  unsigned int Flags;
  int Priority;

public:
  ClQueue(cl::CommandQueue q, unsigned int f, int p)
      : Queue(q), Flags(f), Priority(p) {}

  ClQueue(ClQueue &&rhs) {
    Flags = rhs.Flags;
    Priority = rhs.Priority;
    Queue = std::move(rhs.Queue);
  }

  cl::CommandQueue &getQueue() { return Queue; }
  unsigned int getFlags() const { return Flags; }
  int getPriority() const { return Priority; }

  bool finish();
  bool enqueueBarrierForEvent(hipEvent_t event);
  bool addCallback(hipStreamCallback_t callback, void *userData);
  bool recordEvent(hipEvent_t e);

  bool memCopy(void *dst, const void *src, size_t size);
  bool memFill(void *dst, size_t size, void *pattern, size_t pat_size);
};

class ClDevice;

class ClContext {
  std::mutex ContextMutex;
  unsigned Flags;
  ClDevice *Device;
  cl::Context Context;

  SVMemoryRegion Memory;

  std::set<hipStream_t> Queues;
  hipStream_t DefaultQueue;
  std::stack<ExecItem *> ExecStack;

  std::map<const void *, ClProgram *> BuiltinPrograms;
  std::set<ClProgram *> Programs;

  hipStream_t findQueue(hipStream_t stream);

public:
  ClContext(ClDevice *D, unsigned f);
  ~ClContext();

  ClDevice *getDevice() const { return Device; }
  unsigned getFlags() const { return Flags; }
  hipStream_t getDefaultQueue() { return DefaultQueue; }
  void reset();

  bool eventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop);
  ClEvent *createEvent(unsigned Flags);
  bool createQueue(hipStream_t *stream, unsigned int Flags, int priority);
  bool releaseQueue(hipStream_t stream);
  hipError_t memCopy(void *dst, const void *src, size_t size,
                     hipStream_t stream);
  hipError_t memFill(void *dst, size_t size, void *pattern, size_t pat_size,
                     hipStream_t stream);
  hipError_t recordEvent(hipStream_t stream, hipEvent_t event);
  bool finishAll();

  void *allocate(size_t size);
  bool free(void *p);
  bool hasPointer(const void *p);
  bool getPointerSize(void *ptr, size_t *size);
  bool findPointerInfo(hipDeviceptr_t dptr, hipDeviceptr_t *pbase,
                       size_t *psize);

  hipError_t configureCall(dim3 grid, dim3 block, size_t shared, hipStream_t q);
  hipError_t setArg(const void *arg, size_t size, size_t offset);
  hipError_t launchHostFunc(const void *HostFunction);
  hipError_t createProgramBuiltin(std::string *module, const void *HostFunction,
                                  std::string &FunctionName);
  hipError_t destroyProgramBuiltin(const void *HostFunction);

  hipError_t launchWithKernelParams(dim3 grid, dim3 block, size_t shared,
                                    hipStream_t stream, void **kernelParams,
                                    hipFunction_t kernel);
  hipError_t launchWithExtraParams(dim3 grid, dim3 block, size_t shared,
                                   hipStream_t stream, void **extraParams,
                                   hipFunction_t kernel);

  ClProgram *createProgram(std::string &binary);
  hipError_t destroyProgram(ClProgram *prog);
};

class ClDevice {
  std::mutex DeviceMutex;

  hipDevice_t Index;
  hipDeviceProp_t Properties;
  std::map<hipDeviceAttribute_t, int> Attributes;
  size_t TotalUsedMem, GlobalMemSize;

  std::vector<std::string *> Modules;
  std::map<const void *, std::string *> HostPtrToModuleMap;
  std::map<const void *, std::string> HostPtrToNameMap;
  cl::Device Device;
  cl::Platform Platform;
  ClContext *PrimaryContext;
  std::set<ClContext *> Contexts;

  void setupProperties(int Index);

public:
  ClDevice(cl::Device d, cl::Platform p, hipDevice_t index);
  ClDevice(ClDevice &&rhs);
  ~ClDevice();
  void reset();
  cl::Device &getDevice() { return Device; }
  hipDevice_t getHipDeviceT() const { return Index; }
  ClContext *getPrimaryCtx() const { return PrimaryContext; }

  ClContext *newContext(unsigned int flags);
  bool addContext(ClContext *ctx);
  bool removeContext(ClContext *ctx);

  void registerModule(std::string *module);
  void unregisterModule(std::string *module);
  bool registerFunction(std::string *module, const void *HostFunction,
                        const char *FunctionName);
  bool getModuleAndFName(const void *HostFunction, std::string &FunctionName,
                         std::string **module);

  const char *getName() const { return Properties.name; }
  int getAttr(int *pi, hipDeviceAttribute_t attr);
  void copyProperties(hipDeviceProp_t *prop);

  size_t getGlobalMemSize() const { return GlobalMemSize; }
  size_t getUsedGlobalMem() const { return TotalUsedMem; }
  bool reserveMem(size_t bytes);
  bool releaseMem(size_t bytes);
};

void InitializeOpenCL();
void UnInitializeOpenCL();

extern size_t NumDevices;

ClDevice &CLDeviceById(int deviceId);

/********************************************************************/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
