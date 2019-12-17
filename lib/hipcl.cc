#include <algorithm>
#include <cassert>
#include <fstream>
#include <stack>

#include "backend.hh"

static thread_local hipError_t tls_LastError = hipSuccess;

static thread_local ClContext *tls_defaultCtx = nullptr;

static thread_local std::stack<ClContext *> tls_ctxStack;

static thread_local bool tls_getPrimaryCtx = true;

static hipFunctionMap g_functions INIT_PRIORITY(120);

static ClContext *getTlsDefaultCtx() {
  if ((tls_defaultCtx == nullptr) && (NumDevices > 0))
    tls_defaultCtx = CLDeviceById(0).getPrimaryCtx();
  return tls_defaultCtx;
}

#define RETURN(x)                                                              \
  do {                                                                         \
    hipError_t err = (x);                                                      \
    tls_LastError = err;                                                       \
    return err;                                                                \
  } while (0)

#define ERROR_IF(cond, err)                                                    \
  if (cond)                                                                    \
    do {                                                                       \
      logError("{} : {}", #cond, err);                                         \
      tls_LastError = err;                                                     \
      return err;                                                              \
  } while (0)

#define ERROR_CHECK_DEVNUM(device)                                             \
  ERROR_IF(((device < 0) || ((size_t)device >= NumDevices)),                   \
           hipErrorInvalidDevice)

/***********************************************************************/

hipError_t hipGetDevice(int *deviceId) {
  InitializeOpenCL();

  ERROR_IF((deviceId == nullptr), hipErrorInvalidValue);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  *deviceId = cont->getDevice()->getHipDeviceT();
  RETURN(hipSuccess);
}

hipError_t hipGetDeviceCount(int *count) {
  InitializeOpenCL();
  ERROR_IF((count == nullptr), hipErrorInvalidValue);
  *count = NumDevices;
  RETURN(hipSuccess);
}

hipError_t hipSetDevice(int deviceId) {
  InitializeOpenCL();

  ERROR_CHECK_DEVNUM(deviceId);

  tls_defaultCtx = CLDeviceById(deviceId).getPrimaryCtx();

  RETURN(hipSuccess);
}

hipError_t hipDeviceSynchronize(void) {
  getTlsDefaultCtx()->finishAll();
  RETURN(hipSuccess);
}

hipError_t hipDeviceReset(void) {

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  ClDevice *dev = cont->getDevice();
  dev->reset();
  RETURN(hipSuccess);
}

hipError_t hipDeviceGet(hipDevice_t *device, int ordinal) {
  InitializeOpenCL();

  ERROR_IF(((ordinal < 0) || ((size_t)ordinal >= NumDevices)),
           hipErrorInvalidValue);

  ERROR_IF((device == nullptr), hipErrorInvalidDevice);

  *device = ordinal;
  RETURN(hipSuccess);
}

hipError_t hipDeviceComputeCapability(int *major, int *minor,
                                      hipDevice_t deviceId) {
  InitializeOpenCL();
  ERROR_CHECK_DEVNUM(deviceId);

  hipDeviceProp_t props;
  CLDeviceById(deviceId).copyProperties(&props);
  if (major)
    *major = props.major;
  if (minor)
    *minor = props.minor;

  RETURN(hipSuccess);
}

hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr,
                                 int deviceId) {
  InitializeOpenCL();
  ERROR_CHECK_DEVNUM(deviceId);

  if (CLDeviceById(deviceId).getAttr(pi, attr))
    RETURN(hipErrorInvalidValue);
  else
    RETURN(hipSuccess);
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId) {
  InitializeOpenCL();
  ERROR_CHECK_DEVNUM(deviceId);

  CLDeviceById(deviceId).copyProperties(prop);

  RETURN(hipSuccess);
}

hipError_t hipDeviceGetLimit(size_t *pValue, enum hipLimit_t limit) {
  ERROR_IF((pValue == nullptr), hipErrorInvalidValue);

  RETURN(hipErrorUnsupportedLimit);
}

hipError_t hipDeviceGetName(char *name, int len, hipDevice_t deviceId) {
  InitializeOpenCL();
  ERROR_CHECK_DEVNUM(deviceId);

  size_t namelen = strlen(CLDeviceById(deviceId).getName()) + 1;
  if (namelen <= (size_t)len)
    memcpy(name, CLDeviceById(deviceId).getName(), namelen);
  else if (name && (len > 0))
    name[0] = 0;
  RETURN(hipSuccess);
}

hipError_t hipDeviceTotalMem(size_t *bytes, hipDevice_t deviceId) {
  InitializeOpenCL();
  ERROR_CHECK_DEVNUM(deviceId);

  if (bytes)
    *bytes = CLDeviceById(deviceId).getGlobalMemSize();
  RETURN(hipSuccess);
}

hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t *cacheConfig) {
  if (cacheConfig)
    *cacheConfig = hipFuncCachePreferNone;
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig *pConfig) {
  if (pConfig)
    *pConfig = hipSharedMemBankSizeFourByte;
  RETURN(hipSuccess);
}

hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig pConfig) {
  RETURN(hipSuccess);
}

hipError_t hipFuncSetCacheConfig(const void *func, hipFuncCache_t config) {
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetPCIBusId(char *pciBusId, int len, int device) {
  // TODO this requires OpenCL extension(s)
  RETURN(hipErrorInvalidDevice);
}

hipError_t hipDeviceGetByPCIBusId(int *device, const char *pciBusId) {
  // TODO this requires OpenCL extension(s)
  RETURN(hipErrorInvalidDevice);
}

hipError_t hipSetDeviceFlags(unsigned flags) {
  // TODO this requires OpenCL extension(s)
  return hipSuccess;
}

hipError_t hipChooseDevice(int *device, const hipDeviceProp_t *prop) {
  hipDeviceProp_t tempProp;
  ERROR_IF(((device == nullptr) || (prop == nullptr)), hipErrorInvalidValue);

  int inPropCount = 0;
  int matchedPropCount = 0;

  *device = 0;
  for (size_t i = 0; i < NumDevices; i++) {
    CLDeviceById(i).copyProperties(&tempProp);
    if (prop->major != 0) {
      inPropCount++;
      if (tempProp.major >= prop->major) {
        matchedPropCount++;
      }
      if (prop->minor != 0) {
        inPropCount++;
        if (tempProp.minor >= prop->minor) {
          matchedPropCount++;
        }
      }
    }
    if (prop->totalGlobalMem != 0) {
      inPropCount++;
      if (tempProp.totalGlobalMem >= prop->totalGlobalMem) {
        matchedPropCount++;
      }
    }
    if (prop->sharedMemPerBlock != 0) {
      inPropCount++;
      if (tempProp.sharedMemPerBlock >= prop->sharedMemPerBlock) {
        matchedPropCount++;
      }
    }
    if (prop->maxThreadsPerBlock != 0) {
      inPropCount++;
      if (tempProp.maxThreadsPerBlock >= prop->maxThreadsPerBlock) {
        matchedPropCount++;
      }
    }
    if (prop->totalConstMem != 0) {
      inPropCount++;
      if (tempProp.totalConstMem >= prop->totalConstMem) {
        matchedPropCount++;
      }
    }
    if (prop->multiProcessorCount != 0) {
      inPropCount++;
      if (tempProp.multiProcessorCount >= prop->multiProcessorCount) {
        matchedPropCount++;
      }
    }
    if (prop->maxThreadsPerMultiProcessor != 0) {
      inPropCount++;
      if (tempProp.maxThreadsPerMultiProcessor >=
          prop->maxThreadsPerMultiProcessor) {
        matchedPropCount++;
      }
    }
    if (prop->memoryClockRate != 0) {
      inPropCount++;
      if (tempProp.memoryClockRate >= prop->memoryClockRate) {
        matchedPropCount++;
      }
    }
    if (inPropCount == matchedPropCount) {
      *device = i;
      RETURN(hipSuccess);
    }
  }
  RETURN(hipErrorInvalidValue);
}

hipError_t hipDriverGetVersion(int *driverVersion) {
  if (driverVersion) {
    *driverVersion = 4;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipRuntimeGetVersion(int *runtimeVersion) {
  if (runtimeVersion) {
    *runtimeVersion = 1;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);
}

/********************************************************************/

hipError_t hipGetLastError(void) {
  hipError_t temp = tls_LastError;
  tls_LastError = hipSuccess;
  return temp;
}

hipError_t hipPeekAttls_LastError(void) { return tls_LastError; }

const char *hipGetErrorName(hipError_t hip_error) {
  switch (hip_error) {
  case hipSuccess:
    return "hipSuccess";
  case hipErrorOutOfMemory:
    return "hipErrorOutOfMemory";
  case hipErrorNotInitialized:
    return "hipErrorNotInitialized";
  case hipErrorDeinitialized:
    return "hipErrorDeinitialized";
  case hipErrorProfilerDisabled:
    return "hipErrorProfilerDisabled";
  case hipErrorProfilerNotInitialized:
    return "hipErrorProfilerNotInitialized";
  case hipErrorProfilerAlreadyStarted:
    return "hipErrorProfilerAlreadyStarted";
  case hipErrorProfilerAlreadyStopped:
    return "hipErrorProfilerAlreadyStopped";
  case hipErrorInvalidImage:
    return "hipErrorInvalidImage";
  case hipErrorInvalidContext:
    return "hipErrorInvalidContext";
  case hipErrorContextAlreadyCurrent:
    return "hipErrorContextAlreadyCurrent";
  case hipErrorMapFailed:
    return "hipErrorMapFailed";
  case hipErrorUnmapFailed:
    return "hipErrorUnmapFailed";
  case hipErrorArrayIsMapped:
    return "hipErrorArrayIsMapped";
  case hipErrorAlreadyMapped:
    return "hipErrorAlreadyMapped";
  case hipErrorNoBinaryForGpu:
    return "hipErrorNoBinaryForGpu";
  case hipErrorAlreadyAcquired:
    return "hipErrorAlreadyAcquired";
  case hipErrorNotMapped:
    return "hipErrorNotMapped";
  case hipErrorNotMappedAsArray:
    return "hipErrorNotMappedAsArray";
  case hipErrorNotMappedAsPointer:
    return "hipErrorNotMappedAsPointer";
  case hipErrorECCNotCorrectable:
    return "hipErrorECCNotCorrectable";
  case hipErrorUnsupportedLimit:
    return "hipErrorUnsupportedLimit";
  case hipErrorContextAlreadyInUse:
    return "hipErrorContextAlreadyInUse";
  case hipErrorPeerAccessUnsupported:
    return "hipErrorPeerAccessUnsupported";
  case hipErrorInvalidKernelFile:
    return "hipErrorInvalidKernelFile";
  case hipErrorInvalidGraphicsContext:
    return "hipErrorInvalidGraphicsContext";
  case hipErrorInvalidSource:
    return "hipErrorInvalidSource";
  case hipErrorFileNotFound:
    return "hipErrorFileNotFound";
  case hipErrorSharedObjectSymbolNotFound:
    return "hipErrorSharedObjectSymbolNotFound";
  case hipErrorSharedObjectInitFailed:
    return "hipErrorSharedObjectInitFailed";
  case hipErrorOperatingSystem:
    return "hipErrorOperatingSystem";
  case hipErrorSetOnActiveProcess:
    return "hipErrorSetOnActiveProcess";
  case hipErrorInvalidHandle:
    return "hipErrorInvalidHandle";
  case hipErrorNotFound:
    return "hipErrorNotFound";
  case hipErrorIllegalAddress:
    return "hipErrorIllegalAddress";

  case hipErrorMissingConfiguration:
    return "hipErrorMissingConfiguration";
  case hipErrorMemoryAllocation:
    return "hipErrorMemoryAllocation";
  case hipErrorInitializationError:
    return "hipErrorInitializationError";
  case hipErrorLaunchFailure:
    return "hipErrorLaunchFailure";
  case hipErrorPriorLaunchFailure:
    return "hipErrorPriorLaunchFailure";
  case hipErrorLaunchTimeOut:
    return "hipErrorLaunchTimeOut";
  case hipErrorLaunchOutOfResources:
    return "hipErrorLaunchOutOfResources";
  case hipErrorInvalidDeviceFunction:
    return "hipErrorInvalidDeviceFunction";
  case hipErrorInvalidConfiguration:
    return "hipErrorInvalidConfiguration";
  case hipErrorInvalidDevice:
    return "hipErrorInvalidDevice";
  case hipErrorInvalidValue:
    return "hipErrorInvalidValue";
  case hipErrorInvalidDevicePointer:
    return "hipErrorInvalidDevicePointer";
  case hipErrorInvalidMemcpyDirection:
    return "hipErrorInvalidMemcpyDirection";
  case hipErrorUnknown:
    return "hipErrorUnknown";
  case hipErrorInvalidResourceHandle:
    return "hipErrorInvalidResourceHandle";
  case hipErrorNotReady:
    return "hipErrorNotReady";
  case hipErrorNoDevice:
    return "hipErrorNoDevice";
  case hipErrorPeerAccessAlreadyEnabled:
    return "hipErrorPeerAccessAlreadyEnabled";

  case hipErrorPeerAccessNotEnabled:
    return "hipErrorPeerAccessNotEnabled";
  case hipErrorRuntimeMemory:
    return "hipErrorRuntimeMemory";
  case hipErrorRuntimeOther:
    return "hipErrorRuntimeOther";
  case hipErrorHostMemoryAlreadyRegistered:
    return "hipErrorHostMemoryAlreadyRegistered";
  case hipErrorHostMemoryNotRegistered:
    return "hipErrorHostMemoryNotRegistered";
  case hipErrorTbd:
    return "hipErrorTbd";
  default:
    return "hipErrorUnknown";
  }
}

const char *hipGetErrorString(hipError_t hipError) {
  return hipGetErrorName(hipError);
}

/********************************************************************/

hipError_t hipStreamCreate(hipStream_t *stream) {
  return hipStreamCreateWithFlags(stream, 0);
}

hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags) {
  return hipStreamCreateWithPriority(stream, flags, 0);
}

hipError_t hipStreamCreateWithPriority(hipStream_t *stream, unsigned int flags,
                                       int priority) {
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  // TODO priority & flags require an OpenCL extensions
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (cont->createQueue(stream, flags, priority))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipDeviceGetStreamPriorityRange(int *leastPriority,
                                           int *greatestPriority) {
  if (leastPriority)
    *leastPriority = 1;
  if (greatestPriority)
    *greatestPriority = 0;
  RETURN(hipSuccess);
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (cont->releaseQueue(stream))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipStreamQuery(hipStream_t stream) {
  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  // TODO requires OpenCL extension
  return hipSuccess;
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  stream->finish();
  RETURN(hipSuccess);
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                              unsigned int flags) {
  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (stream->enqueueBarrierForEvent(event))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags) {
  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  ERROR_IF((flags == nullptr), hipErrorInvalidValue);

  *flags = stream->getFlags();
  RETURN(hipSuccess);
}

hipError_t hipStreamGetPriority(hipStream_t stream, int *priority) {
  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  ERROR_IF((priority == nullptr), hipErrorInvalidValue);

  *priority = stream->getPriority();
  RETURN(hipSuccess);
}

hipError_t hipStreamAddCallback(hipStream_t stream,
                                hipStreamCallback_t callback, void *userData,
                                unsigned int flags) {
  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  ERROR_IF((callback == nullptr), hipErrorInvalidValue);

  if (stream->addCallback(callback, userData))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

/********************************************************************/

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags, hipDevice_t device) {

  ERROR_CHECK_DEVNUM(device);

  ClContext *cont = CLDeviceById(device).newContext(flags);
  // ClContext *cont = new ClContext(device);
  ERROR_IF((cont == nullptr), hipErrorOutOfMemory);

  // device->addContext(cont)
  *ctx = cont;
  tls_defaultCtx = cont;
  tls_getPrimaryCtx = false;
  tls_ctxStack.push(cont);
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDestroy(hipCtx_t ctx) {
  ClContext *primaryCtx = ctx->getDevice()->getPrimaryCtx();
  ERROR_IF((primaryCtx == ctx), hipErrorInvalidValue);

  ClContext *currentCtx = getTlsDefaultCtx();
  if (currentCtx == ctx) {
    // need to destroy the ctx associated with calling thread
    tls_ctxStack.pop();
  }

  ctx->getDevice()->removeContext(ctx);

  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPopCurrent(hipCtx_t *ctx) {
  ClContext *currentCtx = getTlsDefaultCtx();
  ClDevice *device = currentCtx->getDevice();
  *ctx = currentCtx;

  if (!tls_ctxStack.empty()) {
    tls_ctxStack.pop();
  }

  if (!tls_ctxStack.empty()) {
    currentCtx = tls_ctxStack.top();
  } else {
    currentCtx = device->getPrimaryCtx();
  }

  tls_defaultCtx = currentCtx;
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  hipError_t e = hipSuccess;
  if (ctx != nullptr) {
    tls_defaultCtx = ctx;
    tls_ctxStack.push(ctx);
    tls_getPrimaryCtx = false;
  } else {
    e = hipErrorInvalidContext;
  }
  RETURN(e);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  if (ctx == nullptr) {
    tls_ctxStack.pop();
  } else {
    tls_defaultCtx = ctx;
    tls_ctxStack.push(ctx);
    tls_getPrimaryCtx = false;
  }
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCurrent(hipCtx_t *ctx) {
  if ((tls_getPrimaryCtx) || tls_ctxStack.empty()) {
    *ctx = getTlsDefaultCtx();
  } else {
    *ctx = tls_ctxStack.top();
  }
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetDevice(hipDevice_t *device) {

  ClContext *ctx = getTlsDefaultCtx();

  ERROR_IF(((ctx == nullptr) || (device == nullptr)), hipErrorInvalidContext);

  ClDevice *dev = ctx->getDevice();
  *device = dev->getHipDeviceT();
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int *apiVersion) {
  if (apiVersion) {
    *apiVersion = 4;
  }
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCacheConfig(hipFuncCache_t *cacheConfig) {
  if (cacheConfig)
    *cacheConfig = hipFuncCachePreferNone;

  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig *pConfig) {
  if (pConfig)
    *pConfig = hipSharedMemBankSizeFourByte;
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSynchronize(void) {
  ClContext *ctx = getTlsDefaultCtx();
  ctx->finishAll();
  return hipSuccess;
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetFlags(unsigned int *flags) {
  ClContext *ctx = getTlsDefaultCtx();
  ERROR_IF((flags == nullptr), hipErrorInvalidValue);

  *flags = ctx->getFlags();
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
  RETURN(hipErrorInvalidValue);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
  RETURN(hipErrorInvalidValue);
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize,
                                 hipDeviceptr_t dptr) {

  hipCtx_t ctx = getTlsDefaultCtx();
  ERROR_IF((ctx == nullptr), hipErrorInvalidContext);

  if (ctx->findPointerInfo(dptr, pbase, psize))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t deviceId,
                                       unsigned int *flags, int *active) {
  ERROR_CHECK_DEVNUM(deviceId);

  ERROR_IF((flags == nullptr || active == nullptr), hipErrorInvalidValue);

  ClContext *currentCtx = getTlsDefaultCtx();
  ClContext *primaryCtx = CLDeviceById(deviceId).getPrimaryCtx();

  *active = (primaryCtx == currentCtx) ? 1 : 0;
  *flags = primaryCtx->getFlags();

  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t deviceId) {
  ERROR_CHECK_DEVNUM(deviceId);
  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t *pctx, hipDevice_t deviceId) {
  ERROR_CHECK_DEVNUM(deviceId);
  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t deviceId) {
  ERROR_CHECK_DEVNUM(deviceId);

  CLDeviceById(deviceId).getPrimaryCtx()->reset();

  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t deviceId,
                                       unsigned int flags) {
  ERROR_CHECK_DEVNUM(deviceId);

  RETURN(hipErrorContextAlreadyInUse);
}

/********************************************************************/

hipError_t hipEventCreate(hipEvent_t *event) {
  return hipEventCreateWithFlags(event, 0);
}

hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned flags) {
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipEvent_t EventPtr = cont->createEvent(flags);
  if (EventPtr) {
    *event = EventPtr;
    RETURN(hipSuccess);
  } else {
    RETURN(hipErrorOutOfMemory);
  }
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->recordEvent(stream, event));
}

hipError_t hipEventDestroy(hipEvent_t event) {
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  delete event;
  RETURN(hipSuccess);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (event->wait())
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  ERROR_IF((start == nullptr), hipErrorInvalidValue);
  ERROR_IF((stop == nullptr), hipErrorInvalidValue);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->eventElapsedTime(ms, start, stop));
}

hipError_t hipEventQuery(hipEvent_t event) {
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (event->isFinished())
    RETURN(hipSuccess);
  else
    RETURN(hipErrorNotReady);
}

/********************************************************************/

hipError_t hipMalloc(void **ptr, size_t size) {

  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  if (size == 0) {
    *ptr = nullptr;
    RETURN(hipSuccess);
  }

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  void *retval = cont->allocate(size);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
}

DEPRECATED("use hipHostMalloc instead")
hipError_t hipMallocHost(void **ptr, size_t size) {
  return hipMalloc(ptr, size);
}

DEPRECATED("use hipHostMalloc instead")
hipError_t hipHostAlloc(void **ptr, size_t size, unsigned int flags) {
  return hipMalloc(ptr, size);
}

hipError_t hipFree(void *ptr) {

  ERROR_IF((ptr == nullptr), hipSuccess);
  //  if (ptr == nullptr)
  //    RETURN(hipSuccess);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (cont->free(ptr))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidDevicePointer);
}

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  return hipMalloc(ptr, size);
}

hipError_t hipHostFree(void *ptr) { return hipFree(ptr); }

DEPRECATED("use hipHostFree instead")
hipError_t hipFreeHost(void *ptr) { return hipHostFree(ptr); }

hipError_t hipHostGetDevicePointer(void **devPtr, void *hstPtr,
                                   unsigned int flags) {
  ERROR_IF(((hstPtr == nullptr) || (devPtr == nullptr)), hipErrorInvalidValue);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (!cont->hasPointer(hstPtr))
    RETURN(hipErrorInvalidDevicePointer);

  *devPtr = hstPtr;
  RETURN(hipSuccess);
}

hipError_t hipHostGetFlags(unsigned int *flagsPtr, void *hostPtr) {
  // TODO dummy implementation
  *flagsPtr = 0;
  RETURN(hipSuccess);
}

hipError_t hipHostRegister(void *hostPtr, size_t sizeBytes,
                           unsigned int flags) {
  RETURN(hipSuccess);
}

hipError_t hipHostUnregister(void *hostPtr) { RETURN(hipSuccess); }

static hipError_t hipMallocPitch3D(void **ptr, size_t *pitch, size_t width,
                                   size_t height, size_t depth) {
  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  *pitch = ((((int)width - 1) / SVM_ALIGNMENT) + 1) * SVM_ALIGNMENT;
  const size_t sizeBytes = (*pitch) * height * ((depth == 0) ? 1 : depth);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  void *retval = cont->allocate(sizeBytes);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
}

hipError_t hipMallocPitch(void **ptr, size_t *pitch, size_t width,
                          size_t height) {
  return hipMallocPitch3D(ptr, pitch, width, height, 0);
}

hipError_t hipMallocArray(hipArray **array, const hipChannelFormatDesc *desc,
                          size_t width, size_t height, unsigned int flags) {

  ERROR_IF((width == 0), hipErrorInvalidValue);

  auto cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidContext);

  *array = new hipArray;
  ERROR_IF((*array == nullptr), hipErrorOutOfMemory);

  array[0]->type = flags;
  array[0]->width = width;
  array[0]->height = height;
  array[0]->depth = 1;
  array[0]->desc = *desc;
  array[0]->isDrv = false;
  array[0]->textureType = hipTextureType2D;
  void **ptr = &array[0]->data;

  size_t size = width;
  if (height > 0) {
    size = size * height;
  }
  const size_t allocSize = size * ((desc->x + desc->y + desc->z + desc->w) / 8);

  void *retval = cont->allocate(allocSize);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
}

hipError_t hipArrayCreate(hipArray **array,
                          const HIP_ARRAY_DESCRIPTOR *pAllocateArray) {
  ERROR_IF((pAllocateArray->width == 0), hipErrorInvalidValue);

  auto cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidContext);

  *array = new hipArray;
  ERROR_IF((*array == nullptr), hipErrorOutOfMemory);

  array[0]->drvDesc = *pAllocateArray;
  array[0]->width = pAllocateArray->width;
  array[0]->height = pAllocateArray->height;
  array[0]->isDrv = true;
  array[0]->textureType = hipTextureType2D;
  void **ptr = &array[0]->data;

  size_t size = pAllocateArray->width;
  if (pAllocateArray->height > 0) {
    size = size * pAllocateArray->height;
  }
  size_t allocSize = 0;
  switch (pAllocateArray->format) {
  case HIP_AD_FORMAT_UNSIGNED_INT8:
    allocSize = size * sizeof(uint8_t);
    break;
  case HIP_AD_FORMAT_UNSIGNED_INT16:
    allocSize = size * sizeof(uint16_t);
    break;
  case HIP_AD_FORMAT_UNSIGNED_INT32:
    allocSize = size * sizeof(uint32_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT8:
    allocSize = size * sizeof(int8_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT16:
    allocSize = size * sizeof(int16_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT32:
    allocSize = size * sizeof(int32_t);
    break;
  case HIP_AD_FORMAT_HALF:
    allocSize = size * sizeof(int16_t);
    break;
  case HIP_AD_FORMAT_FLOAT:
    allocSize = size * sizeof(float);
    break;
  default:
    allocSize = size;
    break;
  }

  void *retval = cont->allocate(allocSize);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
}

hipError_t hipFreeArray(hipArray *array) {
  ERROR_IF((array == nullptr), hipErrorInvalidValue);

  assert(array->data != nullptr);

  hipError_t e = hipFree(array->data);

  delete array;

  return e;
}

hipError_t hipMalloc3D(hipPitchedPtr *pitchedDevPtr, hipExtent extent) {

  ERROR_IF((extent.width == 0 || extent.height == 0), hipErrorInvalidValue);
  ERROR_IF((pitchedDevPtr == nullptr), hipErrorInvalidValue);

  size_t pitch;

  hipError_t hip_status = hipMallocPitch3D(
      &pitchedDevPtr->ptr, &pitch, extent.width, extent.height, extent.depth);

  if (hip_status == hipSuccess) {
    pitchedDevPtr->pitch = pitch;
    pitchedDevPtr->xsize = extent.width;
    pitchedDevPtr->ysize = extent.height;
  }
  RETURN(hip_status);
}

hipError_t hipMemGetInfo(size_t *free, size_t *total) {

  ERROR_IF((total == nullptr || free == nullptr), hipErrorInvalidValue);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  auto device = cont->getDevice();
  *total = device->getGlobalMemSize();
  *free = device->getUsedGlobalMem();

  RETURN(hipSuccess);
}

hipError_t hipMemPtrGetInfo(void *ptr, size_t *size) {

  ERROR_IF((ptr == nullptr || size == nullptr), hipErrorInvalidValue);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (cont->getPointerSize(ptr, size))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

/********************************************************************/

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if ((kind == hipMemcpyDeviceToDevice) || (kind == hipMemcpyDeviceToHost)) {
    if (!cont->hasPointer(src))
      RETURN(hipErrorInvalidDevicePointer);
  }

  if ((kind == hipMemcpyDeviceToDevice) || (kind == hipMemcpyHostToDevice)) {
    if (!cont->hasPointer(dst))
      RETURN(hipErrorInvalidDevicePointer);
  }

  if (kind == hipMemcpyHostToHost) {
    memcpy(dst, src, sizeBytes);
    RETURN(hipSuccess);
  } else {
    RETURN(cont->memCopy(dst, src, sizeBytes, stream));
  }
}

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipError_t e =
      hipMemcpyAsync(dst, src, sizeBytes, kind, cont->getDefaultQueue());
  if (e != hipSuccess)
    return e;

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src,
                              size_t sizeBytes, hipStream_t stream) {
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream);
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src,
                         size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice);
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void *src, size_t sizeBytes,
                              hipStream_t stream) {
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream);
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void *src, size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToDevice);
}

hipError_t hipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream);
}

hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToHost);
}

/********************************************************************/

hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                             hipStream_t stream) {
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->memFill(dst, 4 * count, &value, 4, stream));
}

hipError_t hipMemsetD32(hipDeviceptr_t dst, int value, size_t count) {

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipError_t e = hipMemsetD32Async(dst, value, count, cont->getDefaultQueue());
  if (e != hipSuccess)
    return e;

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemset2DAsync(void *dst, size_t pitch, int value, size_t width,
                            size_t height, hipStream_t stream) {

  size_t sizeBytes = pitch * height;
  return hipMemsetAsync(dst, value, sizeBytes, stream);
}

hipError_t hipMemset2D(void *dst, size_t pitch, int value, size_t width,
                       size_t height) {
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipError_t e = hipMemset2DAsync(dst, pitch, value, width, height,
                                  cont->getDefaultQueue());
  if (e != hipSuccess)
    return e;

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value,
                            hipExtent extent, hipStream_t stream) {

  size_t sizeBytes = pitchedDevPtr.pitch * extent.height * extent.depth;
  return hipMemsetAsync(pitchedDevPtr.ptr, value, sizeBytes, stream);
}

hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value,
                       hipExtent extent) {

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipError_t e =
      hipMemset3DAsync(pitchedDevPtr, value, extent, cont->getDefaultQueue());
  if (e != hipSuccess)
    return e;

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes,
                          hipStream_t stream) {
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  char c_value = value;
  RETURN(cont->memFill(dst, sizeBytes, &c_value, 1, stream));
}

hipError_t hipMemset(void *dst, int value, size_t sizeBytes) {
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipError_t e = hipMemsetAsync(dst, value, sizeBytes, cont->getDefaultQueue());
  if (e != hipSuccess)
    return e;

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value,
                       size_t sizeBytes) {
  return hipMemset(dest, value, sizeBytes);
}

/********************************************************************/

hipError_t hipMemcpyParam2D(const hip_Memcpy2D *pCopy) {
  ERROR_IF((pCopy == nullptr), hipErrorInvalidValue);

  return hipMemcpy2D(pCopy->dstArray->data, pCopy->widthInBytes, pCopy->srcHost,
                     pCopy->srcPitch, pCopy->widthInBytes, pCopy->height,
                     hipMemcpyDefault);
}

hipError_t hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                            size_t spitch, size_t width, size_t height,
                            hipMemcpyKind kind, hipStream_t stream) {
  if (spitch == 0)
    spitch = width;
  if (dpitch == 0)
    dpitch = width;

  if (spitch == 0 || dpitch == 0)
    RETURN(hipErrorInvalidValue);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  for (size_t i = 0; i < height; ++i) {
    if (kind == hipMemcpyHostToHost) {
      memcpy(dst, src, width);
    } else {
      if (cont->memCopy(dst, src, width, stream) != hipSuccess)
        RETURN(hipErrorLaunchFailure);
    }
    src = (char *)src + spitch;
    dst = (char *)dst + dpitch;
  }
  RETURN(hipSuccess);
}

hipError_t hipMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width, size_t height, hipMemcpyKind kind) {
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipError_t e = hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind,
                                  cont->getDefaultQueue());
  if (e != hipSuccess)
    return e;

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemcpy2DToArray(hipArray *dst, size_t wOffset, size_t hOffset,
                              const void *src, size_t spitch, size_t width,
                              size_t height, hipMemcpyKind kind) {
  size_t byteSize;
  if (dst) {
    switch (dst[0].desc.f) {
    case hipChannelFormatKindSigned:
      byteSize = sizeof(int);
      break;
    case hipChannelFormatKindUnsigned:
      byteSize = sizeof(unsigned int);
      break;
    case hipChannelFormatKindFloat:
      byteSize = sizeof(float);
      break;
    case hipChannelFormatKindNone:
      byteSize = sizeof(size_t);
      break;
    }
  } else {
    RETURN(hipErrorUnknown);
  }

  if ((wOffset + width > (dst->width * byteSize)) || width > spitch) {
    RETURN(hipErrorInvalidValue);
  }

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  size_t src_w = spitch;
  size_t dst_w = (dst->width) * byteSize;

  for (size_t i = 0; i < height; ++i) {
    void *dst_p = ((unsigned char *)dst->data + i * dst_w);
    void *src_p = ((unsigned char *)src + i * src_w);

    if (kind == hipMemcpyHostToHost) {
      memcpy(dst_p, src_p, width);
    } else {
      if (cont->memCopy(dst_p, src_p, width, cont->getDefaultQueue()) !=
          hipSuccess)
        RETURN(hipErrorLaunchFailure);
    }
  }

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemcpyToArray(hipArray *dst, size_t wOffset, size_t hOffset,
                            const void *src, size_t count, hipMemcpyKind kind) {

  void *dst_p = (unsigned char *)dst->data + wOffset;

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (kind == hipMemcpyHostToHost) {
    memcpy(dst_p, src, count);
  } else {
    if (cont->memCopy(dst_p, src, count, cont->getDefaultQueue()) != hipSuccess)
      RETURN(hipErrorLaunchFailure);
  }

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemcpyFromArray(void *dst, hipArray_const_t srcArray,
                              size_t wOffset, size_t hOffset, size_t count,
                              hipMemcpyKind kind) {

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  void *src_p = (unsigned char *)srcArray->data + wOffset;

  if (kind == hipMemcpyHostToHost) {
    memcpy(dst, src_p, count);
  } else {
    if (cont->memCopy(dst, src_p, count, cont->getDefaultQueue()) != hipSuccess)
      RETURN(hipErrorLaunchFailure);
  }

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemcpyAtoH(void *dst, hipArray *srcArray, size_t srcOffset,
                         size_t count) {
  return hipMemcpy((char *)dst, (char *)srcArray->data + srcOffset, count,
                   hipMemcpyDeviceToHost);
}

hipError_t hipMemcpyHtoA(hipArray *dstArray, size_t dstOffset,
                         const void *srcHost, size_t count) {
  return hipMemcpy((char *)dstArray->data + dstOffset, srcHost, count,
                   hipMemcpyHostToDevice);
}

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *p) {

  ERROR_IF((p == nullptr), hipErrorInvalidValue);

  size_t byteSize;
  size_t depth;
  size_t height;
  size_t widthInBytes;
  size_t srcPitch;
  size_t dstPitch;
  void *srcPtr;
  void *dstPtr;
  size_t ySize;

  if (p->dstArray != nullptr) {
    if (p->dstArray->isDrv == false) {
      switch (p->dstArray->desc.f) {
      case hipChannelFormatKindSigned:
        byteSize = sizeof(int);
        break;
      case hipChannelFormatKindUnsigned:
        byteSize = sizeof(unsigned int);
        break;
      case hipChannelFormatKindFloat:
        byteSize = sizeof(float);
        break;
      case hipChannelFormatKindNone:
        byteSize = sizeof(size_t);
        break;
      }
      depth = p->extent.depth;
      height = p->extent.height;
      widthInBytes = p->extent.width * byteSize;
      srcPitch = p->srcPtr.pitch;
      srcPtr = p->srcPtr.ptr;
      ySize = p->srcPtr.ysize;
      dstPitch = p->dstArray->width * byteSize;
      dstPtr = p->dstArray->data;
    } else {
      depth = p->Depth;
      height = p->Height;
      widthInBytes = p->WidthInBytes;
      dstPitch = p->dstArray->width * 4;
      srcPitch = p->srcPitch;
      srcPtr = (void *)p->srcHost;
      ySize = p->srcHeight;
      dstPtr = p->dstArray->data;
    }
  } else {
    // Non array destination
    depth = p->extent.depth;
    height = p->extent.height;
    widthInBytes = p->extent.width;
    srcPitch = p->srcPtr.pitch;
    srcPtr = p->srcPtr.ptr;
    dstPtr = p->dstPtr.ptr;
    ySize = p->srcPtr.ysize;
    dstPitch = p->dstPtr.pitch;
  }

  if ((widthInBytes == dstPitch) && (widthInBytes == srcPitch)) {
    return hipMemcpy((void *)dstPtr, (void *)srcPtr,
                     widthInBytes * height * depth, p->kind);
  } else {

    ClContext *cont = getTlsDefaultCtx();
    ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

    for (size_t i = 0; i < depth; i++) {
      for (size_t j = 0; j < height; j++) {

        unsigned char *src =
            (unsigned char *)srcPtr + i * ySize * srcPitch + j * srcPitch;
        unsigned char *dst =
            (unsigned char *)dstPtr + i * height * dstPitch + j * dstPitch;
        hipMemcpyAsync(dst, src, widthInBytes, p->kind,
                       cont->getDefaultQueue());
      }
    }

    cont->getDefaultQueue()->finish();
    RETURN(hipSuccess);
  }
}

/********************************************************************/

hipError_t hipInit(unsigned int flags) {
  InitializeOpenCL();
  RETURN(hipSuccess);
}

/********************************************************************/

hipError_t hipFuncGetAttributes(hipFuncAttributes *attr, const void *func) {
  logError("hipFuncGetAttributes not implemented \n");
  RETURN(hipErrorInvalidValue);
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes,
                              hipModule_t hmod, const char *name) {
  // TODO global variable support will require some Clang changes
  logError("Global variables are not supported ATM\n");
  return hipErrorNotFound;
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
  logError("hipModuleLoadData not implemented\n");
  return hipErrorNoBinaryForGpu;
}

hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image,
                               unsigned int numOptions, hipJitOption *options,
                               void **optionValues) {
  return hipModuleLoadData(module, image);
}

hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                            hipStream_t stream) {

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->configureCall(gridDim, blockDim, sharedMem, stream));
}

hipError_t hipSetupArgument(const void *arg, size_t size, size_t offset) {

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->setArg(arg, size, offset));
}

hipError_t hipLaunchByPtr(const void *hostFunction) {
  logDebug("hipLaunchByPtr\n");
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->launchHostFunc(hostFunction));
}

/********************************************************************/

hipError_t hipModuleLoad(hipModule_t *module, const char *fname) {

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  std::ifstream file(fname, std::ios::in | std::ios::binary | std::ios::ate);
  ERROR_IF((file.fail()), hipErrorFileNotFound);

  size_t size = file.tellg();
  char *memblock = new char[size];
  file.seekg(0, std::ios::beg);
  file.read(memblock, size);
  file.close();
  std::string content(memblock, size);
  delete[] memblock;

  *module = cont->createProgram(content);
  if (*module == nullptr)
    RETURN(hipErrorInvalidValue);
  else
    RETURN(hipSuccess);
}

hipError_t hipModuleUnload(hipModule_t module) {
  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->destroyProgram(module));
}

hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module,
                                const char *kname) {

  ClProgram *p = (ClProgram *)module;
  ClKernel *k = p->getKernel(kname);

  ERROR_IF((k == nullptr), hipErrorInvalidDeviceFunction);

  *function = k;
  RETURN(hipSuccess);
}

hipError_t hipModuleLaunchKernel(hipFunction_t k, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 hipStream_t stream, void **kernelParams,
                                 void **extra) {

  logDebug("hipModuleLaunchKernel\n");

  if (sharedMemBytes > 0) {
    logError("Dynamic shared memory isn't supported ATM\n");
    RETURN(hipErrorLaunchFailure);
  }

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (kernelParams == nullptr && extra == nullptr) {
    logError("either kernelParams or extra is required!\n");
    RETURN(hipErrorLaunchFailure);
  }

  dim3 grid(gridDimX, gridDimY, gridDimZ);
  dim3 block(blockDimX, blockDimY, blockDimZ);

  if (kernelParams)
    RETURN(cont->launchWithKernelParams(grid, block, sharedMemBytes, stream,
                                        kernelParams, k));
  else
    RETURN(cont->launchWithExtraParams(grid, block, sharedMemBytes, stream,
                                       extra, k));
}

/*******************************************************************************/

#include "hip_fatbin.h"

#define SPIR_TRIPLE "hip-spir64-unknown-unknown"

static unsigned binaries_loaded = 0;

extern "C" void **__hipRegisterFatBinary(const void *data) {
  InitializeOpenCL();

  const __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<const __CudaFatBinaryWrapper *>(data);
  if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
    logCritical("The given object is not hipFatBinary !\n");
    std::abort();
  }

  const __ClangOffloadBundleHeader *header = fbwrapper->binary;
  std::string magic(reinterpret_cast<const char *>(header),
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC)) {
    logCritical("The bundled binaries are not Clang bundled "
                "(CLANG_OFFLOAD_BUNDLER_MAGIC is missing)\n");
    std::abort();
  }

  std::string *module = new std::string;
  if (!module) {
    logCritical("Failed to allocate memory\n");
    std::abort();
  }

  const __ClangOffloadBundleDesc *desc = &header->desc[0];
  bool found = false;

  for (uint64_t i = 0; i < header->numBundles;
       ++i, desc = reinterpret_cast<const __ClangOffloadBundleDesc *>(
                reinterpret_cast<uintptr_t>(&desc->triple[0]) +
                desc->tripleSize)) {

    std::string triple{&desc->triple[0], sizeof(SPIR_TRIPLE) - 1};
    logDebug("Triple of bundle {} is: {}\n", i, triple);

    if (triple.compare(SPIR_TRIPLE) == 0) {
      found = true;
      break;
    } else {
      logDebug("not a SPIR triple, ignoring\n");
      continue;
    }
  }

  if (!found) {
    logDebug("Didn't find any suitable compiled binary!\n");
    std::abort();
  }

  const char *string_data = reinterpret_cast<const char *>(
      reinterpret_cast<uintptr_t>(header) + (uintptr_t)desc->offset);
  size_t string_size = desc->size;
  module->assign(string_data, string_size);

  logDebug("Register module: {} \n", (void *)module);

  for (size_t deviceId = 0; deviceId < NumDevices; ++deviceId) {
    CLDeviceById(deviceId).registerModule(module);
  }

  ++binaries_loaded;
  logDebug("__hipRegisterFatBinary {}\n", binaries_loaded);

  return (void **)module;
}

extern "C" void __hipUnregisterFatBinary(void *data) {
  std::string *module = reinterpret_cast<std::string *>(data);

  logDebug("Unregister module: {} \n", (void *)module);
  for (size_t deviceId = 0; deviceId < NumDevices; ++deviceId) {
    CLDeviceById(deviceId).unregisterModule(module);
  }

  --binaries_loaded;
  logDebug("__hipUnRegisterFatBinary {}\n", binaries_loaded);

  if (binaries_loaded == 0) {
    UnInitializeOpenCL();
  }

  delete module;
}

extern "C" void __hipRegisterFunction(void **data, const void *hostFunction,
                                      char *deviceFunction,
                                      const char *deviceName,
                                      unsigned int threadLimit, void *tid,
                                      void *bid, dim3 *blockDim, dim3 *gridDim,
                                      int *wSize) {
  InitializeOpenCL();

  std::string *module = reinterpret_cast<std::string *>(data);
  logDebug("RegisterFunction on module {}\n", (void *)module);

  for (size_t deviceId = 0; deviceId < NumDevices; ++deviceId) {

    if (CLDeviceById(deviceId).registerFunction(module, hostFunction,
                                                deviceName)) {
      logDebug("__hipRegisterFunction: kernel {} found\n", deviceName);
    } else {
      logCritical("__hipRegisterFunction can NOT find kernel: {} \n",
                  deviceName);
      std::abort();
    }
  }
}

extern "C" void __hipRegisterVar(std::vector<hipModule_t> *modules,
                                 char *hostVar, char *deviceVar,
                                 const char *deviceName, int ext, int size,
                                 int constant, int global) {
  logError("__hipRegisterVar not implemented yet\n");
  InitializeOpenCL();
}
