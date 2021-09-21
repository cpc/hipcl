#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <limits.h>
#include <unistd.h>

#define NUM 100

#define CHECK(cmd)                                                                   \
  do {                                                                               \
    hipError_t error = (cmd);                                                        \
    if (error != hipSuccess) {                                                       \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, \
              __FILE__, __LINE__);                                                   \
      exit(1);                                                                       \
    }                                                                                \
  } while(0)


using namespace std;

int main(int argc, char* argv[])
{
  // set up arrays for vector add
  int i=0;
  float* hostA;
  float* hostB;
  float* hostC;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  struct {
    size_t _n;
    void* _Ad;
    void* _Bd;
    void* _Cd;
  } args1;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(NUM * sizeof(float));
  hostC = (float*)malloc(NUM * sizeof(float));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostA[i] = (float)i;
    hostB[i] = (float)i;
  }

  CHECK(hipInit(0));
  CHECK(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  CHECK(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  CHECK(hipMalloc((void**)&deviceC, NUM * sizeof(float)));

  CHECK(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(deviceA, hostA, NUM*sizeof(float), hipMemcpyHostToDevice));

  hipModule_t hipModule = NULL;
  hipError_t error;

  char result[ PATH_MAX ];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  std::string executablePath( result, (count > 0) ? count : 0 );
  size_t last_pos = executablePath.find_last_of("/");
  if (last_pos == std::string::npos)
    executablePath.assign("./");
  else
    executablePath.resize(last_pos+1);
  const std::string  binaryFilename(executablePath + "hipModuleLoadBinary");

  error = hipModuleLoad(&hipModule, binaryFilename.c_str());
  if (error) {
    printf("%s\n",  binaryFilename.c_str());
    cout << "Loading Module ("+binaryFilename+")" << endl;
    exit(1);
  }

  // get the function from the module
  hipFunction_t hipFunction = NULL;
  error = hipModuleGetFunction(&hipFunction, hipModule, "_occa_addVectors_0");
  if (error) {
    cout << "Getting Function (_occa_addVectors_0)" << endl;
    exit(1);
  }

  args1._n = NUM;
  args1._Ad = deviceA;
  args1._Bd = deviceB;
  args1._Cd = deviceC;

  size_t size = sizeof(args1);

  void *config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
    HIP_LAUNCH_PARAM_END
  };

  // launch the function
  error = hipModuleLaunchKernel( hipFunction, 1, 1, 1, NUM, 1, 1, 0, NULL, NULL,
                                 reinterpret_cast<void**>(&config) );
  if (error) {
    cout << "hipmodulelaunch error" << endl;
    exit(1);
  }

  CHECK(hipMemcpy(hostC, deviceC, NUM*sizeof(float), hipMemcpyDeviceToHost));

  // verify the results
  int errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostC[i] != (hostB[i] + hostA[i])) {
      printf( "%f\n", hostC[i]);
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
    printf("PASSED!\n");
  }

  CHECK(hipFree(deviceA));
  CHECK(hipFree(deviceB));
  CHECK(hipFree(deviceC));

  return 0;
}
