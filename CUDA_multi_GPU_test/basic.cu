#include "main.cuh"

void fCUDA::getDeviceName() {
	printf("CUDA Device Info\n");

	int deviceCount = 0;
	cudaError_t error_id;

	error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("Error cudaGetDeviceCount() : %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
		exit(EXIT_FAILURE);
	}
	else 
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);

	cudaDeviceProp deviceProp;

	// Iterate through all the devices found
	for (int i = 0; i < deviceCount; ++i) {
		cudaGetDeviceProperties(&deviceProp, i);
		printf("Device: %d, %s\n\n", i, deviceProp.name);
	}
}

void fCUDA::getDeviceInfo(size_t deviceID)
{
	cudaSetDevice(deviceID);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
    printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total shared memory per multiprocessor:        %zu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %zu bytes\n",
           deviceProp.memPitch);
    printf("  Texture alignment:                             %zu bytes\n",
           deviceProp.textureAlignment);
    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
    printf("  Run time limit on kernels:                     %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n",
           deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("  Alignment requirement for Surfaces:            %s\n",
           deviceProp.surfaceAlignment ? "Yes" : "No");
    printf("  Device has ECC support:                        %s\n",
           deviceProp.ECCEnabled ? "Enabled" : "Disabled");
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           deviceProp.unifiedAddressing ? "Yes" : "No");
    printf("  Device supports Managed Memory:                %s\n",
           deviceProp.managedMemory ? "Yes" : "No");
    printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
    printf("  Supports Cooperative Kernel Launch:            %s\n",
           deviceProp.cooperativeLaunch ? "Yes" : "No");
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
}

void fCUDA::setGPUDirect() {
	int deviceCount = 0;
	cudaError_t error_id;
       int canAccessPeer = 0;

	error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("Error cudaGetDeviceCount() : %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

       for (size_t i = 0; i < deviceCount; ++i) {
              cudaSetDevice(i);
              for (size_t j = 0; j < deviceCount; ++j) {
                     if (i == j)
                            continue;
                     cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                     if (canAccessPeer) {
                            printf("GPU #%zu and GPU #%zu: Start Setup\n", i, j);
                            cudaDeviceEnablePeerAccess(j, 0);
                     }
                     else {
                            printf("GPU #%zu and GPU #%zu: Not Supported\n", i, j);
                     }
              }
       }
}

template <int div>
__global__ void fCUDA::dGPUDiv(unsigned int size, const unsigned char* __restrict__ input, unsigned char* __restrict__ output) {
       const unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

       if (globalIdx < size) {
              output[globalIdx] = input[globalIdx] / div;
       }
}

struct char_vec
{
public:
       static constexpr int size = 8;
       unsigned char data[size];

       __device__ char_vec (const unsigned char* __restrict__ input) {
              *reinterpret_cast<int2*>(data) = *reinterpret_cast<const int2*>(input);
       }

       __device__  void store (unsigned char* output) const {
              *reinterpret_cast<int2*>(output) = *reinterpret_cast<const int2*>(data);
       }
};

template <int div>
__global__ void fCUDA::dGPUDiv_opt(const unsigned char* __restrict__ input, unsigned char* __restrict__ output) {
       auto globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

       char_vec vec(input + globalIdx * 8);

       unsigned char *data = vec.data;
       for (int i = 0; i < vec.size; ++i)
              data[i] /= div;

       vec.store(output + globalIdx * 8);
}

void fTEST::fillArray(unsigned int* data, const unsigned int size) {
       for (unsigned int i = 0; i < size; ++i) {
              data[i] = i;
       }
}

void fTEST::checkArray(char* dPrefix, unsigned int* data, const unsigned int size) {
       bool error = false;
       for (unsigned int i = 0; i < size; ++i) {
              if (data[i] != (i * 2)) {
                     printf("%s Error : %u %u", dPrefix, i, data[i]);
                     error = true;
              }
       }
       if (error == false) {
              printf("%s Array check passed\n", dPrefix);
       }
}

__global__ void fCUDA::dGPUTestKernel(unsigned int* data) {
       const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
       
       data[globalIdx] *= 2;
}

double fTEST::measure_allocation_time(unsigned int max_allocations)
{
       std::vector<char*> pointers(max_allocations, nullptr);

       const auto begin = std::chrono::high_resolution_clock::now();
       for (unsigned int alloc_id = 0; alloc_id < max_allocations; ++alloc_id)
              cudaMalloc(&pointers[alloc_id], 2 * 1024 * 1024);
       const auto end = std::chrono::high_resolution_clock::now();

       for (auto &ptr : pointers)
              cudaFree(ptr);

       return std::chrono::duration<double> (end - begin).count();
}

void fTEST::p2p_alloc_bench()
{
  const unsigned int allocations_count = 1000;
  const double basic_time = measure_allocation_time (allocations_count);
  cudaDeviceEnablePeerAccess (1, 0);
  const double p2p_time = measure_allocation_time (allocations_count);

  std::cout << basic_time << " s => " << p2p_time << std::endl;
}
