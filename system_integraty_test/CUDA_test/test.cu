#include <stdio.h>
#include <stdlib.h>

#include "deviceQuery.h"

#include <thread>
#include <atomic>

#define DATA_SIZE (1024*1024*16)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void device_vector_adder(int *a, int *b, int *c) {
    int x_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_global_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int z_global_idx = blockIdx.z * blockDim.z + threadIdx.z;

    int x_global_size = gridDim.x * blockDim.x;
    int y_global_size = gridDim.y * blockDim.y;
    int z_global_size = gridDim.z * blockDim.z;
    // printf("Global ID: [x, y, z] = [%d, %d, %d]\n", x_global_idx, y_global_idx, z_global_idx);

    if(x_global_idx + y_global_idx + z_global_idx == 0) {
        printf("\nGlobal Size: [x, y, z] = [%d, %d, %d]\n", x_global_size, y_global_size, z_global_size);
    }

    int global_idx = z_global_idx * (x_global_size * y_global_size) + y_global_idx * x_global_size + x_global_idx;

    for (int i = 0; i < 20; ++i)
        c[global_idx] += a[global_idx] + b[global_idx];
}

__global__ void device_local_checker() {
    __shared__ int sharedMemory [32];

    sharedMemory[threadIdx.x] = 255;
    __syncthreads();
    if (sharedMemory[threadIdx.x] != 255)
        printf("Shared Memory %d Failed\n", threadIdx.x);
    else
        printf("Shared Memory %d Passed\n", threadIdx.x);
}

void host_initializer(int *data, const int &value) {
    for (int idx = 0; idx < DATA_SIZE; ++idx)
        data[idx] = value;
}

bool host_checker(int *data, const int &value) {
    for (int idx = 0; idx < DATA_SIZE; ++idx) {
        if(data[idx] != value)
            return false;
    }
    return true;
}

int main(void)
{
    printf("\n\n------------------------------ [A] CPU Setup Start ----------------------------\n\n");

    int *a, *b, *c, *c2;
    int size = DATA_SIZE * sizeof(int);

    printf("[0] Setup CPU Memory:\n\n");

    cudaError_t status = cudaMallocHost((void**)&a, size);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");
    status = cudaMallocHost((void**)&b, size);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");
    status = cudaMallocHost((void**)&c, size);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    status = cudaMallocHost((void**)&c2, size);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    printf("[1] Start host initializer:");
    static const int value_a = 4, value_b = 3;
    host_initializer(a, value_a);
    host_initializer(b, value_b);

    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    printf("\n\n------------------------------ [A] CPU Setup End ----------------------------\n\n");

    // GPU Info Test

    printf("\n\n------------------------------ [B] GPU Info Test Start ----------------------------\n\n");
    deviceQuery_func();
    printf("\n\n------------------------------ [B] GPU Info Test End ------------------------------\n\n");

    // GPU Comput Test

    printf("\n\n------------------------------ [0] GPU Compute Test Start ------------------------------\n\n");
    {
        printf("\n\n------------------------------ [C] GPU Setup Start ------------------------------\n\n");
        dim3 dimBlock(1, 1, 1); // local work size
        dim3 dimGrid(256, 256, 256); // local work group size

        printf("[0] Setup GPU Memory:\n\n");
        int *d_a, *d_b, *d_c;

        // allocate space for device copies
        gpuErrchk(cudaMalloc((void **)&d_a, size));
        gpuErrchk(cudaMalloc((void **)&d_b, size));
        gpuErrchk(cudaMalloc((void **)&d_c, size));

        printf("[1] Start GPU Memory transfer:");

        // copy inputs to device
        gpuErrchk(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
        printf("\n\n------------------------------ [C] GPU Setup End ------------------------------\n\n");

        printf("\n\n------------------------------ [D] GPU Start ------------------------------\n\n");
        device_vector_adder<<<dimGrid, dimBlock, 0, stream1>>>(d_a, d_b, d_c);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaDeviceSynchronize());
        printf("\n\n------------------------------ [D] GPU End ------------------------------\n\n");
        // copy result to host
        gpuErrchk(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

        printf("\n\n------------------------------ [E] Host Checking&Cleaning ------------------------------\n\n");
        if(host_checker(c, 140))
            printf("[0] Pass!\n\n");
        else
            printf("[0] Fail!\n\n");

        printf("[2] GPU Clean up:");

        gpuErrchk(cudaFree(d_a));
        gpuErrchk(cudaFree(d_b));
        gpuErrchk(cudaFree(d_c));

        printf("\n\n------------------------------ [E] Host Checking&Cleaning ------------------------------\n\n");

        printf("\n\n------------------------------ [0] GPU Compute Test End ------------------------------\n\n");
    }
    // GPU Stream Test
    {
        printf("\n\n------------------------------ [1] GPU Stream Test Start ------------------------------\n\n");

        printf("\n\n------------------------------ [C] GPU Setup Start ------------------------------\n\n");

        dim3 dimBlock(1, 1, 1); // local work size
        dim3 dimGrid(256, 256, 256); // local work group size

        printf("[0] Setup GPU Memory:\n\n");
        int *d_a, *d_b, *d_c, *d_c2;

        // allocate space for device copies
        gpuErrchk(cudaMalloc((void **)&d_a, size));
        gpuErrchk(cudaMalloc((void **)&d_b, size));
        gpuErrchk(cudaMalloc((void **)&d_c, size));
        gpuErrchk(cudaMalloc((void **)&d_c2, size));

        printf("[1] Start GPU Memory transfer:");

        // copy inputs to device
        gpuErrchk(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
        printf("\n\n------------------------------ [C] GPU Setup End ------------------------------\n\n");

        printf("\n\n------------------------------ [D] GPU Start ------------------------------\n\n");
        device_vector_adder<<<dimGrid, dimBlock, 0, stream1>>>(d_a, d_b, d_c);
        device_vector_adder<<<dimGrid, dimBlock, 0, stream2>>>(d_a, d_b, d_c2);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaDeviceSynchronize());
        printf("\n\n------------------------------ [D] GPU End ------------------------------\n\n");
        // copy result to host
        gpuErrchk(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(c2, d_c2, size, cudaMemcpyDeviceToHost));

        printf("\n\n------------------------------ [E] Host Checking&Cleaning ------------------------------\n\n");
        if(host_checker(c, 140))
            printf("[0] Stream 0 Pass!\n\n");
        else
            printf("[0] Stream 0 Fail!\n\n");

        if(host_checker(c, 140))
            printf("[1] Stream 1 Pass!\n\n");
        else
            printf("[1] Stream 1 Fail!\n\n");

        printf("[2] GPU Clean up:");

        gpuErrchk(cudaFree(d_a));
        gpuErrchk(cudaFree(d_b));
        gpuErrchk(cudaFree(d_c));
        gpuErrchk(cudaFree(d_c2));

        printf("\n\n------------------------------ [E] Host Checking&Cleaning ------------------------------\n\n");

        printf("\n\n------------------------------ [1] GPU Stream Test End ------------------------------\n\n");
    }
    // GPU Local Memory Test
    {
        printf("\n\n------------------------------ [1] GPU Local Memory Test Start ------------------------------\n\n");

        printf("\n\n------------------------------ [C] GPU Setup Start ------------------------------\n\n");

        dim3 dimBlock(32); // local work size
        dim3 dimGrid(1); // local work group size

        printf("\n\n------------------------------ [C] GPU Setup End ------------------------------\n\n");

        printf("\n\n------------------------------ [D] GPU Start ------------------------------\n\n");
        device_local_checker<<<dimGrid, dimBlock, 16, stream1>>>();
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaDeviceSynchronize());
        printf("\n\n------------------------------ [D] GPU End ------------------------------\n\n");

        printf("\n\n------------------------------ [1] GPU Local Memory Test End ------------------------------\n\n");
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(c2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}