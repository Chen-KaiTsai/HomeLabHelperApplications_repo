#include "main.cuh"

constexpr int NUM_DEVICE = 2;
constexpr int DATA_SIZE = 1024 * 1024 * 8;

int main(int argc, char** argv)
{
    fTEST::p2p_alloc_bench();

    return 0;

    fCUDA::getDeviceName();
    fCUDA::setGPUDirect();
    
    cudaStream_t stream[NUM_DEVICE];
    char device_prefix[NUM_DEVICE][300];
    unsigned int* gpu_data[NUM_DEVICE];
    unsigned int* cpu_src_data[NUM_DEVICE];
    unsigned int* cpu_dest_data[NUM_DEVICE];

    cudaEvent_t kernel_start_event[NUM_DEVICE];
    cudaEvent_t memcpy_to_start_event[NUM_DEVICE];
    cudaEvent_t memcpy_from_start_event[NUM_DEVICE];
    cudaEvent_t memcpy_from_stop_event[NUM_DEVICE];

    float time_copy_to_ms;
    float time_kernel_ms;
    float time_copy_from_ms;
    float time_exec_ms;

    const int shared_memory_usage = 0;
    const size_t single_gpu_chunk_size = (sizeof(unsigned int) * DATA_SIZE);
    const int num_threads = 256;
    const int num_blocks = ((DATA_SIZE + (num_threads - 1))/num_threads);
    int num_devices;

    cudaGetDeviceCount(&num_devices);
    
    if(num_devices > NUM_DEVICE)
      num_devices = NUM_DEVICE;
    
    for(int device_num = 0 ; device_num < num_devices; device_num++)
    {
        cudaSetDevice(device_num);
        struct cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device_num);
        sprintf(&device_prefix[device_num][0], "ID: %d %s", device_num, device_prop.name);
        
        cudaStreamCreate(&stream[device_num]);
        cudaEventCreate(&memcpy_to_start_event[device_num]);
        cudaEventCreate(&kernel_start_event[device_num]);
        cudaEventCreate(&memcpy_from_start_event[device_num]);
        cudaEventCreate(&memcpy_from_stop_event[device_num]);

        cudaMalloc((void **)&gpu_data[device_num], single_gpu_chunk_size);
        cudaMallocHost((void **)& cpu_src_data[device_num], single_gpu_chunk_size);
        cudaMallocHost((void **)& cpu_dest_data[device_num], single_gpu_chunk_size);

        fTEST::fillArray(cpu_src_data[device_num], DATA_SIZE);

        cudaEventRecord(memcpy_to_start_event[device_num], stream[device_num]);
        cudaMemcpyAsync(gpu_data[device_num], cpu_src_data[device_num],
                                  single_gpu_chunk_size, cudaMemcpyHostToDevice,
                                  stream[device_num]);
        cudaEventRecord(kernel_start_event[device_num], stream[device_num]);
        fCUDA::dGPUTestKernel<<<num_blocks, num_threads, shared_memory_usage, stream[device_num]>>>(gpu_data[device_num]);
        cudaEventRecord(memcpy_from_start_event[device_num], stream[device_num]);
        cudaMemcpyAsync(cpu_dest_data[device_num], gpu_data[device_num],
                                  single_gpu_chunk_size, cudaMemcpyHostToDevice,
                                  stream[device_num]);
        cudaEventRecord(memcpy_from_stop_event[device_num], stream[device_num]);
    }

    for(int device_num = 0; device_num < num_devices; device_num++)
    {
        cudaSetDevice(device_num);
        cudaStreamSynchronize(stream[device_num]);

        cudaEventElapsedTime(&time_copy_to_ms, memcpy_to_start_event[device_num], kernel_start_event[device_num]);
        cudaEventElapsedTime(&time_kernel_ms, kernel_start_event[device_num], memcpy_from_start_event[device_num]);
        cudaEventElapsedTime(&time_copy_from_ms, memcpy_from_start_event[device_num], memcpy_from_stop_event[device_num]);
        cudaEventElapsedTime(&time_exec_ms, memcpy_to_start_event[device_num], memcpy_from_stop_event[device_num]);

        const float gpu_time = (time_copy_to_ms + time_kernel_ms + time_copy_from_ms);

        printf("%s Copy To : %.2f ms\n", device_prefix[device_num], time_copy_to_ms);
        printf("%s Kernel : %.2f ms\n", device_prefix[device_num], time_kernel_ms);
        printf("%s Copy Back : %.2f ms\n", device_prefix[device_num], time_copy_from_ms);
        printf("%s Execution Time : %.2f ms\n", device_prefix[device_num], time_exec_ms);
        printf("%s Component Time : %.2f ms\n", device_prefix[device_num], gpu_time);

        cudaStreamDestroy(stream[device_num]);
        cudaFree(gpu_data[device_num]);

        fTEST::checkArray(device_prefix[device_num], cpu_dest_data[device_num], DATA_SIZE);
        cudaFreeHost(cpu_src_data[device_num]);
        cudaFreeHost(cpu_dest_data[device_num]);
        cudaDeviceReset();
    }
    
    return 0;
}
