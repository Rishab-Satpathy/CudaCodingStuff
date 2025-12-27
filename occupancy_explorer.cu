#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void baseline_kernel(float* out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = idx * 1.0f;
        out[idx] = x * x;
    }
}

//adds register pressure by increasing number of variables which are stored each per register usually
__global__ void reg_heavy_kernel(float* out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = idx, b = a + 1, c = b + 2, d = c + 3;
        float e = d + 4, f = e + 5, g = f + 6, h = g + 7;
        float i = h + 8, j = i + 9, k = j + 10, l = k + 11;
        out[idx] = a + b + c + d + e + f + g + h + i + j + k + l;
    }
}

//increase latency by increasing the number of floatingpoint functions due to increase in floating point precission
__global__ void latency_kernel(float* out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = idx;
        #pragma unroll 100
        for (int i = 0; i < 100; i++)
            x = x * 1.000001f;
        out[idx] = x;
    }
}



int main()
{
    const int N = 1 << 24;
    
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    float* d_out;
    cudaMalloc(&d_out, sizeof(float)*N);

    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int iter = 100;
    for(int i = 0; i<iter;i++)
    baseline_kernel<<<blocks, threadsPerBlock>>>(d_out,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms,start,stop);
    ms /=iter;

    cudaEventRecord(start);
    for(int i = 0; i<iter;i++)
    reg_heavy_kernel<<<blocks, threadsPerBlock>>>(d_out,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1;
    cudaEventElapsedTime(&ms1,start,stop);
    ms1 /=iter;

    cudaEventRecord(start);
    for(int i = 0; i<iter;i++)
    latency_kernel<<<blocks, threadsPerBlock>>>(d_out,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2;
    cudaEventElapsedTime(&ms2,start,stop);
    ms2 /=iter;

    printf("baseline %.2f,\nregheavy %.2f,\nlatencyheavy %.2f",ms,ms1,ms2);

    cudaFree(d_out);
    return 0;
}