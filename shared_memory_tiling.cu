#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void reduce_gmem(float* in, float* out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        out[idx] = in[idx];
}

__global__ void reduce_smem_conflict(float* in, float* out, int N)
{
    __shared__ float tile[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = (idx < N) ? in[idx] : 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        int i = threadIdx.x * 2;
        if (i < stride)
            tile[i] += tile[i + 1];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[blockIdx.x] = tile[0];
}



 __global__ void reduce_smem(float* in, float* out, int N)
{
    __shared__ float tile[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = (idx < N) ? in[idx] : 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            tile[threadIdx.x] += tile[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[blockIdx.x] = tile[0];
}

 int main()
 {
    const int N = 1 << 24;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, (N / 256) * sizeof(float));
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //bad event only using gmem not smem
    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
    reduce_gmem<<<blocks, threadsPerBlock>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);
    ms1 /= ITERS;

    //good event
    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
    reduce_smem<<<blocks, threadsPerBlock>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERS;

    //bad event 2 bad accessing of smem
    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
    reduce_smem_conflict<<<blocks, threadsPerBlock>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms3;
    cudaEventElapsedTime(&ms3, start, stop);
    ms3 /= ITERS;

    printf("%.3f\n",ms1);
    printf("%.3f\n",ms);
    printf("%.3f\n",ms3);

    return 0;
 }


