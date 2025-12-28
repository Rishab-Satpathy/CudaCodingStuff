#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void cache_prove(float* data, float* out, int N, int iters)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    float sum = 0.0f;
    for(int x = 0;x<iters;x++)
    {
        sum += data[idx];
    }

    out[idx] = sum;
}

int main()
{
    float *d_out, *d_in;
    float* data;
    int N = 1<<24;
    data = new float[N];
    cudaMalloc(&d_in,sizeof(float)*N);
    cudaMalloc(&d_out, sizeof(float)*N);

    for(int i=0;i<N;i++)
    {
        data[i] = 1.0f;
    }
    cudaMemcpy(d_in,data,sizeof(float)*N,cudaMemcpyHostToDevice);

    int threadsperblock = 256;
    int blocks = ( (threadsperblock+N-1) / threadsperblock );

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int iters = 100;
    cudaEventRecord(start);

    for(int i=0;i<iters;i++)
    {
        cache_prove<<<blocks,threadsperblock>>>(d_in,d_out,N,iters);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);
    ms /= 100;


    int iters = 10;
    cudaEventRecord(start);

    for(int i=0;i<iters;i++)
    {
        cache_prove<<<blocks,threadsperblock>>>(d_in,d_out,N,iters);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms1;
    cudaEventElapsedTime(&ms1,start,stop);
    ms1 /= 10;

 
    iters = 1;
    cudaEventRecord(start);

    for(int i=0;i<iters;i++)
    {
        cache_prove<<<blocks,threadsperblock>>>(d_in,d_out,N,iters);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms2;
    cudaEventElapsedTime(&ms2,start,stop);
    ms2 /= 1;

    printf("iterations = 100   %.3f,\n iterations = 10  %.3f,\n iterations = 1  %.3f",ms,ms1,ms2);

    printf("smaller data size\n");
    N = 1<<20;

   
    iters = 100;
    cudaEventRecord(start);

    for(int i=0;i<iters;i++)
    {
        cache_prove<<<blocks,threadsperblock>>>(d_in,d_out,N,iters);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);
    ms /= 100;

    
    iters = 10;
    cudaEventRecord(start);

    for(int i=0;i<iters;i++)
    {
        cache_prove<<<blocks,threadsperblock>>>(d_in,d_out,N,iters);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms1;
    cudaEventElapsedTime(&ms1,start,stop);
    ms1 /= 10;

 
    iters = 1;
    cudaEventRecord(start);

    for(int i=0;i<iters;i++)
    {
        cache_prove<<<blocks,threadsperblock>>>(d_in,d_out,N,iters);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2;
    cudaEventElapsedTime(&ms2,start,stop);
    ms2 /= 1;

    printf("iterations = 100   %.3f,\n iterations = 10  %.3f,\n iterations = 1  %.3f",ms,ms1,ms2);

    return 0;

}