#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

__global__ void read_kernel(float *in, float *out,int stride, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if((idx*stride)<N)
    {
        out[idx] = in[idx*stride]; //stride is to make the memory noncontiguous and increase read times, this doesn't change the data traversed through size as The GPU does not walk through all the data sequentially.
        //It jumps directly to the requested addresses, but each jump pulls in an entire cache line, including data you didn’t ask for.
    }
}

int main()
{
    int N = 1<<24;
    float *in,*out;
    cudaMalloc(&in,sizeof(float)*N);
    cudaMalloc(&out,sizeof(float)*N);
    float *h_in;
    h_in = new float[N];
   
    int i =0;
    while(i<N)
    {
        h_in[i] = i++;
    }

    cudaMemcpy(in,h_in,sizeof(float)*N,cudaMemcpyHostToDevice);
    // cudaMemcpy(out,h_out,sizeof(float)*N,cudaMemcpyHostToDevice);

    int threadsperblock = 256;
    int blocks = ( (threadsperblock+N-1) / threadsperblock );

    //to record the event time stamps
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float bytes = N * sizeof(float);

    for(int j=1;j<33;j=j*2)
    {
        const int ITERS = 100;

        cudaEventRecord(start);
        for (int k = 0; k < ITERS; k++)
            read_kernel<<<blocks, threadsperblock>>>(in, out, j, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop); //synchronizes the GPU with the CPU when the stop event is recorded

        float ms;
        cudaEventElapsedTime(&ms, start, stop);  //to convert the timestamps into humnan readable time;
        ms /= ITERS; //to get the average of all the 100 iterations as ms is the total run time, more iterations to find out consistency 

        float bandwidth = bytes / (ms * 1e6); // GB/s to record the memory bandwidth for different striding data
        printf("stride=%d  time=%.3f ms  bandwidth=%.2f GB/s\n", j, ms, bandwidth);
    }

    cudaFree(in);
    cudaFree(out);
    delete[] h_in;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;

}