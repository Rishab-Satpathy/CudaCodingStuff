#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

__global__ void hello_kernel()
{
    int tid = threadIdx.x;
    printf("Hello from GPU! %d \n",tid); //address of each thread is local to the gpu block
    printf("hello from block %d thread %d and global address %d\n",blockIdx.x,threadIdx.x,blockDim.x*blockIdx.x+threadIdx.x);
}//blockdim.x is the number of threads present in each block/size of the block

__global__ void warp_hello()
{
    printf("hello warp thingy %d %d\n",threadIdx.x,blockIdx.x);
}

__global__ void add_kernel(int *out)
{
    int tid = threadIdx.x;
    out[tid] = tid*2;
    for(int i =0;i<8;i++)
    {
    printf("%d from GPU\n",out[i]);//from GPU
    }
}

int main()
{
    hello_kernel<<<2,2>>>();
    printf("hello from cpu\n");
    // warp_hello<<<1,40>>>();
    cudaDeviceSynchronize();//without devicesynchro the cpu commands might be executed before the GPU commands
    // printf("hello from cpu\n");
    printf("\n");
    int h_out[8];
    int* d_out;
    cudaMalloc(&d_out, 8 * sizeof(int)); //cuda style allocation of dynamic memory to an array, makes a pointe d_out of size 8
    add_kernel<<<1, 8>>>(d_out);
    cudaMemcpy(h_out, d_out, 8 * sizeof(int), cudaMemcpyDeviceToHost); //copies data from GPU array d_out to cpu array h_out
    for (int i = 0; i < 8; i++)
        printf("h_out[%d] = %d from CPU\n", i, h_out[i]); //printed from CPU

    cudaFree(d_out);
    return 0;
}