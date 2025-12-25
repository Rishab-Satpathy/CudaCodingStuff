#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

void vector_add_CPU(float * A, float *B,float *C, int N)
{
    for(int i=0; i<N;i++)
    {
        C[i] = A[i]+B[i];
    }
}

__global__ void vector_add_GPU(float * A, float *B,float *C, int N)
{
    int idx= blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vector_sum_GPU_2(float * A, float *B,float *C, int N)
{
    int idx= blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < N) {
    if (idx % 2 == 0)
        C[idx] = A[idx] + B[idx];
    else
        C[idx] = A[idx] - B[idx];
}

}

int main()
{
    float *h_A,*h_B,*h_C;
    float *d_A,*d_B,*d_C;
    int N;
    scanf("%d",&N);
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];
    for(int i =0;i<N;i++)
    {
        scanf("%f",&h_A[i]);
    }
    for(int i =0;i<N;i++)
    {
        scanf("%f",&h_B[i]);
    }
    cudaMalloc(&d_A,sizeof(float)*N);
    cudaMalloc(&d_B,sizeof(float)*N);
    cudaMalloc(&d_C,sizeof(float)*N);
    cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);


    printf("CPU ADDITION\n");
    vector_add_CPU(h_A,h_B,h_C,N);
    for(int i =0;i<N;i++)
    {
        printf("%f\n",h_C[i]);
    }
    int threadsperblock = 256;
    int blockspergrid = (N+256-1)/256;

    printf("GPU ADDITION\n");
    vector_add_GPU<<<blockspergrid , threadsperblock>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i =0;i<N;i++)
    {
        printf("%f\n",h_C[i]);
    }

    printf("GPU SUMMING WITH DIVERGENCE\n");
    vector_sum_GPU_2<<<blockspergrid , threadsperblock>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i =0;i<N;i++)
    {
        printf("%f\n",h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
