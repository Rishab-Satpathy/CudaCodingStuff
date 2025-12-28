#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void matmul_naive(float* A, float* B, float* C,
                             int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}


__global__ void matmul_tiled(float* A, float* B, float* C,
                             int M, int N, int K)
{
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + 15) / 16; t++) {
        if (row < M && t*16 + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] =
                A[row * K + t*16 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t*16 + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] =
                B[(t*16 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < 16; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}


// forward declarations
__global__ void matmul_naive(float* A, float* B, float* C,
                             int M, int N, int K);

__global__ void matmul_tiled(float* A, float* B, float* C,
                             int M, int N, int K);

int main()
{
    // Matrix size
    const int N = 1024;        // 1024×1024 matrix
    const int SIZE = N * N;
    const size_t BYTES = SIZE * sizeof(float);


    // Host memory
    float* h_A = (float*)malloc(BYTES);
    float* h_B = (float*)malloc(BYTES);
    float* h_C = (float*)malloc(BYTES);

    // Initialize A and B
    for (int i = 0; i < SIZE; i++) {
        h_A[i] = 1.0f;         // simple values
        h_B[i] = 1.0f;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, BYTES);
    cudaMalloc(&d_B, BYTES);
    cudaMalloc(&d_C, BYTES);

    cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);


    // Kernel launch configuration
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Naive kernel timing
    cudaMemset(d_C, 0, BYTES);

    cudaEventRecord(start);
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N, N, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);

    // Tiled kernel timing
    cudaMemset(d_C, 0, BYTES);

    cudaEventRecord(start);
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N, N, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms_tiled;
    cudaEventElapsedTime(&ms_tiled, start, stop);


    // Copy result back (optional correctness check)
    cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost);

    // Simple sanity check
    printf("C[0] = %f (expected %f)\n", h_C[0], (float)N);

    // Print results
    printf("Naive matmul time : %.3f ms\n", ms_naive);
    printf("Tiled matmul time : %.3f ms\n", ms_tiled);
    printf("Speedup           : %.2fx\n", ms_naive / ms_tiled);

 
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

