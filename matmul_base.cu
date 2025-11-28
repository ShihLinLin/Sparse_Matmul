#include "matmul_base.h"
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float val = 0;
        for (int k = 0; k < N; ++k)
            val += A[row * N + k] * B[k * N + col];
        C[row * N + col] = val;
    }
}

extern "C" void matmul_base (const std::vector<float>& A,
                           const std::vector<float>& B,
                           std::vector<float>& C,
                           int N) {
    size_t bytes = N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16); // total 256 threads
    dim3 grid((N + block.x - 1) / block.x, // N+m-1/m
              (N + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
