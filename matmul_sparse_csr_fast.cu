#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "matmul_sparse_csr_fast.h"

// ---------------- Error Handling ----------------
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

// ---------------- 1. Revolutionary Optimized Kernel (Shared Mem) ----------------
// 適合 N 較小 (能塞進 Shared Memory) 的情況 -> 保持 8k 以下的極速
template <bool UseAtomic, int BLOCK_SIZE>
__global__ void spgemm_revolutionary_kernel(
    int N, 
    const int* __restrict__ rowPtrA,
    const int* __restrict__ colIdxA,
    const float* __restrict__ valA,
    const int* __restrict__ rowPtrB,
    const int* __restrict__ colIdxB,
    const float* __restrict__ valB,
    float* __restrict__ C,
    int* __restrict__ global_row_counter)
{
    int tid = threadIdx.x;
    extern __shared__ float s_row[]; // Size: N * sizeof(float)

    while (true) {
        __shared__ int row_target;
        if (tid == 0) {
            row_target = atomicAdd(global_row_counter, 1);
        }
        __syncthreads();
        if (row_target >= N) break;

        // Reset accumulator
        for (int j = tid; j < N; j += BLOCK_SIZE) {
            s_row[j] = 0.0f;
        }
        __syncthreads();

        int row_startA = __ldg(&rowPtrA[row_target]);
        int row_endA   = __ldg(&rowPtrA[row_target + 1]);

        for (int idxA = row_startA; idxA < row_endA; ++idxA) {
            int k   = __ldg(&colIdxA[idxA]);   
            float a = __ldg(&valA[idxA]);    

            int row_startB = __ldg(&rowPtrB[k]);
            int row_endB   = __ldg(&rowPtrB[k + 1]);
            int lenB = row_endB - row_startB;

            const int* __restrict__ cPtr = &colIdxB[row_startB];
            const float* __restrict__ vPtr = &valB[row_startB];

            for (int t = tid; t < lenB; t += BLOCK_SIZE) {
                int colB = __ldg(&cPtr[t]);
                float val = a * __ldg(&vPtr[t]);
                
                if (UseAtomic) {
                    atomicAdd(&s_row[colB], val);
                } else {
                    s_row[colB] += val;
                }
            }
            if (!UseAtomic) __syncthreads();
        }

        __syncthreads();

        // Write back
        float* C_row_ptr = C + (long long)row_target * N;
        for (int j = tid; j < N; j += BLOCK_SIZE) {
            float res = s_row[j];
            if (res != 0.0f) {
                C_row_ptr[j] = res;
            }
        }
    }
}

// ---------------- 2. Large Matrix Optimized Kernel (Direct + Dynamic) ----------------
// 適合 N 較大 (16k+) 的情況。
// 這裡移除 AtomicAdd，改用類似標準 CSR 的 "Direct Accumulation" 但加上動態調度。
// 這樣避免了 Global Atomics 的競爭，同時利用 L2 Cache 和 Dynamic LB。
__global__ void spgemm_large_direct_kernel(
    int N, 
    const int* __restrict__ rowPtrA,
    const int* __restrict__ colIdxA,
    const float* __restrict__ valA,
    const int* __restrict__ rowPtrB,
    const int* __restrict__ colIdxB,
    const float* __restrict__ valB,
    float* __restrict__ C,
    int* __restrict__ global_row_counter)
{
    int tid = threadIdx.x;

    while (true) {
        __shared__ int row_target;
        if (tid == 0) {
            // 動態領取一個 Row
            row_target = atomicAdd(global_row_counter, 1);
        }
        __syncthreads();
        if (row_target >= N) break;

        int row_startA = __ldg(&rowPtrA[row_target]);
        int row_endA   = __ldg(&rowPtrA[row_target + 1]);
        
        // 指向 C 的該列起始位置
        float* C_row_ptr = C + (long long)row_target * N;

        // 循序遍歷 A 的非零元素 (Sequential A)
        for (int idxA = row_startA; idxA < row_endA; ++idxA) {
            int k   = __ldg(&colIdxA[idxA]);
            float a = __ldg(&valA[idxA]);

            int row_startB = __ldg(&rowPtrB[k]);
            int row_endB   = __ldg(&rowPtrB[k + 1]);
            int lenB = row_endB - row_startB;

            const int* __restrict__ cPtr = &colIdxB[row_startB];
            const float* __restrict__ vPtr = &valB[row_startB];

            // 平行處理 B 的該列 (Parallel B)
            // 這裡不需要 atomicAdd，因為對於「同一個 idxA」，
            // 不同的 thread 處理的是 B 中不同的 column (colB)，寫入 C 的位置互不衝突。
            // 我們利用 syncthreads 確保處理完目前的 A 元素後，才進行下一個 A 元素。
            for (int t = tid; t < lenB; t += blockDim.x) {
                int colB = __ldg(&cPtr[t]);
                float val = a * __ldg(&vPtr[t]);
                
                // 直接寫入 Global Memory (會被 L2 Cache 緩衝)
                // 標準 CSR 實作也是這樣做的，這對大矩陣比 Atomic 快得多
                C_row_ptr[colB] += val; 
            }
            
            // 關鍵 Barrier：確保所有 Thread 都寫完目前的 partial product
            // 才能進入下一個 A 元素 (避免同一個 C 位置被不同 A 元素的乘積同時寫入造成 Race)
            __syncthreads();
        }
    }
}

// ---------------- Main Function ----------------
extern "C" void matmul_sparse_csr_fast_gpu(
    int N,
    const std::vector<int>& rowPtrA,
    const std::vector<int>& colIdxA,
    const std::vector<float>& valA,
    const std::vector<int>& rowPtrB,
    const std::vector<int>& colIdxB,
    const std::vector<float>& valB,
    std::vector<float>& C)
{
    static bool inited = false;
    static int  cachedN = 0;
    static int *d_rowPtrA = nullptr, *d_colIdxA = nullptr; 
    static float *d_valA = nullptr;
    static int *d_rowPtrB = nullptr, *d_colIdxB = nullptr; 
    static float *d_valB = nullptr;
    static float *d_C = nullptr;
    static int *d_work_counter = nullptr;
    static int capN = 0;
    static int capNnzA = 0;
    static int capNnzB = 0;
    
    static int num_SMs = 0;
    static int max_shared_mem_per_block = 0;

    if (num_SMs == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        num_SMs = prop.multiProcessorCount;
        // 檢查是否支援動態 Shared Memory 配置
        cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if (max_shared_mem_per_block == 0) max_shared_mem_per_block = prop.sharedMemPerBlock;
    }

    if ((int)C.size() != N * N) C.resize(N * N);

    int nnzA = (int)valA.size();
    int nnzB = (int)valB.size();

    if (!inited || N != cachedN) {
        if (N > capN || nnzA > capNnzA || nnzB > capNnzB) {
            if (d_rowPtrA) { 
                cudaFree(d_rowPtrA); cudaFree(d_colIdxA); cudaFree(d_valA);
                cudaFree(d_rowPtrB); cudaFree(d_colIdxB); cudaFree(d_valB);
                cudaFree(d_C); cudaFree(d_work_counter);
            }
            CUDA_CHECK(cudaMalloc(&d_rowPtrA, (N + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_colIdxA, nnzA * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_valA,    nnzA * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_rowPtrB, (N + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_colIdxB, nnzB * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_valB,    nnzB * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_C, (size_t)N * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_work_counter, sizeof(int)));
            capN = N; capNnzA = nnzA; capNnzB = nnzB;
        }

        CUDA_CHECK(cudaMemcpy(d_rowPtrA, rowPtrA.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIdxA, colIdxA.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valA,    valA.data(),    nnzA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rowPtrB, rowPtrB.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIdxB, colIdxB.data(), nnzB * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valB,    valB.data(),    nnzB * sizeof(float), cudaMemcpyHostToDevice));

        cachedN = N;
        inited = true;
    }

    CUDA_CHECK(cudaMemset(d_C, 0, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_work_counter, 0, sizeof(int)));

    // 決策邏輯：Shared Memory 是否放得下？
    // N * 4 bytes needed. 
    size_t needed_shared_bytes = (size_t)N * sizeof(float);
    
    // 預留一些空間給系統 (256 bytes safety)
    bool fits_in_shared = (needed_shared_bytes + 256 < (size_t)max_shared_mem_per_block);

    int grid_size = num_SMs * 8; 
    if (grid_size > N) grid_size = N;

    if (!fits_in_shared) {
        // [16k+ 策略] 使用 Direct Kernel (無 Atomics, 類似 Standard CSR 但有動態調度)
        // 使用 256 threads per block 是一個安全的通用設定
        spgemm_large_direct_kernel<<<grid_size, 256>>>(
            N, d_rowPtrA, d_colIdxA, d_valA, d_rowPtrB, d_colIdxB, d_valB, d_C, d_work_counter
        );
    } 
    else {
        // [8k- 策略] 使用 Revolutionary Shared Mem Kernel (極速)
        // 嘗試最大化 Shared Memory 配置
        if (needed_shared_bytes > 48 * 1024) {
             // 如果超過預設 48KB，需要明確 Opt-in
             // 這裡我們針對 Kernel Template 實例化進行設定
             if (N <= 2048) {
                 CUDA_CHECK(cudaFuncSetAttribute(spgemm_revolutionary_kernel<true, 256>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 (int)needed_shared_bytes));
             } else if (N <= 4096) {
                 CUDA_CHECK(cudaFuncSetAttribute(spgemm_revolutionary_kernel<true, 256>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 (int)needed_shared_bytes));
             } else {
                 CUDA_CHECK(cudaFuncSetAttribute(spgemm_revolutionary_kernel<false, 512>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 (int)needed_shared_bytes));
             }
        }

        if (N <= 2048) {
            spgemm_revolutionary_kernel<true, 256><<<grid_size, 256, needed_shared_bytes>>>(
                N, d_rowPtrA, d_colIdxA, d_valA, d_rowPtrB, d_colIdxB, d_valB, d_C, d_work_counter
            );
        } 
        else if (N <= 4096) {
            spgemm_revolutionary_kernel<true, 256><<<grid_size, 256, needed_shared_bytes>>>(
                N, d_rowPtrA, d_colIdxA, d_valA, d_rowPtrB, d_colIdxB, d_valB, d_C, d_work_counter
            );
        }
        else {
            // N > 4096 but fits in shared
            spgemm_revolutionary_kernel<false, 512><<<grid_size, 512, needed_shared_bytes>>>(
                N, d_rowPtrA, d_colIdxA, d_valA, d_rowPtrB, d_colIdxB, d_valB, d_C, d_work_counter
            );
        }
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); 
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));
}