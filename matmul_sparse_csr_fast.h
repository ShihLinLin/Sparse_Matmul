#ifndef MATMUL_SPARSE_CSR_FAST_H
#define MATMUL_SPARSE_CSR_FAST_H

#include <vector>

// 修改介面：直接接收 CSR 格式的向量
extern "C" void matmul_sparse_csr_fast_gpu(
    int N,
    const std::vector<int>& rowPtrA,
    const std::vector<int>& colIdxA,
    const std::vector<float>& valA,
    const std::vector<int>& rowPtrB,
    const std::vector<int>& colIdxB,
    const std::vector<float>& valB,
    std::vector<float>& C
);

#endif