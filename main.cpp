#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <cstdlib>

// CPU Headers
#include "SparseMatrix.hpp"

// GPU Headers
#include "matmul_base.h"
// #include "matmul_sparse_csr.h" // 不再需要標準版 CSR
#include "matmul_sparse_csr_fast.h"

using namespace std;
using namespace std::chrono;

// ==========================================
// Data Generation & Helpers
// ==========================================

// Generate raw 2D data (Dense logic)
vector<vector<double>> generateDenseMatrix(int N, double sparsity) {
    vector<vector<double>> mat(N, vector<double>(N, 0.0));
    // Data generation using OMP for speed (setup only)
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        unsigned int seed = 2025 + i; 
        for (int j = 0; j < N; j++) {
            if ((rand_r(&seed) / (double)RAND_MAX) > sparsity) {
                mat[i][j] = (double)(rand_r(&seed) % 10 + 1);
            }
        }
    }
    return mat;
}

// Convert Dense to Sparse Object
SparseMatrixCSR denseToSparseObject(const vector<vector<double>>& dense, int N) {
    SparseMatrixCSR mat(N, N);
    for (int i = 0; i < N; ++i) {
        vector<pair<int, double>> rowData;
        for (int j = 0; j < N; ++j) {
            if (dense[i][j] != 0.0) {
                rowData.push_back({j, dense[i][j]});
            }
        }
        mat.addRow(rowData);
    }
    return mat;
}

// Convert to Flat Float (for GPU Base)
vector<float> flattenAndCast(const vector<vector<double>>& source, int N) {
    vector<float> dest(N * N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dest[i * N + j] = (float)source[i][j];
        }
    }
    return dest;
}

// Convert Object to CSR Arrays (for GPU Sparse)
void sparseObjectToCSRArrays(const SparseMatrixCSR& mat, 
                             vector<int>& rowPtr, 
                             vector<int>& colIdx, 
                             vector<float>& values) {
    rowPtr = mat.rowPtr;
    colIdx = mat.colIndex;
    values.resize(mat.values.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < mat.values.size(); ++i) {
        values[i] = (float)mat.values[i];
    }
}

// Helper for Verification
vector<float> sparseResultToFlatFloat(const SparseMatrixCSR& C, int N) {
    vector<float> flat(N * N, 0.0f);
    for (int i = 0; i < C.rows; ++i) {
        for (int idx = C.rowPtr[i]; idx < C.rowPtr[i+1]; ++idx) {
            int j = C.colIndex[idx];
            flat[i * N + j] = (float)C.values[idx];
        }
    }
    return flat;
}

// Correctness Check
void checkCorrectness(const string& methodName, const vector<float>& ref, const vector<float>& result, int N) {
    double max_diff = 0.0;
    const double EPSILON = 1e-2; 

    bool pass = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = abs(ref[i] - result[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > EPSILON && pass) { 
            cout << "  [FAIL] " << methodName << " mismatch at index " << i 
                 << " Ref: " << ref[i] << " vs Res: " << result[i] << endl;
            pass = false;
        }
    }
    if (pass)
        cout << "  [PASS] " << methodName << " (Max Diff: " << max_diff << ")\n";
}

// ==========================================
// Main Benchmark
// ==========================================

int main(int argc, char* argv[]) {
    int N = 4096;
    double sparsity = 0.9;
    int seed = 2025;

    if (argc > 1) N = stoi(argv[1]);
    if (argc > 2) sparsity = stod(argv[2]);
    if (argc > 3) seed = stoi(argv[3]);

    srand(seed);

    cout << "================================================\n";
    cout << " Final Comparison Benchmark (Fast GPU Focus)\n";
    cout << " N = " << N << ", Sparsity = " << sparsity * 100 << "%\n";
    cout << "================================================\n\n";

    // 1. Generate Data
    cout << ">> Generating Data...\n";
    auto rawA = generateDenseMatrix(N, sparsity);
    auto rawB = generateDenseMatrix(N, sparsity);
    
    SparseMatrixCSR matA = denseToSparseObject(rawA, N);
    SparseMatrixCSR matB = denseToSparseObject(rawB, N);

    cout << "   A NNZ: " << matA.values.size() << "\n";
    cout << "   B NNZ: " << matB.values.size() << "\n\n";

    // Timing Variables
    double t_cpu_dense = -1.0;
    double t_cpu_single = -1.0;
    double t_cpu_multi = -1.0;
    double t_gpu_base = -1.0;
    double t_gpu_fast = -1.0;

    // Golden Reference
    vector<float> reference_result;

    // ==========================================
    // PART 1: CPU Methods
    // ==========================================
    cout << ">> [CPU] Benchmarking...\n";

    // 1.1 CPU Dense (Naive IJK - The Slow Baseline)
    if (N < 8192) {
        if (N >= 4096) cout << "   (Warning: N=" << N << " Dense CPU will be slow...)\n";
        auto start = high_resolution_clock::now();
        
        vector<vector<double>> C_dense(N, vector<double>(N, 0.0));
        
        // STRICT NAIVE I-J-K LOOP (No OMP, No Zero Check)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    C_dense[i][j] += rawA[i][k] * rawB[k][j];
                }
            }
        }
        auto end = high_resolution_clock::now();
        t_cpu_dense = duration<double, milli>(end - start).count();
        cout << "   Dense (Naive IJK): " << t_cpu_dense << " ms\n";
    } else {
        cout << "   Dense:             SKIPPED (N >= 8192)\n";
    }

    // 1.2 CPU Sparse Single
    {
        auto start = high_resolution_clock::now();
        auto C = matA.multiplySparse(matB);
        auto end = high_resolution_clock::now();
        t_cpu_single = duration<double, milli>(end - start).count();
        cout << "   Sparse Single:     " << t_cpu_single << " ms\n";
    }

    // 1.3 CPU Sparse Multi (Reference)
    SparseMatrixCSR C_parallel_obj(N, N); 
    {
        auto start = high_resolution_clock::now();
        C_parallel_obj = matA.multiplySparseParallel(matB);
        auto end = high_resolution_clock::now();
        t_cpu_multi = duration<double, milli>(end - start).count();
        cout << "   Sparse Parallel:   " << t_cpu_multi << " ms\n";
    }

    cout << "   -> Generating reference for check...\n";
    reference_result = sparseResultToFlatFloat(C_parallel_obj, N);

    // ==========================================
    // PART 2: GPU Methods
    // ==========================================
    cout << "\n>> [GPU] Benchmarking...\n";

    vector<float> A_flat = flattenAndCast(rawA, N);
    vector<float> B_flat = flattenAndCast(rawB, N);
    
    vector<int> rA, cA, rB, cB;
    vector<float> vA, vB;
    sparseObjectToCSRArrays(matA, rA, cA, vA);
    sparseObjectToCSRArrays(matB, rB, cB, vB);

    // 2.1 GPU Dense Baseline
    {
        vector<float> C_gpu_dense(N * N, 0.0f);
        // Warmup
        matmul_base(A_flat, B_flat, C_gpu_dense, N);
        
        auto start = high_resolution_clock::now();
        matmul_base(A_flat, B_flat, C_gpu_dense, N);
        auto end = high_resolution_clock::now();
        t_gpu_base = duration<double, milli>(end - start).count();
        
        cout << "   Baseline (Dense):  " << t_gpu_base << " ms";
        checkCorrectness("GPU Base", reference_result, C_gpu_dense, N);
    }

    // 2.2 GPU CSR Fast (Revolutionary) ONLY
    {
        vector<float> C_gpu_fast(N * N, 0.0f);
        // Warmup
        matmul_sparse_csr_fast_gpu(N, rA, cA, vA, rB, cB, vB, C_gpu_fast);

        auto start = high_resolution_clock::now();
        matmul_sparse_csr_fast_gpu(N, rA, cA, vA, rB, cB, vB, C_gpu_fast);
        auto end = high_resolution_clock::now();
        t_gpu_fast = duration<double, milli>(end - start).count();

        cout << "   CSR Fast (Rev.):   " << t_gpu_fast << " ms";
        checkCorrectness("GPU Fast", reference_result, C_gpu_fast, N);
    }

    // ==========================================
    // Summary
    // ==========================================
    cout << "\n================================================\n";
    cout << " Final Comparison (ms)\n";
    cout << "================================================\n";
    cout << left << setw(25) << "Method" << "Time (ms)" << endl;
    if(t_cpu_dense > 0) 
        cout << left << setw(25) << "CPU Dense" << t_cpu_dense << endl;
    cout << left << setw(25) << "CPU Sparse (Single)" << t_cpu_single << endl;
    cout << left << setw(25) << "CPU Sparse (OMP)" << t_cpu_multi << endl;
    cout << left << setw(25) << "GPU Dense (Base)" << t_gpu_base << endl;
    cout << left << setw(25) << "GPU CSR Fast" << t_gpu_fast << endl;
    
    cout << "\n------------------------------------------------\n";
    cout << " Speedups (Baseline: GPU CSR Fast)\n";
    cout << "------------------------------------------------\n";
    
    if (t_gpu_fast > 0) {
        // Calculation: Slower Time / Faster Time (GPU Fast)
        // This gives a number > 1 (e.g. 152x)
        
        if (t_cpu_dense > 0)
            cout << " GPU Fast vs CPU Dense:         " << t_cpu_dense / t_gpu_fast << "x faster\n";
        
        if (t_cpu_single > 0)
            cout << " GPU Fast vs CPU Sparse Single: " << t_cpu_single / t_gpu_fast << "x faster\n";
            
        if (t_cpu_multi > 0)
            cout << " GPU Fast vs CPU Sparse OMP:    " << t_cpu_multi / t_gpu_fast << "x faster\n";
            
        if (t_gpu_base > 0)
            cout << " GPU Fast vs GPU Dense Base:    " << t_gpu_base / t_gpu_fast << "x faster\n";
    }

    return 0;
}