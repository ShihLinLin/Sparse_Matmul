#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <vector>
#include <utility>

using namespace std;

// CSR Sparse Matrix class
class SparseMatrixCSR {
public:
    int rows, cols;
    vector<double> values;   
    vector<int> colIndex;    
    vector<int> rowPtr;      

    SparseMatrixCSR(int r, int c);

    void addRow(const vector<pair<int, double>>& rowData);

    // Sparse x Sparse multiplication
    SparseMatrixCSR multiplySparse(const SparseMatrixCSR& B) const;
    SparseMatrixCSR multiplySparseParallel(const SparseMatrixCSR& B) const;
    
    vector<vector<double>> toFullMatrix() const;
    void print() const;
};

#endif
