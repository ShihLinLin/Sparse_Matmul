#include "SparseMatrix.hpp"
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <omp.h>

SparseMatrixCSR::SparseMatrixCSR(int r, int c) : rows(r), cols(c) {
    rowPtr.push_back(0);
}

void SparseMatrixCSR::addRow(const vector<pair<int, double>>& rowData) {
    for (auto& p : rowData) {
        colIndex.push_back(p.first);
        values.push_back(p.second);
    }
    rowPtr.push_back(values.size());
}

// Sparse x Sparse multiplication
SparseMatrixCSR SparseMatrixCSR::multiplySparse(const SparseMatrixCSR& B) const {
    if (cols != B.rows) {
        cerr << "Dimension mismatch!" << endl;
        exit(1);
    }

    SparseMatrixCSR C(rows, B.cols);

    for (int i = 0; i < rows; ++i) {
        unordered_map<int, double> rowMap;

        for (int idxA = rowPtr[i]; idxA < rowPtr[i + 1]; ++idxA) {
            int colA = colIndex[idxA];
            double valA = values[idxA];

            for (int idxB = B.rowPtr[colA]; idxB < B.rowPtr[colA + 1]; ++idxB) {
                int colB = B.colIndex[idxB];
                double valB = B.values[idxB];
                rowMap[colB] += valA * valB;
            }
        }

        vector<pair<int, double>> rowData;
        for (auto& p : rowMap) {
            if (p.second != 0.0) rowData.push_back({p.first, p.second});
        }

        sort(rowData.begin(), rowData.end(), [](auto &a, auto &b){ return a.first < b.first; });

        C.addRow(rowData);
    }

    return C;
}

SparseMatrixCSR SparseMatrixCSR::multiplySparseParallel(const SparseMatrixCSR& B) const {
    if (cols != B.rows) {
        cerr << "Dimension mismatch!" << endl;
        exit(1);
    }

    SparseMatrixCSR C(rows, B.cols);
    vector<vector<pair<int,double>>> rowResults(rows);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rows; ++i) {
        unordered_map<int, double> rowMap;

        for (int idxA = rowPtr[i]; idxA < rowPtr[i + 1]; ++idxA) {
            int colA = colIndex[idxA];
            double valA = values[idxA];

            for (int idxB = B.rowPtr[colA]; idxB < B.rowPtr[colA + 1]; ++idxB) {
                int colB = B.colIndex[idxB];
                double valB = B.values[idxB];
                rowMap[colB] += valA * valB;
            }
        }

        vector<pair<int, double>> rowData;
        for (auto& p : rowMap)
            if (p.second != 0.0) rowData.push_back({p.first, p.second});

        sort(rowData.begin(), rowData.end(),
             [](auto &a, auto &b){ return a.first < b.first; });

        rowResults[i] = move(rowData);
    }

    for (auto &rowData : rowResults)
        C.addRow(rowData);

    return C;
}

vector<vector<double>> SparseMatrixCSR::toFullMatrix() const {
    vector<vector<double>> full(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i) {
        for (int idx = rowPtr[i]; idx < rowPtr[i+1]; ++idx) {
            int j = colIndex[idx];
            full[i][j] = values[idx];
        }
    }
    return full;
}

void SparseMatrixCSR::print() const {
    for (int i = 0; i < rows; ++i) {
        unordered_map<int,double> rowMap;
        for (int idx = rowPtr[i]; idx < rowPtr[i+1]; ++idx)
            rowMap[colIndex[idx]] = values[idx];
        for (int j = 0; j < cols; ++j) {
            if (rowMap.count(j)) cout << rowMap[j] << " ";
            else cout << "0 ";
        }
        cout << endl;
    }
}