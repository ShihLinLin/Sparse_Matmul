#pragma once
#include <vector>

extern "C" void matmul_base(const std::vector<float>& A,
                           const std::vector<float>& B,
                           std::vector<float>& C,
                           int N);
