# Compilers
CXX           := g++
NVCC          := nvcc
HOST_COMPILER := /opt/apps/gcc/6.3.0/bin/g++

# Defaults for run target
N ?= 4096
S ?= 0.9
SEED ?= 42

# CUDA paths
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB1 := $(CUDA_HOME)/lib64
CUDA_LIB2 := $(CUDA_HOME)/targets/x86_64-linux/lib

# GPU arch (sm_75 for T4/RTX, sm_86 for Ampere)
ARCH ?= sm_75

# Files
# Compile main.cpp and SparseMatrix.cpp together for CPU parts
CPP_SRCS := main.cpp SparseMatrix.cpp
CU_SRCS  := matmul_base.cu matmul_sparse_csr_fast.cu
OBJS     := $(CPP_SRCS:.cpp=.o) $(CU_SRCS:.cu=.o)

TARGET := sparse_benchmark

# Flags
# CPU: C++17 for SparseMatrix features
CXXFLAGS   := -O2 -std=c++17 -fopenmp -I. -I$(CUDA_INC)

# GPU: Pass -fopenmp to host compiler via -Xcompiler
# IMPORTANT: Use backslash for multi-line variable definitions
NVCCFLAGS  := -O2 -std=c++14 -I. -I$(CUDA_INC) \
              -gencode arch=compute_$(ARCH:sm_%=%),code=$(ARCH) \
              -ccbin $(HOST_COMPILER) \
              -Xcompiler -fopenmp

# Linker: Need cudart, gomp (OpenMP), cusparse, cublas
LDFLAGS    := -L$(CUDA_LIB1) -L$(CUDA_LIB2) -lcudart -lgomp -lcusparse -lcublas \
              -Xlinker -rpath -Xlinker $(CUDA_LIB1) \
              -Xlinker -rpath -Xlinker $(CUDA_LIB2)

# Rules
.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

# Usage: make run N=4096 S=0.9 SEED=42
run: $(TARGET)
	./$(TARGET) $(N) $(S) $(SEED)
