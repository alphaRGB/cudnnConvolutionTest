#include <iostream>
#include <cublas_v2.h>
#include "utils.cuh"

#define CHECK_CUBLAS(status) \
    if (status != CUBLAS_STATUS_SUCCESS) {  \
        std::cout << cublasGetStatusString(status) << std::endl; \
    }

// https://docs.nvidia.com/cuda/cublas/index.html
// https://zhuanlan.zhihu.com/p/441576790
void cublas_matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    // cublasOperation_t 
    // cublasSgemm(handle,)
}