#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "utils.cuh"
#include "conv2d_impl.cuh"

using algo_perf_t = cudnnConvolutionFwdAlgoPerf_t;

// if exit, algo_arr[0] will be best candidate
bool get_valid_best_algo(std::vector<algo_perf_t>& algo_arr) {
    auto it = std::remove_if(algo_arr.begin(), algo_arr.end(), [](algo_perf_t algo_perf){
        return algo_perf.status != CUDNN_STATUS_SUCCESS;
    });
    algo_arr.erase(it, algo_arr.end());
    if(algo_arr.size() == 0) {
        std::runtime_error("Found no valid conv algorithm!");
    } 
    std::sort(algo_arr.begin(), algo_arr.end(), [](algo_perf_t algo1, algo_perf_t algo2){
        return algo1.time < algo2.time;
    });
    return algo_arr.size()>0;
}

void cudnn_conv2d_out(const Tensor& x_gpu, const Tensor& w_gpu, const Conv2dParam& conv_param, Tensor& y_gpu) {
    cudnnHandle_t h_handle;
    CHECK_CUDNN(cudnnCreate(&h_handle));

    cudnnTensorDescriptor_t x_desc, y_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        x_gpu.n,
        x_gpu.c,
        x_gpu.h,
        x_gpu.w
    ));

    // kernel
    cudnnFilterDescriptor_t w_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NHWC,
        w_gpu.n,
        w_gpu.c,
        w_gpu.h,
        w_gpu.w
    ));

    // conv
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        conv_param.pad_h,
        conv_param.pad_w,
        conv_param.u,
        conv_param.v,
        conv_param.dilation_h,
        conv_param.dilation_w,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    // output
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, 
        x_desc, 
        w_desc, 
        &y_gpu.n, 
        &y_gpu.c, 
        &y_gpu.h, 
        &y_gpu.w
    ));
    y_gpu.alloc_gpu();
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        y_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        y_gpu.n,
        y_gpu.c,
        y_gpu.h,
        y_gpu.w
    ));

    // conv algorithm
    std::vector<algo_perf_t> algo_perf_arr;
    int request_cnt = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(h_handle, &request_cnt));
    algo_perf_arr.resize(request_cnt);
    int algo_count = 0;

    CHECK_CUDNN(cudnnSetConvolutionMathType(conv_desc, CUDNN_FMA_MATH));

     // cudnnGetConvolutionForwardAlgorithm_v7
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        h_handle, 
        x_desc, 
        w_desc, 
        conv_desc, 
        y_desc,
        request_cnt,
        &algo_count,
        algo_perf_arr.data()
        ));

    if(!get_valid_best_algo(algo_perf_arr)) {
        std::runtime_error("Found no valid conv algorithm!");
    }
    cudnnConvolutionFwdAlgo_t best_algo = algo_perf_arr[0].algo;

    size_t ws = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        h_handle, 
        x_desc, 
        w_desc, 
        conv_desc, 
        y_desc, 
        best_algo,
        &ws));
    void* workspace = nullptr;
    if(ws > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, ws));
    }

    // Forward
    float alpha = 1.0f;
    float beta = 1.0f;
    float* out_ptr = y_gpu.get_ptr();
    CHECK_CUDNN(cudnnConvolutionForward(
        h_handle,
        &alpha,
        x_desc,
        x_gpu.get_ptr(),
        w_desc,
        w_gpu.get_ptr(),
        conv_desc,
        best_algo,
        workspace,
        ws,
        &beta,
        y_desc,
        out_ptr
    ));

    cudaDeviceSynchronize();
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(h_handle);
}


Tensor cudnn_conv2d(const Tensor& x_gpu, const Tensor& w_gpu, const Conv2dParam& conv_param) {
    Tensor y_gpu;
    cudnn_conv2d_out(x_gpu,w_gpu, conv_param, y_gpu);
    return y_gpu;
}