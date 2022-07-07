#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include "utils.cuh"

void cudnn_conv2d(const Tensor& x_gpu, const Tensor& w_gpu, const Conv2dParam& conv_param, Tensor& y_gpu) {
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
    cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &y_gpu.n, &y_gpu.c, &y_gpu.h, &y_gpu.w);
    y_gpu.alloc_gpu();

    // conv algorithm
    // cudnnConvolutionFwdAlgoPerf_t algo_perf;
    cudnnConvolutionFwdAlgo_t conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

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
        conv_algo,
        nullptr,
        0,
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


int main() {

    cv::Mat src = cv::imread("/home/penghuiwei/MyWorkspace/ubuntu/CPP/cudaCudnnConv/image/lena.jpg");
    cv::Mat src_fp;
    src.convertTo(src_fp, CV_32FC3);

    // Input
    Tensor tensor_x, tensor_w, tensor_y;
    tensor_x.alloc_gpu(1, 3, src.rows, src.cols);
    float* dev_ptr = tensor_x.get_ptr();
    CHECK_CUDA(cudaMemcpy(dev_ptr, src_fp.data, tensor_x.size_byte, cudaMemcpyHostToDevice));

    // kernel & conv
    Conv2dParam param;
    param.pad_h = param.pad_w = 1;
    param.dilation_h = param.dilation_w = 1;
    param.u = param.v = 1;
    make_kernel(tensor_w);

    cudnn_conv2d(tensor_x, tensor_w, param, tensor_y);

    cv::Mat dst_fp(cv::Size2d(tensor_y.w, tensor_y.h), CV_32FC(tensor_y.c));
    cv::Mat dst;
    CHECK_CUDA(cudaMemcpy(dst_fp.data, tensor_y.get_ptr(), tensor_y.size_byte, cudaMemcpyDeviceToHost));
    dst_fp.convertTo(dst, CV_8UC(tensor_y.c));

    cv::imwrite("dst.png", dst);

    // cv::imshow("src", src);
    // cv::imshow("dst", dst);
    // cv::waitKey(0);
}