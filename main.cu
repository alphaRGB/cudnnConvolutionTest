#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include "conv2d_impl.cuh"

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

    cudnn_conv2d_out(tensor_x, tensor_w, param, tensor_y);

    cv::Mat dst_fp(cv::Size2d(tensor_y.w, tensor_y.h), CV_32FC(tensor_y.c));
    cv::Mat dst;
    CHECK_CUDA(cudaMemcpy(dst_fp.data, tensor_y.get_ptr(), tensor_y.size_byte, cudaMemcpyDeviceToHost));
    dst_fp.convertTo(dst, CV_8UC(tensor_y.c));

    cv::imwrite("dst.png", dst);

    // cv::imshow("src", src);
    // cv::imshow("dst", dst);
    // cv::waitKey(0);
}