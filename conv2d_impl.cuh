#include <cuda_runtime.h>
#include <cudnn.h>
#include "utils.cuh"


void cudnn_conv2d_out(const Tensor& x_gpu, const Tensor& w_gpu, const Conv2dParam& conv_param, Tensor& y_gpu);
Tensor cudnn_conv2d(const Tensor& x_gpu, const Tensor& w_gpu, const Conv2dParam& conv_param);
