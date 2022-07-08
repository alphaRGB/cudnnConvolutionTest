#if 0

#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include "utils.cuh"


int cudnn_conv2d(const Tensor& input, const Tensor& kernel, Tensor& output) {
    cudnnHandle_t h_cudnn;
    checkCUDNN(cudnnCreate(&h_cudnn));

    save_tensor(input, "/home/wei/ubuntu/CPP/cudaCudnnConv/data/input_tensor.dat");
    save_tensor(kernel, "/home/wei/ubuntu/CPP/cudaCudnnConv/data/kernel_tensor.dat");


    // Input
    cudnnTensorDescriptor_t desc_in;
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_in));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        desc_in,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        input.n,
        input.c,
        input.h,
        input.w
    ));
   
    // Kernel  k*c*h*w
    cudnnFilterDescriptor_t desc_kernel;
    checkCUDNN(cudnnCreateFilterDescriptor(&desc_kernel));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        desc_kernel,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NHWC,
        kernel.n,
        kernel.c,
        kernel.h,
        kernel.w
    ));

    // conv
    cudnnConvolutionDescriptor_t desc_conv;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&desc_conv));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        desc_conv,
        1,
        1,
        2,
        2,
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(desc_conv, desc_in, desc_kernel, &output.n, &output.c, &output.h, &output.w));
    output.size_byte = output.n * output.c * output.h * output.w * sizeof(float);
    output.device = Device::CUDA;
    checkCUDA(cudaMalloc((void**)&output.data_ptr, output.size_byte));
    
    // output descriptor 
    cudnnTensorDescriptor_t desc_out;
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_out));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        desc_out,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        output.n,
        output.c,
        output.h,
        output.w
    ));


    // algo
    cudnnConvolutionFwdAlgoPerf_t algo;
    int count = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        h_cudnn,
        desc_in,
        desc_kernel,
        desc_conv,
        desc_out,
        8,
        &count,
        &algo
    ));

    // Set workshapce, beacuse different convolution may require GPU memory
    size_t worksapce_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        h_cudnn,
        desc_in,
        desc_kernel,
        desc_conv,
        desc_out,
        algo.algo,
        &worksapce_size
    ));

    void* workspace = nullptr;
    if(worksapce_size > 0) {
        checkCUDA(cudaMalloc(&workspace, worksapce_size));
    }

    #if 1
    // perform
    float alpha = 1.0f;
    float beta = -100.0f;
    checkCUDNN(cudnnConvolutionForward(
        h_cudnn,
        &alpha,
        desc_in,
        input.data_ptr,
        desc_kernel,
        kernel.data_ptr,
        desc_conv,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        workspace,
        worksapce_size,
        &beta,
        desc_out,
        output.data_ptr
    ));

    #endif // 0

    
    checkCUDA(cudaDeviceSynchronize());
    save_tensor(output, "/home/wei/ubuntu/CPP/cudaCudnnConv/data/output_tensor.dat");
    
    // Deinit
    checkCUDNN(cudnnDestroyTensorDescriptor(desc_in));
    checkCUDNN(cudnnDestroyTensorDescriptor(desc_out));
    checkCUDNN(cudnnDestroyFilterDescriptor(desc_kernel));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(desc_conv));
    if(workspace!=nullptr) {
        checkCUDA(cudaFree(workspace));
    }
    checkCUDNN(cudnnDestroy(h_cudnn));

    return 0;
}

int main() {
    cv::Mat src_img, dst_img;
    cv::Mat src_fp, dst_fp;

    src_img = cv::imread("/home/wei/ubuntu/CPP/cudaCudnnConv/image/lena.jpg");  // BGR
    src_img.convertTo(src_fp, CV_32FC3);


    // // make Tensor
    const int batch_size = 1;
    Tensor input(batch_size, src_fp.channels(), src_fp.rows, src_fp.cols ,Device::CPU, (float*)src_fp.data);
    input.cuda();
    Tensor kernel;
    create_kernel_3x3x3_cuda(kernel);
    Tensor output;  // compute on fly
    

    // // call cudannConv
    cudnn_conv2d(input, kernel, output);

    // save_tensor(output, "../output_tensor.dat");

    // // copy to cpu
    dst_fp = cv::Mat::zeros(cv::Size2d(output.h, output.w), CV_32FC1);
    cudaMemcpy(dst_fp.data, output.data_ptr, output.size_byte, cudaMemcpyDeviceToHost);


    cv::normalize(dst_fp, dst_img,255.0,0.0, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("img", src_img);
    cv::imshow("dst", dst_img);
    cv::waitKey(0);

}

#endif // 0

// # https://github.com/tingshua-yts/BetterDL/blob/main/test/cudnn/test_conv.cpp

#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

cv::Mat load_image(const char* image_path) {
  cv::Mat image = cv::imread(image_path);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  std::cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
            << image.channels() << std::endl;
  return image;
}

void save_image(const char* output_filename,
                float* buffer,
                int height,
                int width) {
  cv::Mat output_image(height, width, CV_32FC3, buffer);
  // Make negative values zero.
  cv::threshold(output_image,
                output_image,
                /*threshold=*/0,
                /*maxval=*/0,
                cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
  std::cerr << "Wrote output to " << output_filename << std::endl;
}

int main(int argc, const char* argv[]) {
  // 参数解析
  if (argc < 2) {
    std::cerr << "usage: conv <image> [gpu=0] [sigmoid=0]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;

  bool with_sigmoid = (argc > 3) ? std::atoi(argv[3]) : 0;
  std::cerr << "With sigmoid: " << std::boolalpha << with_sigmoid << std::endl;

  // 加载数据
  cv::Mat image = load_image(argv[1]);

  cudaSetDevice(gpu_id);

  // create cudnn handle
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  // create input tensor descriptor
  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC, // todo why this format
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/image.rows,
                                        /*image_width=*/image.cols));

  // create filter descriptor
  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/3,
                                        /*in_channels=*/3,
                                        /*kernel_height=*/3,
                                        /*kernel_width=*/3));

  // create conv descriptor
  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/1,
                                             /*pad_width=*/1,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION, //  todo  how to compute
                                             /*computeType=*/CUDNN_DATA_FLOAT));

  // get outputDim
  int batch_size{0}, channels{0}, height{0}, width{0};
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  std::cerr << "Output Image: " << height << " x " << width << " x " << channels
            << std::endl;

  // create output descriptor
  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/channels,
                                        /*image_height=*/height,
                                        /*image_width=*/width));

  // get forward algorithm
  cudnnConvolutionFwdAlgoPerf_t perf;
  cudnnConvolutionFwdAlgo_t convolution_algorithm;

      checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          8,
                                          /*memoryLimitInBytes=*/0,
                                          &perf));

 convolution_algorithm = perf.algo;
  // get forward worksapce size
  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
            << std::endl;
//   assert(workspace_bytes > 0);

  void* d_workspace{nullptr};
//   cudaMalloc(&d_workspace, workspace_bytes);

  int image_bytes = batch_size * channels * height * width * sizeof(float);

  float* d_input{nullptr};
  cudaMalloc(&d_input, image_bytes);
  cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

  float* d_output{nullptr};
  cudaMalloc(&d_output, image_bytes);
  cudaMemset(d_output, 0, image_bytes);

  // clang-format off
  const float kernel_template[3][3] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
  };
  // clang-format on

  float h_kernel[3][3][3][3];
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  float* d_kernel{nullptr};
  cudaMalloc(&d_kernel, sizeof(h_kernel));
  cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

  const float alpha = 1.0f, beta = 1.0f;

  // conv forward
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     d_input,
                                     kernel_descriptor,
                                     d_kernel,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     d_output));

  // active
  if (with_sigmoid) {
    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                            CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_PROPAGATE_NAN,
                                            /*relu_coef=*/0));
    checkCUDNN(cudnnActivationForward(cudnn,
                                      activation_descriptor,
                                      &alpha,
                                      output_descriptor,
                                      d_output,
                                      &beta,
                                      output_descriptor,
                                      d_output));
    cudnnDestroyActivationDescriptor(activation_descriptor);
  }

  float* h_output = new float[image_bytes];
  cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

  save_image("cudnn-out.png", h_output, height, width);

  delete[] h_output;
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}