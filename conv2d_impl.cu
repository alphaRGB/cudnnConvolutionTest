#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "utils.cuh"
#include "conv2d_impl.cuh"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;
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


#if 0
Tensor cudnn_conv2d(const Tensor& x_gpu, const Tensor& w_gpu, const Conv2dParam& conv_param) {
    // BUG, GPU memory released
    Tensor y_gpu;
    cudnn_conv2d_out(x_gpu,w_gpu, conv_param, y_gpu);
    y_gpu.save("tensor_out2.dat");
    return y_gpu;
}
#endif //0

PYBIND11_MODULE(libconv2d, m) {
    m.doc() = "cudnn lib test";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def("alloc_gpu", py::overload_cast<>(&Tensor::alloc_gpu))
        .def("alloc_gpu", py::overload_cast<int, int, int, int>(&Tensor::alloc_gpu))
        .def_readwrite("n", &Tensor::n)
        .def_readwrite("c", &Tensor::c)
        .def_readwrite("h", &Tensor::h)
        .def_readwrite("w", &Tensor::w)
        .def("__repr__", 
            [](const Tensor& tensor){
                std::stringstream ss;
                ss << "shape:" << "[" << tensor.n << "," << tensor.c <<"," 
                << tensor.h <<"," << tensor.w <<"]" << " dtype: " << "float32"
                << " is_gpu: " << tensor.is_gpu;
                return ss.str(); 
            }
        )
        .def("set_array", 
            [](Tensor& tensor, py::array_t<float>& ndarray){
                assert(ndarray.ndim()==4 && "Dim must be 4 (n,c,h,w)");
                assert(ndarray.dtype().char_() == 'f' && "dtype must be float32");
                tensor.n = ndarray.shape(0);
                tensor.c = ndarray.shape(1);
                tensor.h = ndarray.shape(2);
                tensor.w = ndarray.shape(3);
                tensor.alloc_gpu();
                
                py::buffer_info buf = ndarray.request();
                float* ptr = tensor.get_ptr();
                CHECK_CUDA(cudaMemcpy(ptr, buf.ptr, ndarray.nbytes(), cudaMemcpyHostToDevice));
                return true;
            }
        )
        .def("get_array", 
            [](const Tensor& tensor) {
                assert(tensor.size_byte > 0);
                auto ndarray = py::array_t<float>({tensor.n, tensor.c, tensor.h, tensor.w});
                assert(tensor.size_byte == ndarray.nbytes());
                float* ptr = static_cast<float*>(ndarray.request(true).ptr);
                if(tensor.is_gpu) {
                    CHECK_CUDA(cudaMemcpy(ptr, tensor.get_ptr(), ndarray.nbytes(), cudaMemcpyDeviceToHost));
                }else {
                    memcpy(ptr, tensor.get_ptr(), ndarray.nbytes());
                }
                return ndarray;
            }
        );
    
    py::class_<Conv2dParam>(m, "Conv2dParam")
        .def(py::init<>())
        .def(py::init<int, int, int, int, int, int>())
        .def_readwrite("pad_h", &Conv2dParam::pad_h)
        .def_readwrite("pad_w", &Conv2dParam::pad_w)
        .def_readwrite("dilation_h", &Conv2dParam::dilation_h)
        .def_readwrite("dilation_w", &Conv2dParam::dilation_w)
        .def_readwrite("u", &Conv2dParam::u)
        .def_readwrite("v", &Conv2dParam::v);


    // m.def("cudnn_conv2d", cudnn_conv2d, py::arg("input_gpu"), py::arg("weight_gpu"), py::arg("params"));

    m.def("cudnn_conv2d_out", cudnn_conv2d_out, py::arg("input_gpu"), py::arg("weight_gpu"), py::arg("params"), py::arg("output_gpu"));
}
