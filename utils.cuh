 #pragma once

#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <string>
#include <assert.h>


// https://www.codenong.com/6683721/
// https://qa.1r1g.com/sf/ask/467860501/

#define CHECK_CUDA(err) \
    if (err!=cudaSuccess) { \
        std::runtime_error(cudaGetErrorString(err)); \
    }

#define CHECK_CUDNN(s) \
    if (s!=CUDNN_STATUS_SUCCESS) { \
        std::runtime_error(cudnnGetErrorString(s)); \
    }

enum TensorLayout {
    NCHW, 
    NHWC
};

struct Tensor {
    #define LAYOUT_DEFAULT_NHWC (TensorLayout::NHWC)
public:
    int n, c, h, w;
    bool is_gpu;
    int size_byte;
    TensorLayout layout;
public:
    float* ptr;
    bool allocated;
    

public:
    Tensor() {
        n = c = h = w = size_byte = 0;
        ptr = nullptr;
        is_gpu = false;
        allocated = false;
        layout = LAYOUT_DEFAULT_NHWC;
    }

    ~Tensor() {
        if(is_gpu && allocated && ptr!=nullptr) {
            CHECK_CUDA(cudaFree(ptr));
        }
    }

    void alloc_gpu(int n, int c, int h, int w, TensorLayout layout = LAYOUT_DEFAULT_NHWC) {
        this->n = n;
        this->c = c;
        this->h = h;
        this->w = w;
        this->is_gpu = true;
        assert(n>0&&c>0&&h>0&&w>0);
        size_byte = n*c*h*w*sizeof(float);
        is_gpu = true;
        this->layout = layout;
        alloc_gpu();
    }

    void alloc_gpu() {
        assert(n>0&&c>0&&h>0&&w>0);
        if(size_byte!=0) {
            assert(size_byte == n*c*h*w*sizeof(float));
        }else {
            size_byte = n*c*h*w*sizeof(float);
        }
        CHECK_CUDA(cudaMalloc((void**)&ptr, size_byte));
        is_gpu = true;
        allocated = true;
    }

    float* get_ptr() const {
        return ptr;
    }

    float* get_ptr() {
        return ptr;
    }

    void set_ptr(float* ptr) {
        this->ptr = ptr;
        this->allocated = false;
    }

    bool save(const std::string& path) {
        int numel = n * c * h * w;
        assert(numel > 0 && ptr != nullptr);
        if(numel <=0 || ptr == nullptr) 
            return false;
        std::ofstream fout(path);
        assert(fout.is_open());
        fout << "=== Meta data ===:" << std::endl
             << "Shape: " << "[" << n <<"," << c <<"," << h <<"," << w <<"]" << std::endl
             << "Numel: " << numel << std::endl
             << "Dtype: float32" << std::endl
             << "Size_byte: " << size_byte << std::endl;
        fout << "===== Values ====" << std::endl;
        float* buffer = this->ptr;
        if(is_gpu) {
            buffer = new float[numel];
            cudaMemcpy(buffer, this->ptr, this->size_byte, cudaMemcpyDeviceToHost);
        }
        for(int i=0;i<numel;i++) {
            fout << buffer[i] << std::endl;
        }
        if(is_gpu) delete [] buffer;
        return true;

    }
};

struct Conv2dParam {
    int pad_h, pad_w;
    int dilation_h, dilation_w;
    int u,v;

public:
    Conv2dParam(){}
    Conv2dParam(int pad_h, int pad_w, int dilation_h, int dilation_w, int u, int v) {
        this->pad_h = pad_h;
        this->pad_w = pad_w;
        this->dilation_h = dilation_h;
        this->dilation_w = dilation_w;
        this->u = u;
        this->v = v;
    }
};

void make_kernel(Tensor& kernel) {
    // const float kernel_template[3][3] = {
    //     {1, 1, 1},
    //     {1, -8, 1},
    //     {1, 1, 1}
    // };

    float kernel_data[9] = {
        1,1,1,1,-8,1,1,1,1
    };
    kernel.alloc_gpu(3, 3, 3, 3);
    float* ptr = kernel.get_ptr();
    for (int n=0;n<kernel.n;n++) {
        for(int c=0;c<kernel.c;c++) {
            CHECK_CUDA(cudaMemcpy(ptr, kernel_data, 9*sizeof(float), cudaMemcpyHostToDevice));
            ptr+=9;
        }
    }
}
