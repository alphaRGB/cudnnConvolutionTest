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


struct Tensor {
public:
    int n, c, h, w;
    bool is_gpu;
    int size_byte;
public:
    float* ptr;
    bool allocated;

public:
    Tensor() {
        n = c = h = w = size_byte = 0;
        ptr = nullptr;
        is_gpu = true;
        allocated = false;
    }

    ~Tensor() {
        if(is_gpu && allocated && ptr!=nullptr) {
            CHECK_CUDA(cudaFree(ptr));
        }
    }

    void alloc_gpu(int n, int c, int h, int w) {
        this->n = n;
        this->c = c;
        this->h = h;
        this->w = w;
        this->is_gpu = true;
        assert(n>0&&c>0&&h>0&&w>0);
        size_byte = n*c*h*w*sizeof(float);
        is_gpu = true;
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
};

struct Conv2dParam {
    int pad_h, pad_w;
    int dilation_h, dilation_w;
    int u,v;
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