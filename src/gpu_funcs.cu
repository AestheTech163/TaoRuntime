
#include "gpu_funcs.hpp"
#include "common.hpp"
#include "tensor.hpp"


#define GEN_CODE__PRINT_FORMAT(DTYPE, FMT)\
__global__ void gpu_print_kernel(DTYPE* d_data, int size) {\
    printf("[");\
    for (int i=0; i<size; i++) {\
        printf("%"#FMT", ", d_data[i]);\
    }\
    printf("]\n");\
}
GEN_CODE__PRINT_FORMAT(int, d)
GEN_CODE__PRINT_FORMAT(long, ld)
GEN_CODE__PRINT_FORMAT(float, f)
GEN_CODE__PRINT_FORMAT(double, lf)

template <typename Dtype>
void gpu_print(Dtype* d_data, int size) {
    gpu_print_kernel<<<1,1>>>(d_data, size);
}
template void gpu_print(int* d_data, int size);
template void gpu_print(long* d_data, int size);
template void gpu_print(float* d_data, int size);
template void gpu_print(double* d_data, int size);

void* gpu_malloc(const size_t n) {
    void* a = NULL;
    CUDA_CHECK(cudaMalloc(&a, n));
    return a;
}

void gpu_free(void* gpu_data) { cudaFree(gpu_data); }

void gpu_memcpy_h2d(const void *h_data, const size_t n, void *d_data) {
    if (h_data != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice));
    }
}

void gpu_memcpy_d2h(const void *d_data, const size_t n, void *h_data) {
    if (h_data != d_data) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, n, cudaMemcpyDeviceToHost));
    }
}

template <typename Dtype>
__global__ void assign_kernel(Dtype* x, int n, const Dtype a) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x) {
        x[i] = a;
    }
}

template <typename Dtype>
void gpu_memset(Dtype *d_data, const int n, const Dtype alpha) {
    if (alpha == 0) {
        CUDA_CHECK(cudaMemset(d_data, 0, n*sizeof(Dtype)));
    } else {
        assign_kernel<<<cuda_get_blocks(n), CUDA_NUM_THREADS_PER_BLOCK>>>(d_data, n, alpha);
    }
}
template void gpu_memset<int>(int *d_data, const int n, const int alpha);
template void gpu_memset<float>(float *d_data, const int n, const float alpha);
template void gpu_memset<double>(double *d_data, const int n, const double alpha);

