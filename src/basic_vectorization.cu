
#include "basic_vectorization.hpp"
#include "common.hpp"
#include "stdlib.h"
#include "cmath"


#define GEN_CODE__UNARY_FUNC(UFUNC)\
template <typename Dtype>\
__global__ void UFUNC##_kernel(const Dtype* x, int n, Dtype* y) {\
    for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x)\
        y[i] = UFUNC(x[i]);\
}\
template <typename Dtype>\
void gpu_##UFUNC(const Dtype* x, int n, Dtype* y) {\
    UFUNC##_kernel<Dtype><<<cuda_get_blocks(n), CUDA_NUM_THREADS_PER_BLOCK>>>(x, n, y);\
}
GEN_CODE__UNARY_FUNC(abs)
GEN_CODE__UNARY_FUNC(exp)
GEN_CODE__UNARY_FUNC(log)
GEN_CODE__UNARY_FUNC(sqrt)
#define GEN_WRAPPER__UNARY_FUNC(UFUNC)\
template void gpu_##UFUNC(const float* x, int n, float* y);\
template void gpu_##UFUNC(const double* x, int n, double* y);
GEN_WRAPPER__UNARY_FUNC(abs)
GEN_WRAPPER__UNARY_FUNC(exp)
GEN_WRAPPER__UNARY_FUNC(log)
GEN_WRAPPER__UNARY_FUNC(sqrt)


#define GEN_CODE__BINARY_OP(OPNAME, OP)\
template <typename Dtype>\
__global__ void OPNAME##_kernel(const Dtype* x, const Dtype* y, int n, Dtype* z) {\
    for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x)\
        z[i] = x[i] OP y[i];\
}\
template <typename Dtype>\
void gpu_##OPNAME(const Dtype* x, const Dtype* y, int n, Dtype* z) {\
    OPNAME##_kernel<Dtype><<<cuda_get_blocks(n), CUDA_NUM_THREADS_PER_BLOCK>>>(x, y, n, z);\
}

GEN_CODE__BINARY_OP(add, +)
GEN_CODE__BINARY_OP(sub, -)
GEN_CODE__BINARY_OP(mul, *)
GEN_CODE__BINARY_OP(div, /)

#define GEN_WRAPPER__BINARY_OP(OPNAME)\
template void gpu_##OPNAME(const int* x, const int* y, int n, int* z);\
template void gpu_##OPNAME(const float* x, const float* y, int n, float* z);\
template void gpu_##OPNAME(const double* x, const double* y, int n, double* z);

GEN_WRAPPER__BINARY_OP(add)
GEN_WRAPPER__BINARY_OP(sub)
GEN_WRAPPER__BINARY_OP(mul)
GEN_WRAPPER__BINARY_OP(div)


template <typename Dtype>
__global__ void pow_kernel(const Dtype* x, int n, const Dtype e, Dtype* y) {
    for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x)
        y[i] = pow(x[i], e);
}
template <typename Dtype>
void gpu_pow(const Dtype* x, int n, Dtype e, Dtype* y) {
    pow_kernel<Dtype><<<cuda_get_blocks(n), CUDA_NUM_THREADS_PER_BLOCK>>>(x, n, e, y);
}
template void gpu_pow(const float* x, int n, float e, float* y);
template void gpu_pow(const double* x, int n, double e, double* y);


#define GEN_CODE__VEC_OP_SCALAR(OPNAME, OP)\
template <typename Dtype>\
__global__ void OPNAME##_scalar_kernel(const Dtype* x, int n, Dtype s, Dtype* z) {\
    for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x)\
        z[i] = x[i] OP s;\
}\
template <typename Dtype>\
void gpu_##OPNAME##_scalar(const Dtype* x, int n, Dtype s, Dtype* z) {\
    OPNAME##_scalar_kernel<Dtype><<<cuda_get_blocks(n), CUDA_NUM_THREADS_PER_BLOCK>>>(x, n, s, z);\
}
GEN_CODE__VEC_OP_SCALAR(add, +)
GEN_CODE__VEC_OP_SCALAR(sub, -)
GEN_CODE__VEC_OP_SCALAR(mul, *)
GEN_CODE__VEC_OP_SCALAR(div, /)

#define GEN_WRAPPER__VEC_OP_SCALAR(OPNAME)\
template void gpu_##OPNAME##_scalar(const int* x, int n, int s, int* z);\
template void gpu_##OPNAME##_scalar(const float* x, int n, float s, float* z);\
template void gpu_##OPNAME##_scalar(const double* x, int n, double s, double* z);

GEN_WRAPPER__VEC_OP_SCALAR(add)
GEN_WRAPPER__VEC_OP_SCALAR(sub)
GEN_WRAPPER__VEC_OP_SCALAR(mul)
GEN_WRAPPER__VEC_OP_SCALAR(div)

