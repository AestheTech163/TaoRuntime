#pragma once


template <typename Dtype>
void gpu_add(const Dtype* x, const Dtype* y, int n, Dtype* z);

template <typename Dtype>
void gpu_sub(const Dtype* x, const Dtype* y, int n, Dtype* z);

template <typename Dtype>
void gpu_mul(const Dtype* x, const Dtype* y, int n, Dtype* z);

template <typename Dtype>
void gpu_div(const Dtype* x, const Dtype* y, int n, Dtype* z);

#define GEN_HEADER_GPU__UNARY_FUNC(UFUNC)\
template <typename Dtype>\
void gpu_##UFUNC(const Dtype* x, int n, Dtype* y);
GEN_HEADER_GPU__UNARY_FUNC(abs)
GEN_HEADER_GPU__UNARY_FUNC(exp)
GEN_HEADER_GPU__UNARY_FUNC(log)

template <typename Dtype>
void gpu_pow(const Dtype* x, int n, Dtype e, Dtype* y);

#define GEN_HEADER_GPU__VEC_OP_SCALAR(OPNAME)\
template <typename Dtype>\
void gpu_##OPNAME##_scalar(const Dtype* x, int n, Dtype s, Dtype* y);
GEN_HEADER_GPU__VEC_OP_SCALAR(add)
GEN_HEADER_GPU__VEC_OP_SCALAR(sub)
GEN_HEADER_GPU__VEC_OP_SCALAR(mul)
GEN_HEADER_GPU__VEC_OP_SCALAR(div)
