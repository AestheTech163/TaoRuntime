#pragma once


#include "common.hpp"

template <typename Dtype>
void gpu_print(Dtype* d_data, int size);

void* gpu_malloc(size_t n);

void gpu_free(void* gpu_data);

void gpu_memcpy_h2d(const void *h_data, size_t n, void *d_data);

void gpu_memcpy_d2h(const void *d_data, size_t n, void *h_data);

template <typename Dtype>
void gpu_memset(Dtype *d_data, const int n, const Dtype alpha);

