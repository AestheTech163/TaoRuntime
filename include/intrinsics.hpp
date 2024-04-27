#pragma once

#include <immintrin.h>


template <typename T>
void vec_dot_avx2(const T* x, const T* y, uint64_t n, T* dot) {
    // static_assert(false, "Not implemented datatype for vec_dot_avx2.");
}

template <>
void vec_dot_avx2(const float* x, const float* y, uint64_t n, float* dot) {
    __m256 x_vec;
    __m256 y_vec;
    __m256 sum_vec = _mm256_setzero_ps();

    uint64_t n_8 = n / 8 * 8;
    for (uint64_t i=0; i<n_8; i+=8) {
        x_vec = _mm256_load_ps(x+i);
        y_vec = _mm256_load_ps(y+i);
        _mm256_fmadd_ps(x_vec, y_vec, sum_vec);
    }
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec); 
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec); 
    sum_vec = _mm256_add_ps(sum_vec, _mm256_permute2f128_ps(sum_vec, sum_vec, 1));

    float remainder = 0.f;
    for (uint64_t i=n_8; i<n; i++) {
        // TODO 使用 intrincs
        remainder += x[i] * y[i];
    }
    remainder += sum_vec[0];
    *dot = remainder;
}
