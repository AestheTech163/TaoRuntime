#include <iostream>
#include "mem_utils.hpp"
#include "cuda_runtime.h"
#include "basic_vectorization.hpp"
#include "utils.hpp"


int main() {

    const uint64_t N = 1 << 20;
    const uint64_t N_bytes = N * sizeof(float);

    float* x_cpu = (float*)malloc(N_bytes);
    float* x_cpu2 = (float*)malloc(N_bytes);
    random_init(x_cpu, N, -0.5f, .5f);

    float* x_gpu = (float*)gpu_malloc(N_bytes);
    float* x_gpu2 = (float*)gpu_malloc(N_bytes);

    gpu_memset(x_gpu, 10, 1.f);
    gpu_print(x_gpu, 10);

    gpu_memcpy_h2d(x_cpu, N_bytes, x_gpu);
    printf("============= X_gpu ===================\n");
    gpu_print(x_gpu, 10);
    cudaDeviceSynchronize();

    printf("============= ADD SCALAR ===================\n");
    gpu_add_scalar(x_gpu, N, 10.f, x_gpu2);
    gpu_print(x_gpu2, 10);
    cudaDeviceSynchronize();
    // gpu_memcpy_d2h(x_gpu2, N_bytes, x_cpu2);
    // for (int i=0; i<N; i++) {
    //     if (x_cpu2[i] != x_cpu[i] + 10.f) {
    //         std::cout << x_cpu2[i] << ' ' << x_cpu[i]+10.f << '_';
    //     }
    // }

    printf("============= ABS ===================\n");
    gpu_abs(x_gpu, N, x_gpu2);
    gpu_print(x_gpu2, 10);
    cudaDeviceSynchronize();

    printf("============= EXP ===================\n");
    gpu_exp(x_gpu, N, x_gpu2);
    gpu_print(x_gpu2, 10);
    cudaDeviceSynchronize();

    printf("============= LOG ===================\n");
    gpu_add_scalar(x_gpu, N, 0.51f, x_gpu);
    gpu_log(x_gpu, N, x_gpu2);
    gpu_print(x_gpu2, 10);
    cudaDeviceSynchronize();

    printf("============= POW3 ===================\n");
    gpu_pow(x_gpu, N, 3.f, x_gpu2);
    gpu_print(x_gpu2, 10);
    cudaDeviceSynchronize();

    free(x_cpu);
    gpu_free(x_gpu);
    return 0;
}
