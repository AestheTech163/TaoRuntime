
#include "tensor.hpp"
#include "common.hpp"
#include "cuda_runtime.h"

template <typename T>
__global__ void test_kernel(DimsK* dims, T* p) {
    printf("%d\n", dims->size_);
    for (int i=0; i<dims->size_; i++)
        printf("%ld, ", (dims->data_)[i]);
    printf("\n");

    p[0] = 10;
    p[1] = 2;
    p[2] = 3;
}

template <typename T>
void test_cuda(tensor<T>& t) {
    test_kernel<<<1, 1>>>(t.dims_gpu_, t.data_);
    cudaDeviceSynchronize();
}

template void test_cuda(tensor<float>& t);
template void test_cuda(tensor<double>& t);
// template void test_cuda(tensor<int>& t);
// template void test_cuda(tensor<int64_t>& t);

