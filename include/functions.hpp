#pragma once

#include "tensor.hpp"

template <typename Dtype, typename Itype>
tensor<Dtype> arange(Itype start, Itype stop, Itype step) {
    int64_t num = (stop-start)/step;
    tensor<Dtype> tmp(num);
    Dtype* tmp_data = tmp.data_;
    for (int i=start; i<stop; i+=step) {
        tmp_data[i] = static_cast<Dtype>(i);
    }
    return tmp;
}

template <typename Dtype, typename Itype>
tensor<Dtype> arange(Itype stop) {
    return arange<Dtype>(0, stop, 1);
}
