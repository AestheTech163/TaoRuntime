#pragma once

#include "common.hpp"
#include "gpu_funcs.hpp"
#include <ostream>

inline bool is_incre(const AxisK& x) {
    bool is = true;
    for (int i=1; i<x.size_; i++)
        if (x[i] < x[i-1])
            is = false;
    return is;
}

template <typename Dtype>
class tensor {

    tensor(const tensor<Dtype>&) = delete;
    tensor<Dtype>& operator=(const tensor<Dtype>&) = delete;
public:

    tensor(const DimsK& shape);

    tensor(tensor<Dtype>&& shape);

    tensor<Dtype>& operator=(tensor<Dtype>&&);

    ~tensor();

    template <typename ConstIterType>
    void from_iter(ConstIterType, ConstIterType);

    tensor<Dtype>& reshape(const DimsK& shape);

    tensor<Dtype>& transpose(const AxisK& axis);

    tensor<Dtype>& contiguous();

    int64_t n_elem() const { return dims_cpu_->n_elem(); }

    const DimsK& shape() const { return *dims_cpu_; }
    const DimsK& stride() const { return *strd_cpu_; }

    Dtype& operator()(int64_t idx);
    Dtype& operator()(int64_t idx1, int64_t idx2);
    Dtype& operator()(int64_t idx1, int64_t idx2, int64_t idx3);
    Dtype& operator()(int64_t idx1, int64_t idx2, int64_t idx3, int64_t idx4);

    DimsK cal_strides() const {
        // DimsK* dims = (where_== DEVICE_CPU) ? dims_cpu_ : dims_gpu_;
        DimsK strides(*dims_cpu_);
        strides[-1] = 1;
        for (int i=dims_cpu_->size()-2; i>=0; i--) {
            strides[i] = (*dims_cpu_)[i+1] * strides[i+1];
        }
        return strides;
    }

    const Dtype* raw_data() const { return data_; }
    Dtype* raw_data() { return data_; }

    const char* where() const {
        if (where_ == DEVICE_CPU) return "CPU";
        else return "GPU";
    }

    void to(Device dev);

    Dtype* data_;       // keep only one copy of data whether in device or host
    DimsK* dims_cpu_;   //
    DimsK* strd_cpu_;
    DimsK* dims_gpu_;
    DimsK* strd_gpu_;
    // DimsK* dims_;
    // DimsK* strd_;
    Device where_;
    AxisK trans_;
};

template <typename Dtype>
Dtype& tensor<Dtype>::operator()(int64_t idx) {
    CHECK_EQ(dims_cpu_->size_, 1);
    return data_[idx];
}

template <typename Dtype>
Dtype& tensor<Dtype>::operator()(int64_t idx1, int64_t idx2) {
    CHECK_EQ(dims_cpu_->size_, 2);
    int64_t* strds = strd_cpu_->data_;
    return data_[idx1*strds[0]+idx2];
}
template <typename Dtype>
Dtype& tensor<Dtype>::operator()(int64_t idx1, int64_t idx2, int64_t idx3) {
    CHECK_EQ(dims_cpu_->size_, 3);
    int64_t* strds = strd_cpu_->data_;
    return data_[idx1*strds[0]+idx2*strds[1]+idx3];
}
template <typename Dtype>
Dtype& tensor<Dtype>::operator()(int64_t idx1, int64_t idx2, int64_t idx3, int64_t idx4) {
    CHECK_EQ(dims_cpu_->size_, 4);
    int64_t* strds = strd_cpu_->data_;
    return data_[idx1*strds[0]+idx2*strds[1]+idx3*strds[2]+idx4];
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const tensor<T>& t) {
    int64_t size_ = t.n_elem();
    os << '[' ;
    const T* raw_data = t.raw_data();
    for (int i=0; i<size_-1; i++)
        os << raw_data[i] << ", ";
    os << raw_data[size_-1] << ']';
    return os;
}

template <typename Dtype>
tensor<Dtype>::tensor(const DimsK& dims) : data_(nullptr),
    dims_cpu_(nullptr), dims_gpu_(nullptr),
    strd_cpu_(nullptr), strd_gpu_(nullptr), 
    where_(DEVICE_CPU) {

    dims_cpu_ = (DimsK*)malloc(sizeof(DimsK));
    strd_cpu_ = (DimsK*)malloc(sizeof(DimsK));
    new (dims_cpu_) DimsK(dims);
    new (strd_cpu_) DimsK(dims);
    *strd_cpu_ = cal_strides();

    int64_t n_elem = dims_cpu_->n_elem();
    taoAssert(n_elem >= 0, "Invalid dims for tensor.");

    if (n_elem > 0)
        data_ = (Dtype*)malloc(n_elem*sizeof(Dtype));

    trans_.size_ = dims_cpu_->size_;
    for (int i=0; i<dims_cpu_->size_; i++) {
        trans_[i] = i;
    }
}

template <typename Dtype>
tensor<Dtype>::tensor(tensor<Dtype>&& other) {
    std::swap(data_, other.data_);
    std::swap(dims_cpu_, other.dims_cpu_);
    std::swap(strd_cpu_, other.strd_cpu_);
    std::swap(dims_gpu_, other.dims_gpu_);
    std::swap(strd_gpu_, other.strd_gpu_);
    std::swap(where_, other.where_);
    std::swap(trans_, other.trans_);
}

template <typename Dtype>
tensor<Dtype>& tensor<Dtype>::operator=(tensor<Dtype>&& other) {
    if (this == &other) return *this;
    std::swap(data_, other.data_);
    std::swap(dims_cpu_, other.dims_cpu_);
    std::swap(strd_cpu_, other.strd_cpu_);
    std::swap(dims_gpu_, other.dims_gpu_);
    std::swap(strd_gpu_, other.strd_gpu_);
    std::swap(where_, other.where_);
    std::swap(trans_, other.trans_);
    return *this;
}

template <typename Dtype>
tensor<Dtype>::~tensor() {
    if ((where_ == DEVICE_CPU) && data_) {
        free(data_);
    } else if ((where_ == DEVICE_GPU) && data_) {
        gpu_free(data_);
    }
    if (dims_cpu_) {
        free(dims_cpu_);
        free(strd_cpu_);
    }
    if (dims_gpu_) {
        gpu_free(dims_gpu_);
        gpu_free(strd_gpu_);
    }
}

template <typename Dtype>
tensor<Dtype>& tensor<Dtype>::reshape(const DimsK& dims) {

    taoAssert(dims.n_elem() == dims_cpu_->n_elem(), "Reshape should keep the size of tensor.");
    *dims_cpu_ = dims;
    *strd_cpu_ = cal_strides();
    if (where_ == DEVICE_GPU) {
        gpu_memcpy_h2d(dims_cpu_, sizeof(DimsK), dims_gpu_);
        gpu_memcpy_h2d(strd_cpu_, sizeof(DimsK), strd_gpu_);
    }
    return *this;
}

template <typename Dtype>
tensor<Dtype>& tensor<Dtype>::transpose(const AxisK& axis) {
    taoAssert(axis.size() == 2 || axis.size() == dims_cpu_->size(), "Transposing with invalid axis.");

    if (axis.size() == 2) {
        if (dims_cpu_->size_ == 2 && is_incre(axis))
            return *this;
        std::swap((*dims_cpu_)[axis[0]], (*dims_cpu_)[axis[1]]);
        trans_ = axis;
    } else {
        if (is_incre(axis))  // axes are monotonicly increasing, like : (0,1,2,3), there nothing todo
            return *this;
        DimsK tmp_dim, tmp_strd;
        tmp_dim.size_ = dims_cpu_->size_;
        for (int i=0; i<axis.size(); i++) {
            tmp_dim[i] = (*dims_cpu_)[axis[i]];
        }
        *dims_cpu_ = tmp_dim;
        trans_ = axis;
    }

    if (where_ == DEVICE_GPU) {
        gpu_memcpy_h2d(dims_cpu_, sizeof(DimsK), dims_gpu_);
    }

    return *this;
}

template <typename Dtype>
tensor<Dtype>& tensor<Dtype>::contiguous() {
    if (where_ == DEVICE_CPU) {
        if (is_incre(trans_))
            return *this;
        Dtype* new_data = new Dtype[n_elem()];
        int K = dims_cpu_->size_;
        int64_t N = n_elem();
        int64_t I[dims_cpu_->size()] = {0};
        int cur_axis = 0;
        int64_t* dims = dims_cpu_->data_;
        int64_t* strd = strd_cpu_->data_;
        DimsK tran_strd = cal_strides();
        int j = -1;
        while (++j < N) {
            I[0] = j/strd[0];
            for (int k=1; k<K; k++)
                I[k] = (j%strd[k-1])/strd[k];

            int64_t orig_idx = 0, new_idx = 0;
            for (int k=0; k<K; k++) orig_idx += I[k] * strd[k];
            for (int k=0; k<K; k++) new_idx  += I[trans_[k]] * tran_strd[k];

            new_data[new_idx] = data_[orig_idx];
        }
        delete data_;
        data_ = new_data;
        *strd_cpu_ = cal_strides();

    } else {
        // gpu
        if (where_ == DEVICE_GPU) {
            gpu_memcpy_h2d(dims_cpu_, sizeof(DimsK), dims_gpu_);
        }

    }

    return *this;
}

template <typename Dtype>
template <typename ConstIterType>
void tensor<Dtype>::from_iter(ConstIterType begin, ConstIterType end) {
    typedef typename std::iterator_traits<ConstIterType>::value_type v_type;
    static_assert(std::is_same<v_type, Dtype>::value, "Inconsistent data type.");

    if (where_ == DEVICE_CPU)
        std::copy(begin, end, data_);
    else {
        gpu_memcpy_d2d(begin, (end-begin)*sizeof(Dtype), data_);
    }
}

template <typename Dtype>
void tensor<Dtype>::to(Device dev) {
    if (where_ == DEVICE_GPU && dev == DEVICE_CPU) {
        // from GPU to CPU
        int64_t n_elem = dims_cpu_->n_elem();
        int64_t n_elem_bytes = n_elem * sizeof(Dtype);
        Dtype* to_ptr = (Dtype*)malloc(n_elem_bytes);

        gpu_memcpy_d2h(data_, n_elem_bytes, to_ptr);
        gpu_free(data_);
        data_ = to_ptr;

        gpu_memcpy_d2h(dims_gpu_, sizeof(DimsK), dims_cpu_);
        gpu_memcpy_d2h(strd_gpu_, sizeof(DimsK), strd_cpu_);

        where_ = DEVICE_CPU;

    } else if (where_ == DEVICE_CPU && dev == DEVICE_GPU) {
        // from CPU to GPU
        int64_t n_elem_bytes = dims_cpu_->n_elem() * sizeof(Dtype);
        Dtype* to_ptr = (Dtype*)gpu_malloc(n_elem_bytes);

        gpu_memcpy_h2d(data_, n_elem_bytes, to_ptr);
        free(data_);
        data_ = to_ptr;

        if (dims_gpu_ == nullptr) {
            dims_gpu_ = (DimsK*)gpu_malloc(sizeof(DimsK));
            strd_gpu_ = (DimsK*)gpu_malloc(sizeof(DimsK));
        }
        gpu_memcpy_h2d(dims_cpu_, sizeof(DimsK), dims_gpu_);
        gpu_memcpy_h2d(strd_cpu_, sizeof(DimsK), strd_gpu_);

        where_ = DEVICE_GPU;
    }
}
