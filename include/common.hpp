#pragma once

#include <iostream>
#include <cstdio>

// #ifdef DEBUG
    #define check(cond, str) if (!(cond)) fatal(str)
    #define taoAssert(cond, str) if (!(cond)) fatal(str)
    #define fatal(str) _fatal(str, __FILE__, __LINE__)
// #else
//     #define taoAssert(cond, str)
//     #define check(cond, str)
//     #define fatal(str)
// #endif

static inline void _fatal(const char *str, const char *file, int line) {
    fprintf(stderr, "Fatal error in %s, line %d: %s\n", file, line, str);
    abort();
}


const int kMaxTensorAxes = 10;
const int CUDA_NUM_THREADS_PER_BLOCK = 512;

inline int cuda_get_blocks(const int n) {
    return (n + CUDA_NUM_THREADS_PER_BLOCK - 1) / CUDA_NUM_THREADS_PER_BLOCK;
}

enum Device {
    DEVICE_CPU = 0,
    DEVICE_GPU = 1,
};


#define CHECK_LE(a, b) if (a > b) fatal("Check less or equal failed")
#define CHECK_EQ(a, b) if (a != b) fatal("Check euqal failed")

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess); \
  } while (0)



// template <typename T, int C>
// struct TinyVec {
//     const int C_ = C;
//     T data_[C];
//     int size_;

//     TinyVec() {}

//     TinyVec(const TinyVec& other) {
//         taoAssert(other.C_ == C, "Only equally sized TinyVec can use copy constructor.");
//         if (&other == this) return;
//         size_ = other.size_;
//         for (int i=0; i<other.size_; i++)
//             data_[i] = other.data_[i];
//     }

//     TinyVec& operator=(const TinyVec& other) {
//         taoAssert(other.C_ == C, "Only equally sized TinyVec can use copy constructor.");
//         if (&other == this) return *this;
//         size_ = other.size_;
//         for (int i=0; i<other.size_; i++)
//             data_[i] = other.data_[i];
//     }

//     TinyVec(const T* x, int n_dim) {
//         taoAssert(n_dim <= C, "n_dim should be less than or equal to C.");
//         for (int i=0; i<n_dim; i++)
//             data_[i] = x[i];
//         size_ = n_dim;
//     }

//     void fill(const T* x, int n_dim) {
//         taoAssert(n_dim <= C, "Input array size out of range.");
//         for (int i=0; i<n_dim; i++)
//             data_[i] = x[i]; 
//         size_ = n_dim;
//     }

//     T operator[](int i) const { return data_[i]; }

// };

#include <iostream>

template <int C>
struct TinyVec {
    const int C_ = C;
    int64_t data_[C];
    int size_;

    TinyVec() : C_(C), size_(0) {}

    TinyVec(const TinyVec<C>& other) {
        taoAssert(other.C_ == C_, "Only equally sized TinyVec can use copy constructor.");
        if (&other == this) return;
        size_ = other.size_;
        for (int i=0; i<other.size_; i++)
            data_[i] = other.data_[i];
    }

    TinyVec& operator=(const TinyVec<C>& other) {
        taoAssert(other.C_ == C_, "Only equally sized TinyVec can use copy constructor.");
        if (&other == this) return *this;
        size_ = other.size_;
        for (int i=0; i<other.size_; i++)
            data_[i] = other.data_[i];
        return *this;
    }

    template <typename T>
    TinyVec(T a)  {
        data_[0] = static_cast<int64_t>(a);
        size_ = 1;
    }

    template <typename T>
    TinyVec(T a, T b)  {
        data_[0] = static_cast<int64_t>(a);
        data_[1] = static_cast<int64_t>(b);
        size_ = 2;
    }

    template <typename T>
    TinyVec(T a, T b, T c) {
        data_[0] = static_cast<int64_t>(a);
        data_[1] = static_cast<int64_t>(b);
        data_[2] = static_cast<int64_t>(c);
        size_ = 3;
    }

    template <typename T>
    TinyVec(T a, T b, T c, T d) {
        data_[0] = static_cast<int64_t>(a);
        data_[1] = static_cast<int64_t>(b);
        data_[2] = static_cast<int64_t>(c);
        data_[3] = static_cast<int64_t>(d);
        size_ = 4;
    }
    
    template <typename T>
    TinyVec(T a, T b, T c, T d, T e)  {
        data_[0] = static_cast<int64_t>(a);
        data_[1] = static_cast<int64_t>(b);
        data_[2] = static_cast<int64_t>(c);
        data_[3] = static_cast<int64_t>(d);
        data_[4] = static_cast<int64_t>(e);
        size_ = 5;
    }

    const int64_t& operator[](int i) const { 
        if (i < 0) return data_[size_+i];
        return data_[i]; 
    }

    int64_t& operator[](int i) {
        if (i < 0) return data_[size_+i];
        return data_[i]; 
    }

    int64_t n_elem() const { return n_elem(0, this->size_-1); }

    int64_t n_elem(int from) const { return n_elem(from, this->size_-1); }
    
    int64_t n_elem(int from, int to) const {
        int64_t s = 1;
        for (int i=from; i<=to; i++)
            s *= data_[i];
        return s;
    }

    void fill(const int64_t* x, int n_dim) {
        taoAssert(n_dim <= C_, "Input array size out of range.");
        for (int i=0; i<n_dim; i++)
            data_[i] = x[i]; 
        size_ = n_dim;
    }

    int size() const { return size_; }
};

template <int size_>
std::ostream& operator<<(std::ostream& os, const TinyVec<size_>& v) {
    os << '[' ;
    for (int i=0; i<v.size_-1; i++)
        os << v[i] << ", ";
    os << v[v.size_-1] << ']';
    return os;
}

typedef TinyVec<kMaxTensorAxes> DimsK;
typedef TinyVec<kMaxTensorAxes> AxisK;
