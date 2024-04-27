#include "common.hpp"
#include "tensor.hpp"
#include <vector>
#include "gpu_funcs.hpp"
#include "test.hpp"
#include "utils.hpp"
#include "functions.hpp"


int main() {

    // int64_t d[5] = {2,2,3,4,4};

    // std::vector<int> x = {1,2,3};
    // std::cout << x[0] << x[1] << x[2]<<std::endl;

    // Dims<5> dims = {2,3,4,4,5};  
    // // dims.fill(d, 5);


    // std::cout << dims << std::endl;

    // std::cout << dims.n_elem() << std::endl;

    // Dims<5> dims2 = {1,1,1,1,1};
    // std::cout << dims2 << std::endl;
    // dims2 = dims;
    // std::cout << dims2 << std::endl;


    // // DimsK dim3 = {1,2,3};
    // // std::cout << "dim3: " << dim3 << std::endl;
    // // tensor<float> t1(dim3);

    // std::cout << t1.n_elem() << std::endl;
    // std::cout << t1.raw_data() << std::endl;
    // std::cout << t1.where() << std::endl;

    // DimsK shp = t1.shape();
    // std::cout << "shape: " << shp << std::endl;

    // float data[] = {1.2, 2.1, 3.1, 4.2, 5.1, 6.1};

    // t1.from_iter(data, data+6);

    // std::cout << t1 << std::endl;


    // tensor<float> t1({1,2,3});
    // std::cout << t1.n_elem() << std::endl;
    // std::cout << t1.raw_data() << std::endl;
    // std::cout << t1.where() << std::endl;
    // t1.to(DEVICE_GPU);
    // std::cout << t1.where() << std::endl;

    // t1.reshape({1,3*2});
    // std::cout << t1.shape() << std::endl;
    // std::cout << t1.raw_data() << std::endl;
    // std::cout << t1.where() << std::endl;

    // test_cuda<float>(t1);
    // t1 = t1.to(DEVICE_CPU);
    // std::cout << "back to: " << t1.where() << std::endl;
    // std::cout << "return to cpu:" << t1 << std::endl;

    // DimsK dim3(1,2,3,4,5);

    // std::cout << "============================" << std::endl;
    // for (int i=0; i<=5; i++)
    //     std::cout << dim3[-i] << ' ';
    // std::cout << std::endl;

    // tensor<float> t2({2,3,6});
    // std::cout << t2.n_elem() << std::endl;
    // DimsK strides = t2.cal_strides();
    // std::cout << t2.shape() << std::endl;
    // std::cout << strides << std::endl;

    // std::cout << "elem " << t2(1,2,1) << std::endl;


    std::cout << "111" << std::endl;
    tensor<float> t3 = std::move(arange<float>(0, 3*4, 1));
    print_vec(t3.data_, 3*4);
    t3.reshape({3,4});
    std::cout << t3.shape() << std::endl;
    t3.transpose({1,0});
    std::cout << t3.shape() << std::endl;
    print_vec(t3.data_, 3*4);
    t3.contiguous();
    print_vec(t3.data_, 3*4);

    // tensor<float> t3 = std::move(arange<float>(0, 3*2*4, 1));
    tensor<float> t4 = std::move(arange<float>(3*2*4));
    print_vec(t4.data_, 3*2*4);
    t4.reshape({3, 2, 4});
    std::cout << t4.shape() << std::endl;
    t4.transpose({2,1,0});
    std::cout << t4.shape() << std::endl;
    print_vec(t4.data_, 3*2*4);
    t4.contiguous();
    print_vec(t4.data_, 3*2*4);

    int num = 3*7*1*5;
    tensor<float> t5 = std::move(arange<float>(num));
    print_vec(t5.data_, num);
    t5.reshape({3,7,1,5});
    std::cout << t5.shape() << std::endl;
    t5.transpose({2,1,0,3});
    std::cout << t5.shape() << std::endl;
    print_vec(t5.data_, num);
    t5.contiguous();
    print_vec(t5.data_, num);
    // std::cout << "222" << std::endl;
    // t3.reshape({2, 3, 4});
    // std::cout << "444" << std::endl;
    // tensor<float> t4({2,3});
    // t4.to(DEVICE_GPU);
    // t4 = std::move(t3);
    // std::cout << "555" << std::endl;


    // t2.reshape({3,2,2,3});
    // std::cout << "reshape" << std::endl;
    // std::cout << t2.shape() << std::endl;
    // std::cout << t2.stride() << std::endl;
    
    // t2.to(DEVICE_GPU);
    // std::cout << "where " << t2.where() << std::endl;
    // gpu_print(t2.dims_gpu_->data_, t2.dims_cpu_->size());

    // t2.to(DEVICE_CPU);
    // std::cout << "where " << t2.where() << std::endl;
    // gpu_print(t2.dims_gpu_->data_, t2.dims_cpu_->size());

}