#include <iostream>
#include <vector>
#include "thread_pool.hpp"
#include "utils.hpp"
#include "intrinsics.hpp"


int main() {

    concur::ThreadPool tp(8);

    std::cout << "ThreadPool is open: " << tp.isOpen() << std::endl;
    std::cout << "ThreadPool is running: " << tp.isRunning() << std::endl;

    const int M=512, K=1027, N=1024;  // 使用 avx 时 K=1027 引起 非32 字节对齐的内存，最终导致程序崩归
    // const int M=128, K=128, N=128;
    float* A = (float*)aligned_alloc(64, M*K*sizeof(float));
    float* B_tran = (float*)aligned_alloc(64, N*K*sizeof(float));
    float* C = (float*)aligned_alloc(64, M*N*sizeof(float));
    float* C_ref = (float*)aligned_alloc(64, M*N*sizeof(float));
    float* A_pad;
    float* B_tran_pad;

    random_init(A, M*K, 0.f, 1.f);
    random_init(B_tran, N*K, 0.f, 1.f);

    uint64_t padding = K % 16;  // 16 个float，共 64 个字节
    uint64_t K_padded = ((K-1)/16+1)*16;
    std::cout << "K " << K << " K_padded " << K_padded << std::endl;

    if (K_padded > K) {
        A_pad = (float*)aligned_alloc(64, M*(K_padded)*sizeof(float));
        for (uint64_t i=0; i<M; i++) {
            for (uint64_t j=0; j<K; j++) {
                A_pad[i*K+j] = A[i*K+j];
            }
            for (uint64_t j=K; j<K_padded; j++) {
                A_pad[i*K+j] = 0.f;
            }
        }

        B_tran_pad = (float*)aligned_alloc(64, N*(K_padded)*sizeof(float));
        for (uint64_t i=0; i<N; i++) {
            for (uint64_t j=0; j<K; j++) {
                B_tran_pad[i*K+j] = B_tran[i*K+j];
            }
            for (uint64_t j=K; j<K_padded; j++) {
                B_tran_pad[i*K+j] = 0.f;
            }
        }
    }

    {
        Timer timer("sequential");
        for (int i=0; i<M; i++) {
            for (int j=0; j<N; j++) {
                int64_t dot = 0;
                for (int k=0; k<K; k++) {
                    dot += A[i*K+k] * B_tran[j*K+k];
                }
                C_ref[i*N+j] = dot;
            }
        }
    }

    {
        Timer timer("parallel");
        tp.parallelFor(M, [&](uint64_t begin, uint64_t end) {
            for (uint64_t i=begin; i<end; i++) {
                for (uint64_t j=0; j<N; j++) {
                    int64_t dot = 0;
                    for (uint64_t k=0; k<K; k++) {
                        dot += A[i*K+k] * B_tran[j*K+k];
                    }
                    C[i*N+j] = dot;
                }
            }
        });
    }

    if (!vec_equal(C, C_ref, M*N))
        std::cout << "No! not same." << std::endl;
    else
        std::cout << "Yes! they are same." << std::endl;


    {
        Timer timer("omp parallel");

        #pragma omp parallel for num_threads(8)
        for (int i=0; i<M; i++) {
            for (int j=0; j<N; j++) {
                int64_t dot = 0;
                for (int k=0; k<K; k++) {
                    dot += A[i*K+k] * B_tran[j*K+k];
                }
                C[i*N+j] = dot;
            }
        }
    }
    if (!vec_equal(C, C_ref, M*N))
        std::cout << "No! not same." << std::endl;
    else
        std::cout << "Yes! they are same." << std::endl;

    std::cout << "IS open: " << tp.isOpen() << std::endl;
    std::cout << "IS running: " << tp.isRunning() << std::endl;
    {
        Timer timer("parallel avx2");
        tp.parallelFor(M, [&](uint64_t begin, uint64_t end) {
            // std::cout << "M " << M << " begin " << begin << " end " << end << std::endl;
            for (uint64_t i=begin; i<end; i++) {
                for (uint64_t j=0; j<N; j++) {
                    // std::cout << " i,j " << i << ' '<< j <<std::endl;
                    float dot;
                    vec_dot_avx2(A_pad+i*K_padded, B_tran_pad+j*K_padded, K_padded, &dot);
                    C[i*N+j] = dot;
                }
            }
        });
    }
    if (!vec_equal(C, C_ref, M*N))
        std::cout << "No! not same." << std::endl;
    else
        std::cout << "Yes! they are same." << std::endl;


    free(A);
    free(B_tran);
    free(C);
    free(C_ref);
    if (K_padded > K) {
        free(B_tran_pad);
        free(A_pad);
    }

    tp.close();

    auto task = [](int i) {
        std::this_thread::sleep_for(std::chrono::seconds(i));
        std::cout << "exit " << i << std::endl;
    };

    // tp.addTask(task, 1);

    return 0;
}
