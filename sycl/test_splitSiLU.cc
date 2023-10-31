#include "utils.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/math.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

#define M 512
#define N 4096

int main() {
    float eps = 0.0001f;
    constexpr size_t VL = 16;
    property_list properties{property::queue::enable_profiling()};
    queue q(properties);
    auto context = q.get_info<info::queue::context>();
    auto device = q.get_info<info::queue::device>();

    constexpr size_t size_a = M * N;
    // auto src = alloc_device_and_init<float>(
    //     size_a,
    //     [](float* data, size_t idx) {
    //         data[idx] = generate_random<float>();
    //     },
    //     q, device, context);
    auto src = alloc_host_and_init<float>(
        size_a,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
        });
    auto src_gpu = static_cast<float *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_a * sizeof(float), device, context));
    q.memcpy((void *)src_gpu, (void *)src, size_a * sizeof(float)).wait();

    // auto dst = alloc_device_and_init<float>(
    //     size_a,
    //     [](float* data, size_t idx) {
    //         data[idx] = 0.0f;
    //     },
    //     q, device, context);
    auto dst = alloc_host_and_init<float>(
        size_a,
        [](float* data, size_t idx) {
            data[idx] = 0.0f;
        });
    auto dst_gpu = static_cast<float *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_a * sizeof(float), device, context));
    q.memcpy((void *)dst_gpu, (void *)dst, size_a * sizeof(float)).wait();

    auto dst_gpu_result_on_cpu = alloc_host_and_init<float>(
        size_a,
        [](float* data, size_t idx) {
            data[idx] = 0.0f;
        });

    for (size_t it = 0; it < 5; it++) {
        auto t1 = std::chrono::steady_clock::now();   // Start timing
        auto one_v = _mm512_set1_ps(1.0f);
        auto neg_one_v = _mm512_set1_ps(-1.0f);
#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N / 2; j += 16) {
                auto left = _mm512_loadu_ps(src + i * N + j);
                auto right = _mm512_loadu_ps(src + i * N + j + N / 2);
                auto x0 = _dil_exp_kernel(_mm512_mul_ps(left, neg_one_v));
                auto x1 = _mm512_add_ps(one_v, x0);
                auto x2 = _mm512_div_ps(left, x1);
                auto res = _mm512_mul_ps(right, x2);
                if (i == 256 && j == 0) {
                    std::cout << "res: " << res[0] << std::endl;
                }
                _mm512_storeu_ps(static_cast<float *>(dst) + i * N / 2 + j, res);
            }
        }

        auto t2 = std::chrono::steady_clock::now();   // Stop timing
        auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "warmup iter:" << it << " cpu kernel time: " << exec_time << "us " << std::endl;
    }

    range size{M};
    // // warmup stage
    // std::cout << "*****warmup stage*****" << std::endl;
    // for (int it = 0; it < 5; it++) {
    //     auto t1 = std::chrono::steady_clock::now();   // Start timing
    //     auto event = q.submit([&](handler& h) {
    //         h.parallel_for(size, [=](id<1> idx) {
    //             int i = idx[0];
    //             float mean = 0.0f;
    //             float var = 0.0f;
    //             for (int j = 0; j < N; j++) {
    //                 auto tmp = src_gpu[i * N + j];
    //                 if (i == 0 && j == 15) {
    //                     sycl::ext::oneapi::experimental::printf("non vec src_gpu[%d]:%f\n", j, tmp);
    //                 }
    //                 mean += tmp;
    //                 var += tmp * tmp;
    //             }
    //             mean /= N;
    //             var = var / N - mean * mean;
    //             auto rsqrt_var = sycl::rsqrt(var + eps);
    //             if (i == 0) {
    //                 sycl::ext::oneapi::experimental::printf("non vec mean:%f rsqrt_var:%f\n", mean, rsqrt_var);
    //             }

    //             // layernorm
    //             for (int j = 0; j < N; j++) {
    //                 auto amplifier = gamma_gpu[j] * rsqrt_var;
    //                 auto offset = amplifier * mean - beta_gpu[j];
    //                 dst_gpu[i * N + j] = amplifier * src_gpu[i * N + j] - offset;
    //                 if (i == 0 && j == 15) {
    //                     sycl::ext::oneapi::experimental::printf("non vec dst_gpu[%d]:%f\n", j, dst_gpu[i * N + j]);
    //                 }
    //             }
    //         });
    //     });
    //     event.wait();
    //     auto t2 = std::chrono::steady_clock::now();   // Stop timing
    //     auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    //     std::cout << "warmup iter:" << it << " sync kernel time: " << exec_time << "us ";

    //     auto gpu_start = event.template get_profiling_info<
    //             sycl::info::event_profiling::command_start>();
    //     auto gpu_end = event.template get_profiling_info<
    //             sycl::info::event_profiling::command_end>();
    //     auto exec_time_prof = (gpu_end - gpu_start) / 1000.0;
    //     std::cout << " pure gpu kernel time: " << exec_time_prof << "us" << std::endl;
    // }

    // q.memcpy((void *)dst_gpu_result_on_cpu, (void *)dst_gpu, size_a * sizeof(float)).wait();

    std::cout << "*****vectorization warmup stage*****" << std::endl;
    for (int it = 0; it < 5; it++) {
        auto t1 = std::chrono::steady_clock::now();   // Start timing
        auto event = q.submit([&](handler& h) {
            h.parallel_for(size, [=](id<1> idx) SYCL_ESIMD_KERNEL {
                using namespace sycl::ext::intel::esimd;
                int i = idx[0];
                // float mean = 0.0f;
                simd<float, VL> mean_v(0.0f);
                // for (size_t i = 0; i < VL; i++) {
                //     mean_v[i] = 0.0f;
                // }
                // float var = 0.0f;
                simd<float, VL> var_v(0.0f);
                // for (size_t i = 0; i < VL; i++) {
                //     var_v[i] = 0.0f;
                // }
                for (int j = 0; j < N / 2; j += VL) {
                    simd<float, VL> left(src_gpu + i * N + j);
                    simd<float, VL> right(src_gpu + i * N + j + N / 2);
                    auto x0 = sycl::ext::intel::esimd::exp(left * -1.0f);
                    auto x1 = x0 + 1.0f;
                    auto x2 = left / x1;
                    auto res = right * x2;
                    if (i == 0 && j == 0) {
                        float tmp_p = res.select<1,1>(0);
                        sycl::ext::oneapi::experimental::printf("vec res[0]:%f\n", tmp_p);
                    }
                    res.copy_to(dst_gpu + i * N / 2 + j);
                }
                // mean_v /= N;
                // float mean = sum<float, float, VL>(mean_v);
                // float var = sum<float, float, VL>(var_v);
                // var = var / N - mean * mean;
                // auto rsqrt_var = sycl::ext::intel::esimd::rsqrt(var + eps);
                // if (i == 0) {
                //     sycl::ext::oneapi::experimental::printf("mean:%f rsqrt_var:%f\n", mean, rsqrt_var);
                // }

                // // layernorm
                // for (int j = 0; j < N; j += VL) {
                //     simd<float, VL> gamma_gpu_v(gamma_gpu + j);
                //     simd<float, VL> amplifier_v = gamma_gpu_v * rsqrt_var;

                //     simd<float, VL> beta_gpu_v(beta_gpu + j);
                //     simd<float, VL> offset_v = amplifier_v * mean - beta_gpu_v;
                //     simd<float, VL> tmp(src_gpu + i * N + j);
                //     simd<float, VL> dst_gpu_v = amplifier_v * tmp - offset_v;
                //     // if (i == 0 && j == 8) {
                //     //     float tmp_p = dst_gpu_v.select<1,1>(7);
                //     //     sycl::ext::oneapi::experimental::printf("vec dst_gpu[15]:%f\n", tmp_p);
                //     // }
                //     dst_gpu_v.copy_to(dst_gpu + i * N + j);
                // }
            });
        });
        event.wait();
        auto t2 = std::chrono::steady_clock::now();   // Stop timing
        auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "warmup iter:" << it << " sync kernel time: " << exec_time << "us ";

        auto gpu_start = event.template get_profiling_info<
                sycl::info::event_profiling::command_start>();
        auto gpu_end = event.template get_profiling_info<
                sycl::info::event_profiling::command_end>();
        auto exec_time_prof = (gpu_end - gpu_start) / 1000.0;
        std::cout << " pure gpu kernel time: " << exec_time_prof << "us" << std::endl;
    }

    q.memcpy((void *)dst_gpu_result_on_cpu, (void *)dst_gpu, size_a * sizeof(float)).wait();
    float total_var = 0.0f;
    // validate result
    for (size_t i = 0; i < size_a; i++) {
        float v1 = dst_gpu_result_on_cpu[i];
        float v2 = dst[i];
        auto delt = v1 - v2;
        auto delt_2 = delt * delt;
        // if (delt_2 > 0.00001) {
        //     std::cout << "i:" << i << " delt:" << delt << " v2:" << v2 << std::endl;
        //     // std::cout << "v1:" << v1 << " v2:" << v2 << std::endl;
        // }
        total_var += delt_2;
    }
    std::cout << "total error:" << total_var << std::endl;

    free(src_gpu, q);
    free(dst_gpu, q);

    free(src);
    free(dst);

    return 0;
}

