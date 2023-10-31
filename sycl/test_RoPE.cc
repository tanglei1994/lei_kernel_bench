#include "utils.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/math.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

#define M 512
#define N (4096 * 3)

int main() {
    float eps = 0.0001f;
    constexpr size_t VL = 16;
    int num_head = 16;
    int multi_query_group_num = 8;
    property_list properties{property::queue::enable_profiling()};
    queue q(properties);
    auto context = q.get_info<info::queue::context>();
    auto device = q.get_info<info::queue::device>();

    constexpr size_t size_qkv = M * N;
    auto qkv = alloc_host_and_init<float>(
        size_qkv,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
        });
    auto qkv_gpu = static_cast<float *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_qkv * sizeof(float), device, context));
    q.memcpy((void *)qkv_gpu, (void *)qkv, size_qkv * sizeof(float)).wait();

    auto qkv_gpu_result_on_cpu = alloc_host_and_init<float>(
        size_qkv,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
        });

    auto cos_sin = alloc_host_and_init<float>(
        size_qkv,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>(-1.0, 1.0);
        });
    auto cos_sin_gpu = static_cast<float *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_qkv * sizeof(float), device, context));
    q.memcpy((void *)cos_sin_gpu, (void *)cos_sin, size_qkv * sizeof(float)).wait();

    int batchSize = M;
    int qkvOutSize = N;
    int headSize = qkvOutSize / (num_head + 2 * multi_query_group_num);
    int qk_head = num_head + multi_query_group_num;
    int rotateSize = headSize / 4;
    for (size_t it = 0; it < 5; it++) {
        auto t1 = std::chrono::steady_clock::now();   // Start timing
#pragma omp parallel for simd
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < qk_head; ++j) {
                for (int len = 0; len < rotateSize * 2; len += 2) {
                    auto cos_sin_ptr = cos_sin + i * rotateSize * 2 + len;
                    auto qk_ptr = qkv + i * qkvOutSize + j * headSize + len;

                    auto qk_left = qk_ptr[0];
                    auto qk_right = qk_ptr[1];

                    auto cos_dat = cos_sin_ptr[0];
                    auto sin_dat = cos_sin_ptr[1];

                    qk_ptr[0] = qk_left * cos_dat - qk_right * sin_dat;
                    qk_ptr[1] = qk_right * cos_dat + qk_left * sin_dat;
                }
            }
        }

        auto t2 = std::chrono::steady_clock::now();   // Stop timing
        auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "warmup iter:" << it << " cpu kernel time: " << exec_time << "us " << std::endl;
    }

    range size{static_cast<unsigned long>(batchSize), static_cast<unsigned long>(qk_head)};
    std::cout << "*****vectorization warmup stage*****" << std::endl;
    for (int it = 0; it < 5; it++) {
        auto t1 = std::chrono::steady_clock::now();   // Start timing
        auto event = q.submit([&](handler& h) {
            // h.parallel_for(size, [=](id<2> idx) SYCL_ESIMD_KERNEL {
            //     using namespace sycl::ext::intel::esimd;
            //     int i = idx[0];
            //     int j = idx[1];

            //     for (int len = 0; len < rotateSize * 2; len += 2) {
            //         auto cos_sin_ptr = cos_sin_gpu + i * rotateSize * 2 + len;
            //         auto qk_ptr = qkv_gpu + i * qkvOutSize + j * headSize + len;
            //         auto qk_l = qk_ptr[0];
            //         auto qk_r = qk_ptr[1];
            //         auto cos = cos_sin_ptr[0];
            //         auto sin = cos_sin_ptr[1];
            //         qk_ptr[0] = qk_l * cos - qk_r * sin;
            //         qk_ptr[1] = qk_r * cos + qk_l * sin;
            //     }
            // });
            h.parallel_for(size, [=](id<2> idx) SYCL_ESIMD_KERNEL {
                using namespace sycl::ext::intel::esimd;
                int i = idx[0];
                int j = idx[1];

                simd<int, VL> offsets_r = 0;
                simd<int, VL> offsets_l = 0;
                for (int k = 0; k < VL; k++) {
                    offsets_r[k] = 2 * 4 * k + 4;
                    offsets_l[k] = 2 * 4 * k;
                }
                for (int len = 0; len < rotateSize * 2; len += 2 * VL) {
                    auto cos_sin_ptr = cos_sin_gpu + i * rotateSize * 2 + len;
                    auto qk_ptr = qkv_gpu + i * qkvOutSize + j * headSize + len;
                    auto qk_l = gather(qk_ptr, offsets_l);
                    auto qk_r = gather(qk_ptr, offsets_r);
                    auto cos = gather(cos_sin_ptr, offsets_l);
                    auto sin = gather(cos_sin_ptr, offsets_r);
                    auto qk_l_new = qk_l * cos - qk_r * sin;
                    auto qk_r_new = qk_r * cos + qk_l * sin;
                    scatter(qk_ptr, offsets_l, qk_l_new);
                    scatter(qk_ptr, offsets_r, qk_r_new);
                }
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

    q.memcpy((void *)qkv_gpu_result_on_cpu, (void *)qkv_gpu, size_qkv * sizeof(float)).wait();

    float total_var = 0.0f;
    // validate result
    for (size_t i = 0; i < size_qkv; i++) {
        float v1 = qkv_gpu_result_on_cpu[i];
        float v2 = qkv[i];
        // if (i == 123) {
        //     std::cout << "v1:" << v1 << std::endl;
        //     std::cout << "v2:" << v2 << std::endl;
        // }
        auto delt = v1 - v2;
        total_var += delt * delt;
    }
    std::cout << "total error:" << total_var << std::endl;
    free(qkv_gpu, q);
    free(cos_sin_gpu, q);

    free(qkv);
    free(qkv_gpu_result_on_cpu);
    free(cos_sin);

    return 0;
}

