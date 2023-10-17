#include "utils.hpp"

using namespace sycl;

#define M 512
#define K 384
#define N 1024

int main() {

    property_list properties{property::queue::enable_profiling()};
    queue q(properties);
    auto context = q.get_info<info::queue::context>();
    auto device = q.get_info<info::queue::device>();

    constexpr size_t size_a = M * K;
    constexpr size_t size_b = K * N;
    constexpr size_t size_c = M * N;
    auto A = alloc_device_and_init<float>(
        size_a,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
        },
        q, device, context);
    auto B = alloc_device_and_init<float>(
        size_b,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
        },
        q, device, context);
    auto C = alloc_device_and_init<float>(
        size_c,
        [](float* data, size_t idx) {
            data[idx] = 0.0f;
        },
        q, device, context);
    auto bias = alloc_device_and_init<float>(
        N,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
        },
        q, device, context);

    range size{M, N};
    auto t1 = std::chrono::steady_clock::now();   // Start timing
    auto e1= q.submit([&](handler& h) {
        h.parallel_for(size, [=](id<2> idx) {
            int i = idx[0];
            int j = idx[1];
            for (int k = 0; k < K; k++) {
                C[i * N + j] = A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] += bias[j];
        });
    });
    e1.wait();
    auto t2 = std::chrono::steady_clock::now();   // Stop timing
    auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "kernel1 time: " << exec_time << "us" << std::endl;

    t1 = std::chrono::steady_clock::now();   // Start timing
    auto e2= q.submit([&](handler& h) {
        h.parallel_for(size, [=](id<2> idx) {
            int i = idx[0];
            int j = idx[1];
            for (int k = 0; k < K; k++) {
                C[i * N + j] = A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] += bias[j];
        });
    });
    e2.wait();
    t2 = std::chrono::steady_clock::now();   // Stop timing
    exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "kernel2 time: " << exec_time << "us" << std::endl;

    t1 = std::chrono::steady_clock::now();   // Start timing
    auto e3= q.submit([&](handler& h) {
        h.parallel_for(size, [=](id<2> idx) {
            int i = idx[0];
            int j = idx[1];
            for (int k = 0; k < K; k++) {
                C[i * N + j] = A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] += bias[j];
        });
    });
    e3.wait();
    t2 = std::chrono::steady_clock::now();   // Stop timing
    exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "kernel3 time: " << exec_time << "us" << std::endl;

    auto gpu_start = e3.template get_profiling_info<
            sycl::info::event_profiling::command_start>();
    auto gpu_end = e3.template get_profiling_info<
            sycl::info::event_profiling::command_end>();
    auto exec_time_prof = (gpu_end - gpu_start) / 1000.0;
    std::cout << "kernel3 time: " << exec_time_prof << "us" << std::endl;

    free(A, context);
    free(B, context);
    free(C, context);
    free(bias, context);
    return 0;
}