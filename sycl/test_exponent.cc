#include <CL/sycl.hpp>
#include <ext/intel/esimd.hpp>

using namespace sycl;

int main() {

    constexpr int size = 64;
    constexpr int repeat = 1000;
    // std::array<double, size> A;
    std::array<float, size> B;
    // bool pass = true;

    for (int i = 0; i < size; ++i) {
        // A[i] = i;
        B[i] = 0.0f;
    }

    default_selector d_selector;
    queue Q{d_selector, property::queue::enable_profiling{}};
    range sz{size};
    // buffer<double> bufA(A);
    buffer<float> bufB(B);
    // buffer<bool> bufP(&pass, 1);

    // warm up
    event e0 = Q.submit([&](handler &h) {
        // accessor accA{ bufA, h};
        accessor accB{ bufB, h};
        // accessor accP{ bufP, h};
        h.parallel_for(size, [=](id<1> idx) {
            // accA[idx] = std::exp(accA[idx]);
            accB[idx] = sycl::exp(accB[idx]);
            // accB[idx] = sycl::ext::intel::esimd::exp(accB[idx]);
            // if (!sycl::isequal( accA[idx], accB[idx]) ) {
            //     // sycl::ext::oneapi::experimental::printf("hello sycl !\n");
            //     accP[0] = false;
            // }
        });
    });
    e0.wait();
    auto t1 = std::chrono::steady_clock::now();   // Start timing
    event e1 = Q.submit([&](handler &h) {
        // accessor accA{ bufA, h};
        accessor accB{ bufB, h};
        // accessor accP{ bufP, h};
        h.parallel_for(size, [=](id<1> idx) {
            // accA[idx] = std::exp(accA[idx]);
            accB[idx] = sycl::exp(accB[idx]);
            // if (!sycl::isequal( accA[idx], accB[idx]) ) {
            //     // sycl::ext::oneapi::experimental::printf("hello sycl !\n");
            //     accP[0] = false;
            // }
        });
    });
    e1.wait();
    auto t2 = std::chrono::steady_clock::now();   // Stop timing
    auto exec_time_0 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "kernel0 time: " << exec_time_0 << "us" << std::endl;
    event e2 = Q.submit([&](handler &h) {
        // accessor accA{ bufA, h};
        accessor accB{ bufB, h};
        // accessor accP{ bufP, h};
        h.parallel_for(size, [=](id<1> idx) {
            // accA[idx] = std::exp(accA[idx]);
            sycl::ext::oneapi::experimental::printf("idx:%d\n", idx[0]);
            // float tmp = accB[idx];
            float tmp = accB[idx[0]];
            float res = 0.0f;
            float delta = 0.0f;
            // for (int i = 0; i < repeat; i++) {
            for (int i = 0; i < repeat; i++, tmp+=delta) {
                res = sycl::exp(tmp);
                delta = 1 - ((res - 0.001) / res);
            }
            accB[idx[0]] = res;
            // if (!sycl::isequal( accA[idx], accB[idx]) ) {
            //     // sycl::ext::oneapi::experimental::printf("hello sycl !\n");
            //     accP[0] = false;
            // }
        });
    });
    e2.wait();
    auto gpu_start = e2.template get_profiling_info<
            sycl::info::event_profiling::command_start>();
    auto gpu_end = e2.template get_profiling_info<
            sycl::info::event_profiling::command_end>();
    auto exec_time = (gpu_end - gpu_start) / 1000.0;
    std::cout << "kernel1 time: " << exec_time << "us" << std::endl;
    // sycl::ext::oneapi::experimental::printf("hello sycl %d\n", pass);
}