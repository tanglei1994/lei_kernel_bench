#include "utils.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
    constexpr size_t Size = 1024;
    constexpr size_t VL = 4;
    property_list properties{property::queue::enable_profiling()};
    queue q(properties);
    float *A = malloc_shared<float>(Size, q);
    float *B = malloc_shared<float>(Size, q);
    float *C = malloc_shared<float>(Size, q);

    for (unsigned i = 0; i != Size; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(
        Size / VL, [=](id<1> i) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        auto offset = i * VL;
        // pointer arithmetic, so offset is in elements:
        simd<float, VL> va(A + offset);
        simd<float, VL> vb(B + offset);
        simd<float, VL> vc = va + vb;
        vc.copy_to(C + offset);
    });
    }).wait_and_throw();

    for (unsigned i = 0; i != Size; i++) {
        std::cout << "C[" << i << "]:" << C[i] << std::endl; 
    }


    return 0;
}
