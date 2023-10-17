#include "utils.hpp"

using namespace sycl;

#define M 512
#define N 1024

int main() {
    float eps = 0.001f;

    constexpr size_t size_a = M * N;
    auto src = alloc_host_and_init<float>(
        size_a,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
        });
    auto gamma = alloc_host_and_init<float>(
        N,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>(0.5, 1.0);
        });
    auto beta = alloc_host_and_init<float>(
        N,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
        });
    auto dst = alloc_host_and_init<float>(
        size_a,
        [](float* data, size_t idx) {
            data[idx] = 0.0f;
        });

    auto len_v = _mm512_set1_ps(N);
    auto zero_v = _mm512_setzero_ps();
    auto eps_v = _mm512_set1_ps(eps);
#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
    for (int i = 0; i < M; i++) {
        auto sum_mean = _mm512_setzero_ps();
        auto sum_var = _mm512_setzero_ps();
        for (int j = 0; j < N; j += 16) {
            auto dat = _mm512_loadu_ps(src + i * N + j);
            auto avg_dat = _mm512_div_ps(dat, len_v);
            sum_mean = _mm512_add_ps(sum_mean, avg_dat);
            sum_var = _mm512_fmadd_ps(avg_dat, dat, sum_var);
        }

        auto mean_val = _mm512_reduce_add_ps(sum_mean);
        auto var_val = _mm512_reduce_add_ps(sum_var) - mean_val * mean_val;
        
        auto mean = _mm512_set1_ps(mean_val);
        auto var = _mm512_rsqrt14_ps(_mm512_add_ps(eps_v, _mm512_set1_ps(var_val)));

        // layernorm 
        for (int j = 0; j < N; j += 16) {
            auto amplifier = _mm512_mul_ps(_mm512_loadu_ps(static_cast<float *>(gamma) + j), var);
            auto offset = _mm512_fmsub_ps(amplifier, mean, _mm512_load_ps(static_cast<float *>(beta) + j));
            auto dst_val = _mm512_fmsub_ps(amplifier, _mm512_loadu_ps(static_cast<float *>(src) + i * N + j), offset);
            _mm512_storeu_ps(static_cast<float *>(dst) + i * N + j, dst_val);
        }
    }


    free(src);
    free(gamma);
    free(beta);
    free(dst);

    return 0;
}
