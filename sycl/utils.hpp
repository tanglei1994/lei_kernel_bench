#pragma once

#include <iostream>
#include <random>
#include <CL/sycl.hpp>
#include <omp.h>
#include <immintrin.h>
#include <cmath>

using namespace sycl;

#define DEVICE_MEM_ALIGNMENT (64)

template <typename rtype>
inline rtype generate_random(rtype a = 0.0, rtype b = 1.0) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // std::cout << "seed:" << seed << std::endl;
    // unsigned seed = 202310;
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<rtype> distribution(a, b);

    return distribution(engine);
}

template <typename dtype>
inline dtype *alloc_host_and_init(size_t size,
        std::function<void(dtype *data, size_t idx)> init_func) {
    auto host_ptr = static_cast<dtype *>(malloc(size * sizeof(dtype)));

    for (size_t i = 0; i < size; i++) {
        init_func(host_ptr, i);
    }

    return host_ptr;
}

template <typename dtype>
inline dtype *alloc_device_and_init(size_t size,
        std::function<void(dtype *data, size_t idx)> init_func,
        queue& q, device& device, context& context) {
    auto host_ptr = static_cast<dtype *>(malloc(size * sizeof(dtype)));

    for (size_t i = 0; i < size; i++) {
        init_func(host_ptr, i);
    }

    auto device_ptr = static_cast<dtype *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size * sizeof(dtype), device, context));
    
    q.memcpy((void *)device_ptr, (void *)host_ptr, size * sizeof(dtype)).wait();

    free(host_ptr);

    return device_ptr;
}

inline __m512 _loadu(const float *data_base)
{
  return _mm512_loadu_ps(data_base);
}

inline void _storeu(float *data_base, __m512 a)
{
  _mm512_storeu_ps(data_base, a);
}

inline __m512 _dil_exp_kernel(__m512 vec_src)
{
  static __m512 vec_factorial_1 =
      _mm512_set1_ps(0.999999701f); // 1/factorial(1)
  static __m512 vec_factorial_2 =
      _mm512_set1_ps(0.499991506f); // 1/factorial(2)
  static __m512 vec_factorial_3 =
      _mm512_set1_ps(0.166676521f); // 1/factorial(3)
  static __m512 vec_factorial_4 =
      _mm512_set1_ps(0.0418978221f); // 1/factorial(4)
  static __m512 vec_factorial_5 =
      _mm512_set1_ps(0.00828929059f); // 1/factorial(5)
  static __m512 vec_exp_log2ef =
      (__m512)_mm512_set1_epi32(0x3fb8aa3b); // log2(e)
  static __m512 vec_half = _mm512_set1_ps(0.5f);
  static __m512 vec_one = _mm512_set1_ps(1.f);
  static __m512 vec_zero = _mm512_set1_ps(0.f);
  static __m512 vec_two = _mm512_set1_ps(2.f);
  static __m512 vec_ln2f = (__m512)_mm512_set1_epi32(0x3f317218); // ln(2)
  static __m512 vec_ln_flt_min = (__m512)_mm512_set1_epi32(0xc2aeac50);
  static __m512 vec_ln_flt_max = (__m512)_mm512_set1_epi32(0x42b17218);
  static __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
  static int n_mantissa_bits = 23;

  // exp(x) =
  // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
  // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

  auto less_ln_flt_min_mask =
      _mm512_cmp_ps_mask(vec_src, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
  vec_src = _mm512_min_ps(vec_src, vec_ln_flt_max);
  vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

  // fx = floorf(x * log2ef + 0.5)
  auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
  auto vec_fx_i = _mm512_cvt_roundps_epi32(vec_fx, _MM_FROUND_TO_NEG_INF |
                                                       _MM_FROUND_NO_EXC);
  vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

  // x = x - fx * ln2
  auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

  // compute polynomial
  auto vec_res =
      _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);
  // compute 2^(n-1)
  auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
  auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
  auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
  vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
  auto vec_two_pow_n = (__m512)vec_two_pow_n_i;
  vec_two_pow_n =
      _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

  // y = y * 2^n
  vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
  vec_res = _mm512_mul_ps(vec_res, vec_two);
  return vec_res;
}

template<typename T>
T abs_error(T* src1, T* src2, int size) {
    T total_error = 0.0f;
    for (size_t i = 0; i < size; i++) {
        T v1 = src1[i];
        T v2 = src2[i];
        auto delt = v1 - v2;
        if (i == 111) {
            std::cout << "i:" << i << std::endl;
            std::cout << "src1[i]:" << v1 << std::endl;
            std::cout << "src2[i]:" << v2 << std::endl;
        }
        total_error += abs(delt);
    }
    return total_error;
}

template<typename T>
T square_error(T* src1, T* src2, int size) {
    T total_error = 0.0f;
    for (size_t i = 0; i < size; i++) {
        T v1 = src1[i];
        T v2 = src2[i];
        auto delt = v1 - v2;
        total_error += (delt * delt);
    }
    return total_error;
}
