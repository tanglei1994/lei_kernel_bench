#include "utils.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/math.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

#define M 4096
// #define M 32
#define N 128
// #define N 16
// #define N 64

static constexpr unsigned VL = 32; // useless in fma ??
static constexpr unsigned gather_len = 8;

void transpose_cpu(float *src, float * dst, int m, int n) {
  // __m512i offsets;
  // for (int k = 0; k < 16; k++) {
  //   offsets[k] = k + 1;
  // }
// #pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      // auto a_v_512 = _mm512_loadu_ps(src + i * n + j);
      // _mm512_i32scatter_ps((void *)(dst + i), offsets, a_v_512, 4);
      dst[j * m + i] = src[i * n + j];
    }
  }
}

// A: 1 x N
// B: N x M
auto esimd_gemv(queue &q, float* A, float* B, float* output) {
  constexpr unsigned VL = 32;
  unsigned GROUPSIZE = N / VL;
 
  nd_range<1> NDR{range<1>{unsigned(M * GROUPSIZE)}, range<1>{GROUPSIZE}};
 
  event gemv_t = q.submit([&](handler &h) {
    h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      slm_init<64>();
      int local_id = idx.get_local_linear_id();
      int group_id = idx.get_group_linear_id();
      auto offset_A = local_id * VL;
      auto offset_B_base = group_id;
 
      simd<float, VL> acc = 0.0f;
      simd<int, 32> offsets_B;
      for (int k = 0; k < 32; k++) {
        offsets_B[k] = M * k * 4;
      }
// #pragma unroll
      // for (int j = 0; j < 1; ++j) {
      simd<float, VL> A_arr(A + offset_A);
      auto B_arr = gather(B + offset_B_base + local_id * VL * M, offsets_B);
      acc += A_arr * B_arr;
      // }
      slm_scalar_store<float>(
          local_id * 4,
          sycl::ext::intel::esimd::detail::sum<float, float, VL>(acc));
      idx.barrier();
 
      auto sum_val = 0.f;
#pragma unroll
      for (int i = 0; i < GROUPSIZE; ++i) {
        sum_val += slm_scalar_load<float>(i * 4);
      }
      idx.barrier();
 
      if (local_id == 0) {
        output[group_id] = sum_val;
      }
    });
  });
 
  return gemv_t;
}

// bA: batch x 1 x M
// bB: batch x M x N
// b_output: batch x N
// batch work-group, N work-item per work-group 
auto esimd_batch_gemv_v1(queue &q, float* bA, float* bB, float* b_output, size_t batch) {
  constexpr unsigned gather_len = 32;
  unsigned GROUPSIZE = N;
 
  nd_range<1> NDR{range<1>{unsigned(batch * GROUPSIZE)}, range<1>{GROUPSIZE}};
 
  event gemv_t = q.submit([&](handler &h) {
    h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      int local_id = idx.get_local_linear_id();
      int group_id = idx.get_group_linear_id();
      auto offset_A = group_id * M;
      auto offset_B_base = group_id * M * N + local_id;
 
      simd<float, gather_len> acc = 0.0f;
      simd<int, gather_len> offsets_B;
      for (int k = 0; k < gather_len; k++) {
        offsets_B[k] = N * k * sizeof(float);
      }

#pragma unroll
      for (int i = 0; i < M; i += gather_len) {
          simd<float, gather_len> A_arr(bA + offset_A + i);
          auto B_arr = gather(bB + offset_B_base + i, offsets_B);
          acc += A_arr * B_arr;
      }

      auto sum_val = 0.f;
      sum_val = sycl::ext::intel::esimd::detail::sum<float, float, gather_len>(acc);
      b_output[group_id * N + local_id] = sum_val;
    });
  });
 
  return gemv_t;
}

// bA: batch x 1 x M
// bB: batch x M x N
// b_output: batch x N
// batch work-group, M / VL work-item per work-group
// N should be power of 2
auto esimd_batch_gemv_v2_fma(queue &q, float* bA, float* bB, float* b_output, size_t batch) {
  unsigned GROUPSIZE = M / VL;
 
  nd_range<1> NDR{range<1>{unsigned(batch * GROUPSIZE)}, range<1>{GROUPSIZE}};
 
  event gemv_t = q.submit([&](handler &h) {
    h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      slm_init<512 * sizeof(float)>();
      int local_id = idx.get_local_linear_id();
      int group_id = idx.get_group_linear_id();
      auto offset_A = group_id * M;
      auto offset_B_base = group_id * M * N + local_id * VL * N;
 
      simd<float, N> acc = 0.0f;
#pragma unroll
      for (int i = 0; i < VL; i++) {
        simd<float, N> B_arr(bB + offset_B_base + i * N);
        acc += bA[offset_A + local_id * VL + i] * B_arr;
      }
      constexpr int max_slm_bs_len = 8 * 8 * 2 / sizeof(float);
      for (int k = 0; k < N; k += max_slm_bs_len) {
        simd<float, max_slm_bs_len> tmp = acc.select<max_slm_bs_len, 1>(k);
        slm_block_store<float>(local_id * N * sizeof(float) + k * sizeof(float), tmp);
      }
      idx.barrier();

      // not sure if we can utilize reduce primitive inside EU to optimize this sequential procedure.
      simd<float, N> sum_val = 0.f;
      constexpr int max_slm_bl_len = 16 * 8 * 2 / sizeof(float);
#pragma unroll // will triger the compiler's warning which says loop is not unrolled.
      for (int i = 0; i < GROUPSIZE; i++) {
        for (int k = 0; k < N; k += max_slm_bl_len) {
          sum_val.select<max_slm_bl_len, 1>(k) += slm_block_load<float, max_slm_bl_len>(i * N * sizeof(float) + k * sizeof(float));
        }
      }

      // divergence ?
      if (local_id == 0) {
        sum_val.copy_to(b_output + group_id * N);
      }
    });
  });
 
  return gemv_t;
}

// bA: batch x 1 x M
// bB: batch x M x N
// b_output: batch x N
// batch work-group, M / VL work-item per work-group 
auto esimd_batch_gemv_v2_gather(queue &q, float* bA, float* bB, float* b_output, size_t batch) {
  unsigned GROUPSIZE = M / VL;
 
  nd_range<1> NDR{range<1>{unsigned(batch * GROUPSIZE)}, range<1>{GROUPSIZE}};
 
  event gemv_t = q.submit([&](handler &h) {
    h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      slm_init<512 * sizeof(float)>();
      int local_id = idx.get_local_linear_id();
      int group_id = idx.get_group_linear_id();
      auto offset_A = group_id * M;
      auto offset_B_base = group_id * M * N + local_id * VL * N;
 
      simd<int, gather_len> offsets_B;
      for (int k = 0; k < gather_len; k++) {
        offsets_B[k] = N * k * sizeof(float);
      }

#pragma unroll
      for (int i = 0; i < N; i++) {
        simd<float, gather_len> acc = 0.0f;
        for (int j = 0; j < VL; j += gather_len) {
          simd<float, gather_len> A_arr(bA + offset_A + local_id * VL + j);
          auto B_arr = gather(bB + offset_B_base + j * N + i, offsets_B);
          acc += A_arr * B_arr;
        }
        slm_scalar_store<float>(local_id * N * sizeof(float) + i * sizeof(float),
              sycl::ext::intel::esimd::detail::sum<float, float, gather_len>(acc));
      }

      idx.barrier();

      constexpr int max_bs_len = 8 * 8 * 2 / sizeof(float);
      constexpr int max_slm_bl_len = 16 * 8 * 2 / sizeof(float);
      constexpr int max_mem_access_len = std::min(max_bs_len, max_slm_bl_len);
      // can be paralleled in a work group ??
      for (int j = 0; j < N; j += max_mem_access_len) {
        simd<float, max_mem_access_len> tmp = 0.0f;
        for (int i = 0; i < GROUPSIZE; i++) {
          tmp += slm_block_load<float, max_mem_access_len>(i * N * sizeof(float) + j * sizeof(float)); 
        }
        if (local_id == 0) {
          block_store<float, max_mem_access_len>(b_output + group_id * N + j, tmp);
        }
      }
    });
  });
 
  return gemv_t;
}


// A: 1 x N
// B: M x N
auto esimd_gemv_Transposed(queue &q, float* A, float* B, float* output) {
  constexpr unsigned VL = 32;
  unsigned GROUPSIZE = N / VL;
 
  nd_range<1> NDR{range<1>{unsigned(M * GROUPSIZE)}, range<1>{GROUPSIZE}};
 
  event gemv_t = q.submit([&](handler &h) {
    h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      slm_init<64>();
      int local_id = idx.get_local_linear_id();
      int group_id = idx.get_group_linear_id();
      auto offset_A = local_id * VL;
      auto offset_B = group_id * N + offset_A;
 
      simd<float, VL> acc = 0.0f;
// #pragma unroll
      // for (int j = 0; j < 1; ++j) {
      simd<float, VL> A_arr(A + offset_A);
      simd<float, VL> B_arr(B + offset_B);
      acc += A_arr * B_arr;
      // }
      slm_scalar_store<float>(
          local_id * 4,
          sycl::ext::intel::esimd::detail::sum<float, float, VL>(acc));
      idx.barrier();
 
      auto sum_val = 0.f;
#pragma unroll
      for (int i = 0; i < GROUPSIZE; ++i) {
        sum_val += slm_scalar_load<float>(i * 4);
      }
      idx.barrier();
 
      if (local_id == 0) {
        output[group_id] = sum_val;
      }
    });
  });
 
  return gemv_t;
}

// A: 1 x N
// B: N x M
auto esimd_gemv_flatten(queue &q, float* A, float* B, float* output) {
  constexpr unsigned VL = 32;
  range<1> size{M};
 
  event gemv_t = q.submit([&](handler &h) {
    h.parallel_for(size, [=](item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      int group_id = idx[0];
      auto offset_B_base = group_id;
 
      simd<float, VL> acc = 0.0f;
      simd<int, 32> offsets_B;
      for (int k = 0; k < 32; k++) {
        offsets_B[k] = M * k * 4;
      }
#pragma unroll
      for (int j = 0; j < N; j += 32) {
        simd<float, VL> A_arr(A + j);
        auto B_arr = gather(B + offset_B_base + j * M, offsets_B);
        acc += A_arr * B_arr;
      }
 
      float sum_val = 
          sycl::ext::intel::esimd::detail::sum<float, float, VL>(acc);
 
      output[group_id] = sum_val;
    });
  });
 
  return gemv_t;
}

// A: 1 x N
// B: M x N
auto esimd_gemv_Transposed_flatten(queue &q, float* A, float* B, float* output) {
  constexpr unsigned VL = 32;
  range<1> size{M};
 
  event gemv_t = q.submit([&](handler &h) {
    h.parallel_for(size, [=](item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      int group_id = idx[0];
      auto offset_B_base = group_id * N;
 
      simd<float, VL> acc = 0.0f;
#pragma unroll
      for (int j = 0; j < N; j += VL) {
        simd<float, VL> A_arr(A + j);
        simd<float, VL> B_arr(B + offset_B_base + j);
        acc += A_arr * B_arr;
      }
 
      float sum_val = 
          sycl::ext::intel::esimd::detail::sum<float, float, VL>(acc);
      output[group_id] = sum_val;
    });
  });
 
  return gemv_t;
}


int main() {
    property_list properties{property::queue::enable_profiling()};
    queue q(properties);
    auto context = q.get_info<info::queue::context>();
    auto device = q.get_info<info::queue::device>();

    constexpr size_t iter_times = 5;

    constexpr size_t batch_size = 16 * 32;
    constexpr size_t size_a = 1 * M;
    constexpr size_t size_batch_a = batch_size * size_a;
    constexpr size_t size_b = M * N;
    constexpr size_t size_batch_b = batch_size * size_b;
    constexpr size_t size_o = N;
    constexpr size_t size_batch_o = batch_size * size_o;
    // auto src = alloc_host_and_init<float>(
    //     size_a,
    //     [](float* data, size_t idx) {
    //         data[idx] = generate_random<float>();
    //         // data[idx] = 1.0f;
    //     });
    // auto src_gpu = static_cast<float *>(aligned_alloc_device(
    //     DEVICE_MEM_ALIGNMENT, size_a * sizeof(float), device, context));
    // q.memcpy((void *)src_gpu, (void *)src, size_a * sizeof(float)).wait();

    auto batch_src = alloc_host_and_init<float>(
        size_batch_a,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
            // data[idx] = 1.0f;
        });
    auto batch_src_gpu = static_cast<float *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_batch_a * sizeof(float), device, context));
    q.memcpy((void *)batch_src_gpu, (void *)batch_src, size_batch_a * sizeof(float)).wait();

    // auto weight = alloc_host_and_init<float>(
    //     size_b,
    //     [](float* data, size_t idx) {
    //         data[idx] = generate_random<float>();
    //         // data[idx] = 1.0f;
    //     });
    // auto weight_gpu = static_cast<float *>(aligned_alloc_device(
    //     DEVICE_MEM_ALIGNMENT, size_b * sizeof(float), device, context));
    // q.memcpy((void *)weight_gpu, (void *)weight, size_b * sizeof(float)).wait();

    auto batch_weight = alloc_host_and_init<float>(
        size_batch_b,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
            // data[idx] = 1.0f;
        });
    auto batch_weight_gpu = static_cast<float *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_batch_b * sizeof(float), device, context));
    q.memcpy((void *)batch_weight_gpu, (void *)batch_weight, size_batch_b * sizeof(float)).wait();

    // auto weight_T = alloc_host_and_init<float>(
    //     size_b,
    //     [](float* data, size_t idx) {
    //         data[idx] = 0.0f;
    //     });
    
    // std::cout << "tanglei 1" << std::endl;
    // transpose_cpu(weight, weight_T, M, N);
    // for (int i = 0; i < M; i++) {
    //   for (int j = 0; j < N; j++) {
    //     std::cout << weight[i * N + j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "tanglei 2" << std::endl;
    // for (int i = 0; i < N; i++) {
    //   for (int j = 0; j < M; j++) {
    //     std::cout << weight_T[i * M + j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // auto weight_T_gpu = static_cast<float *>(aligned_alloc_device(
    //     DEVICE_MEM_ALIGNMENT, size_b * sizeof(float), device, context));
    // q.memcpy((void *)weight_T_gpu, (void *)weight_T, size_b * sizeof(float)).wait();
      
    // auto dst = alloc_host_and_init<float>(
    //     size_o,
    //     [](float* data, size_t idx) {
    //         data[idx] = 0.0f;
    //     });
    // auto dst_gpu = static_cast<float *>(aligned_alloc_device(
    //     DEVICE_MEM_ALIGNMENT, size_o * sizeof(float), device, context));
    // q.memcpy((void *)dst_gpu, (void *)dst, size_o * sizeof(float)).wait();

    auto batch_dst = alloc_host_and_init<float>(
        size_batch_o,
        [](float* data, size_t idx) {
            data[idx] = 0.0f;
        });
    auto batch_dst_gpu = static_cast<float *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_batch_o * sizeof(float), device, context));
    q.memcpy((void *)batch_dst_gpu, (void *)batch_dst, size_batch_o * sizeof(float)).wait();

    // auto dst_gpu_T = static_cast<float *>(aligned_alloc_device(
    //     DEVICE_MEM_ALIGNMENT, size_o * sizeof(float), device, context));
    // q.memcpy((void *)dst_gpu_T, (void *)dst, size_o * sizeof(float)).wait();

    // auto dst_gpu_result_on_cpu = alloc_host_and_init<float>(
    //     size_o,
    //     [](float* data, size_t idx) {
    //         data[idx] = 0.0f;
    //     });

    // auto dst_gpu_T_result_on_cpu = alloc_host_and_init<float>(
    //     size_o,
    //     [](float* data, size_t idx) {
    //         data[idx] = 0.0f;
    //     });

    auto batch_dst_gpu_result_on_cpu_1 = alloc_host_and_init<float>(
        size_batch_o,
        [](float* data, size_t idx) {
            data[idx] = 0.0f;
        });

    auto batch_dst_gpu_result_on_cpu_2 = alloc_host_and_init<float>(
        size_batch_o,
        [](float* data, size_t idx) {
            data[idx] = 0.0f;
        });
    std::cout << "***** fma *****" << std::endl;
    for (int it = 0; it < iter_times; it++) {
        auto t1 = std::chrono::steady_clock::now();   // Start timing
        // auto event = esimd_gemv_Transposed_flatten(q, src_gpu, weight_gpu, dst_gpu); 
        auto event = esimd_batch_gemv_v2_fma(q, batch_src_gpu, batch_weight_gpu, batch_dst_gpu, batch_size); 
        event.wait();
        auto t2 = std::chrono::steady_clock::now();   // Stop timing
        auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "iter:" << it << " sync kernel time: " << exec_time << "us ";

        auto gpu_start = event.template get_profiling_info<
                sycl::info::event_profiling::command_start>();
        auto gpu_end = event.template get_profiling_info<
                sycl::info::event_profiling::command_end>();
        auto exec_time_prof = (gpu_end - gpu_start) / 1000.0;
        std::cout << " pure gpu kernel time: " << exec_time_prof << "us" << std::endl;
    }
    q.memcpy((void *)batch_dst_gpu_result_on_cpu_1, (void *)batch_dst_gpu, size_o * sizeof(float)).wait();

    std::cout << "***** gather *****" << std::endl;
    for (int it = 0; it < iter_times; it++) {
        auto t1 = std::chrono::steady_clock::now();   // Start timing
        // auto event = esimd_gemv_flatten(q, src_gpu, weight_T_gpu, dst_gpu_T); 
        auto event = esimd_batch_gemv_v2_gather(q, batch_src_gpu, batch_weight_gpu, batch_dst_gpu, batch_size); 
        event.wait();
        auto t2 = std::chrono::steady_clock::now();   // Stop timing
        auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "iter:" << it << " sync kernel time: " << exec_time << "us ";

        auto gpu_start = event.template get_profiling_info<
                sycl::info::event_profiling::command_start>();
        auto gpu_end = event.template get_profiling_info<
                sycl::info::event_profiling::command_end>();
        auto exec_time_prof = (gpu_end - gpu_start) / 1000.0;
        std::cout << " pure gpu kernel time: " << exec_time_prof << "us" << std::endl;
    }
    q.memcpy((void *)batch_dst_gpu_result_on_cpu_2, (void *)batch_dst_gpu, size_o * sizeof(float)).wait();

    float total_var = 0.0f;
    // validate result
    total_var = abs_error(batch_dst_gpu_result_on_cpu_1, batch_dst_gpu_result_on_cpu_2, size_batch_o);
    std::cout << "total error:" << total_var << std::endl;

    // free(src_gpu, q);
    free(batch_src_gpu, q);
    // free(weight_gpu, q);
    free(batch_weight_gpu, q);
    // free(dst_gpu, q);
    free(batch_dst_gpu, q);

    // free(src);
    free(batch_src);
    // free(weight);
    free(batch_weight);
    // free(dst);
    free(batch_dst);
    // free(dst_gpu_result_on_cpu);
    free(batch_dst_gpu_result_on_cpu_1);
    free(batch_dst_gpu_result_on_cpu_2);

    return 0;
}
