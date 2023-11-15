#include "utils.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/math.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

#define M 512
// #define N 4109
// #define N 4351
// #define N 4097
#define N (2049)

// at::Tensor esimd_softmax_fp32(at::Tensor &src, at::Tensor &mask)
// {
//   auto m = unsigned(src.size(0) * src.size(1) * src.size(2));
//   int kvLen = src.size(3);
//   constexpr unsigned VL = 256;
//   unsigned GROUPSIZE = (kvLen - 1) / VL + 1;
//   auto seq_len = src.size(-1) % 256;

//   auto result = at::empty_like(src);

//   auto src_ptr = src.data_ptr<float>();
//   auto dst_ptr = result.data_ptr<float>();
//   auto mask_ptr = (const unsigned short *)(mask.data_ptr<short>());

//   nd_range<1> NDR{range<1>{m * GROUPSIZE}, range<1>{GROUPSIZE}};

//   event softmax = main_queue.submit([&](handler &h)
//                                     { h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL
//                                                      {
//       slm_init<256>();
//       int local_id = idx.get_local_linear_id();
//       int group_id = idx.get_group_linear_id();

//       auto offset = group_id * kvLen + local_id * VL;
//       simd<float, VL> max_arr = std::numeric_limits<float>::lowest();

//       simd<float, VL> src_arr(src_ptr + offset);
//       if (local_id == (GROUPSIZE - 1) && seq_len) {
//         simd_mask<VL> mask_arr;
//         mask_arr.copy_from(mask_ptr);
//         simd<float, VL> tmp = std::numeric_limits<float>::lowest();
//         tmp.merge(src_arr, mask_arr);
//         src_arr = tmp;
//       }
//       max_arr = sycl::ext::intel::esimd::max<float, VL>(src_arr, max_arr);
//       slm_scalar_store<float>(local_id * 4, hmax<float>(max_arr));
//       barrier();

//       auto max_val = std::numeric_limits<float>::lowest();
// #pragma unroll
//       for (int i = 0; i < GROUPSIZE; ++i) {
//         max_val = sycl::ext::intel::esimd::max<float>(
//             max_val, slm_scalar_load<float>(i * 4));
//       }
//       barrier();

//       src_arr = sycl::ext::intel::esimd::exp<float, VL>(src_arr - max_val);
//       slm_scalar_store<float>(local_id * 4, sycl::ext::intel::esimd::detail::sum<float, float, VL>(src_arr));
//       barrier();

//       auto sum_val = 0.f;
// #pragma unroll
//       for (int i = 0; i < GROUPSIZE; ++i) {
//         sum_val += slm_scalar_load<float>(i * 4);
//       }
//       barrier();

//       src_arr = src_arr / sum_val;
//       if (local_id == (GROUPSIZE - 1) && seq_len) {
//         simd<uint32_t, 32> res_offsets;
//         for(int k = 0; k < 32; ++k) {
//           res_offsets[k] = k * 4;
//         }
//         simd_mask<32> mask_arr;
//         auto src_res = src_arr.bit_cast_view<float, 8, 32>();
// #pragma unroll
//         for (int i = 0; i < 8; ++i) {
//           simd<float, 32> src_res_row = src_res.row(i);
//           mask_arr.copy_from(mask_ptr + i * 32);
//           scatter(dst_ptr + offset + i * 32, res_offsets, src_res_row, mask_arr);
//         }
//       } else {
//         src_arr.copy_to(dst_ptr + offset);
//       } }); });

//   softmax.wait();
//   return result;
// }

auto softmax_float_mask_lw(queue &main_queue, float *src, int batchSize,
                  int kvLen, unsigned short *mask_init, unsigned remainder)
{

  constexpr unsigned VL = 256;
  unsigned GROUPSIZE = (kvLen - 1) / VL + 1;
  auto seq_len = kvLen % 256;

  // auto result = at::empty_like(src);

  // auto src_ptr = src.data_ptr<float>();
  // auto dst_ptr = result.data_ptr<float>();
  // auto mask_ptr = (const unsigned short *)(mask.data_ptr<short>());

  nd_range<1> NDR{range<1>{batchSize * GROUPSIZE}, range<1>{GROUPSIZE}};

  event softmax = main_queue.submit([&](handler &h)
                                    { h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL
                                                     {
      using namespace sycl::ext::intel::esimd;
      slm_init<256>();
      int local_id = idx.get_local_linear_id();
      int group_id = idx.get_group_linear_id();

      auto offset = group_id * kvLen + local_id * VL;
      simd<float, VL> max_arr = std::numeric_limits<float>::lowest();

      simd<float, VL> src_arr(src + offset);
      if (local_id == (GROUPSIZE - 1) && seq_len) {
        simd_mask<VL> mask_arr;
        mask_arr.copy_from(mask_init);
        simd<float, VL> tmp = std::numeric_limits<float>::lowest();
        tmp.merge(src_arr, mask_arr);
        src_arr = tmp;
      }
      max_arr = sycl::ext::intel::esimd::max<float, VL>(src_arr, max_arr);
      slm_scalar_store<float>(local_id * 4, hmax<float>(max_arr));
      idx.barrier();

      auto max_val = std::numeric_limits<float>::lowest();

      // simd<float, 16> slm_max_arr_16, tmp_16 = max_val;
      // simd_mask<16> slm_max_arr_mask_16 = 1;

      // simd<float, 8> slm_max_arr_8, tmp_8 = max_val;
      // simd_mask<8> slm_max_arr_mask_8 = 1;
      // simd<float, 4> slm_max_arr_4, tmp_4 = max_val;
      // simd_mask<4> slm_max_arr_mask_4 = 0;

      // if (GROUPSIZE == 16) {
      //   slm_max_arr_16 = slm_block_load<float, 16>(0);
      //   max_val = std::max(max_val, hmax<float>(slm_max_arr_16));
      // } else if (GROUPSIZE > 8 && GROUPSIZE < 16) {
      //     for (int i = 8; i < GROUPSIZE; i++) {
      //       // if (group_id == 0 && local_id == 0) {
      //       //   sycl::ext::oneapi::experimental::printf("local_id:%d group_size:%d\n", local_id, GROUPSIZE);
      //       // }
      //       slm_max_arr_mask_16[i] = 0;
      //     }
      //     slm_max_arr_16 = slm_block_load<float, 16>(0);
      //     tmp_16.merge(slm_max_arr_16, slm_max_arr_mask_16);
      //     slm_max_arr_16 = tmp_16;
      //     max_val = std::max(max_val, hmax<float>(slm_max_arr_16));
      // }

      // switch (GROUPSIZE) {
      //   case 16: 
      //     slm_max_arr_16 = slm_block_load<float, 16>(0);
      //     max_val = std::max(max_val, hmax<float>(slm_max_arr_16));
      //     break;
        // case 15:
        // case 14:
        // case 13:
        // case 12:
        // case 11:
        // case 10:
        // case 9:
        //   for (int i = 8; i < GROUPSIZE; i++) {
        //     // if (group_id == 0 && local_id == 0) {
        //     //   sycl::ext::oneapi::experimental::printf("local_id:%d group_size:%d\n", local_id, GROUPSIZE);
        //     // }
        //     slm_max_arr_mask_16[i] = 0;
        //   }
        //   slm_max_arr_16 = slm_block_load<float, 16>(0);
        //   tmp_16.merge(slm_max_arr_16, slm_max_arr_mask_16);
        //   slm_max_arr_16 = tmp_16;
        //   max_val = std::max(max_val, hmax<float>(slm_max_arr_16));
        //   break;
        // case 8:
        // case 7:
        // case 6:
        // case 5:
        //   for (int i = 4; i < GROUPSIZE; i++) {
        //     slm_max_arr_mask_8[i] = 0;
        //   }
        //   slm_max_arr_8 = slm_block_load<float, 8>(0);
        //   tmp_8.merge(slm_max_arr_8, slm_max_arr_mask_8);
        //   slm_max_arr_8 = tmp_8;
          // if (group_id == 0 && local_id == 0) {
          //   sycl::ext::oneapi::experimental::printf("inside sc\n");
          // }
          // max_val = std::max(max_val, hmax<float>(slm_max_arr_8));
          // break;
        // case 4:
        // case 3:
        // case 2:
        // case 1:
        //   for (int i = 0; i < GROUPSIZE; i++) {
        //     slm_max_arr_mask_4[i] = 1;
        //   }
        //   slm_max_arr_4 = slm_block_load<float, 4>(0);
        //   tmp_4.merge(slm_max_arr_4, slm_max_arr_mask_4);
        //   slm_max_arr_4 = tmp_4;
        //   max_val = std::max(max_val, hmax<float>(slm_max_arr_4));
        //   break;
      // }

      // auto slm_max_arr = slm_block_load<float, 16>(0);
      // max_val = std::max(max_val, hmax<float>(slm_max_arr));

#pragma unroll
      for (int i = 0; i < GROUPSIZE; ++i) {
        max_val = sycl::ext::intel::esimd::max<float>(
            max_val, slm_scalar_load<float>(i * 4));
      }
      idx.barrier();

      src_arr = sycl::ext::intel::esimd::exp<float, VL>(src_arr - max_val);
      slm_scalar_store<float>(local_id * 4, sycl::ext::intel::esimd::detail::sum<float, float, VL>(src_arr));
      idx.barrier();

      auto sum_val = 0.f;
      // auto slm_partial_sum_arr = slm_block_load<float, 16>(0);
      // sum_val = reduce<float>(slm_partial_sum_arr, std::plus<>());
#pragma unroll
      for (int i = 0; i < GROUPSIZE; ++i) {
        sum_val += slm_scalar_load<float>(i * 4);
      }
      idx.barrier();

      src_arr = src_arr / sum_val;
      if (local_id == (GROUPSIZE - 1) && seq_len) {
        simd<uint32_t, 32> res_offsets;
        for(int k = 0; k < 32; ++k) {
          res_offsets[k] = k * 4;
        }
        simd_mask<32> mask_arr;
        auto src_res = src_arr.bit_cast_view<float, 8, 32>();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          simd<float, 32> src_res_row = src_res.row(i);
          mask_arr.copy_from(mask_init + i * 32);
          scatter(src + offset + i * 32, res_offsets, src_res_row, mask_arr);
        }
      } else {
        src_arr.copy_to(src + offset);
      } }); });

  return softmax;
}


void softmax_cpu(float *src, int batchSize, int kvLen) {
// #pragma omp parallel for simd
#pragma omp parallel for
    for (int i = 0; i < batchSize; ++i) {
        auto max_arr = _mm512_set1_ps(std::numeric_limits<float>::lowest());
        for (int j = 0; j < kvLen; j += 16) {
            auto src_arr = _mm512_loadu_ps(src + i * N + j);
            // if (i == 0 && j == 0) {
                // std::cout << "src_cpu:" << src_arr[1] << std::endl;
                // std::cout << "max_arr:" << max_arr[15] << std::endl;
                // std::cout << "src[0]:" << src[0] << std::endl;
            // }
            max_arr = _mm512_max_ps(src_arr, max_arr);
        }
        float max_val_i = _mm512_reduce_max_ps(max_arr);
        max_arr = _mm512_set1_ps(max_val_i);
        // if (i == 311) {
        //   // std::cout << "sum_i_cpu:" << sum_i << std::endl;
        //   // for (int j = 0; j < 16; j++) {
        //   //   std::cout << "max_arr:" << max_arr[j] << std::endl;
        //   // }
        //   std::cout << "max_i_cpu:" << max_i << std::endl;
        // }
        auto sum_arr_i = _mm512_setzero_ps();
        for (int j = 0; j < kvLen; j += 16) {
            auto exp_arr = _mm512_exp_ps(_mm512_sub_ps(_mm512_loadu_ps(src + i * N + j), max_arr));
            sum_arr_i += exp_arr;
        }
        float sum_i = _mm512_reduce_add_ps(sum_arr_i);
        if (i == 311) {
          std::cout << "sum_i_cpu:" << sum_i << std::endl;
        }

        sum_arr_i = _mm512_set1_ps(sum_i);
        for (int j = 0; j < kvLen; j += 16) {
            auto src_arr = _mm512_loadu_ps(src + i * N + j);
            src_arr = _mm512_exp_ps(_mm512_sub_ps(src_arr, max_arr));
            _mm512_store_ps(src + i * N + j, _mm512_div_ps(src_arr, sum_arr_i));
        }
    }
}

void softmax_cpu_mask(float *src, int batchSize, int kvLen) {
// #pragma omp parallel for simd
int remainder = kvLen % 16;
#pragma omp parallel for
    for (int i = 0; i < batchSize; ++i) {
        auto max_arr = _mm512_set1_ps(std::numeric_limits<float>::lowest());
        for (int j = 0; j < kvLen - remainder; j += 16) {
            auto src_arr = _mm512_loadu_ps(src + i * N + j);
            // if (i == 0 && j == 0) {
                // std::cout << "src_cpu:" << src_arr[1] << std::endl;
                // std::cout << "max_arr:" << max_arr[15] << std::endl;
                // std::cout << "src[0]:" << src[0] << std::endl;
            // }
            max_arr = _mm512_max_ps(src_arr, max_arr);
        }
        __mmask16 mask_bit = 0xffff << (16 - remainder);
        if (remainder) {
          auto max_arr_mask = _mm512_set1_ps(std::numeric_limits<float>::lowest());
          auto remainder_arr = _mm512_mask_loadu_ps(max_arr_mask, mask_bit, src + i * N + kvLen - remainder);
          max_arr = _mm512_max_ps(remainder_arr, max_arr);
        }

        float max_val_i = _mm512_reduce_max_ps(max_arr);
        max_arr = _mm512_set1_ps(max_val_i);
        // if (i == 311) {
        //   // std::cout << "sum_i_cpu:" << sum_i << std::endl;
        //   // for (int j = 0; j < 16; j++) {
        //   //   std::cout << "max_arr:" << max_arr[j] << std::endl;
        //   // }
        //   std::cout << "max_i_cpu:" << max_i << std::endl;
        // }
        auto sum_arr_i = _mm512_setzero_ps();
        for (int j = 0; j < kvLen - remainder; j += 16) {
            auto exp_arr = _mm512_exp_ps(_mm512_sub_ps(_mm512_loadu_ps(src + i * N + j), max_arr));
            sum_arr_i += exp_arr;
        }
        float sum_i = _mm512_reduce_add_ps(sum_arr_i);
        if (remainder) {
          auto remainder_arr = _mm512_mask_loadu_ps(max_arr, mask_bit, src + i * N + kvLen - remainder);
          auto exp_remainder_arr = _mm512_exp_ps(_mm512_sub_ps(remainder_arr, max_arr));
          sum_i += (_mm512_reduce_add_ps(exp_remainder_arr) - (16 - remainder));
        }

        if (i == 311) {
          std::cout << "sum_i_cpu:" << sum_i << std::endl;
        }

        sum_arr_i = _mm512_set1_ps(sum_i);
        for (int j = 0; j < kvLen - remainder; j += 16) {
          auto src_arr = _mm512_loadu_ps(src + i * N + j);
          src_arr = _mm512_exp_ps(_mm512_sub_ps(src_arr, max_arr));
          _mm512_storeu_ps(src + i * N + j, _mm512_div_ps(src_arr, sum_arr_i));
        }
        if (remainder) {
          // auto max_arr_mask = _mm512_set1_ps(std::numeric_limits<float>::lowest());
          auto remainder_arr = _mm512_maskz_loadu_ps(mask_bit, src + i * N + kvLen - remainder);
          remainder_arr = _mm512_exp_ps(_mm512_sub_ps(remainder_arr, max_arr));
          _mm512_mask_storeu_ps(src + i * N + kvLen - remainder, mask_bit, _mm512_div_ps(remainder_arr, sum_arr_i));
        }
    }
}

auto softmax_half(queue &main_queue, half *src, int batchSize,
                  int kvLen) {
  unsigned batch_collapse = batchSize;
  constexpr unsigned VL = 16;
  unsigned GROUPSIZE = kvLen / VL;
  // unsigned remainder = kvLen % VL;
  // if (remainder != 0) {
  //   GROUPSIZE += 1;
  // }

  nd_range<1> NDR{range<1>{batch_collapse * GROUPSIZE}, range<1>{GROUPSIZE}};

  event softmax = main_queue.submit([&](handler &h) {
    h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      slm_init<256>();
      int local_id = idx.get_local_linear_id();
      int group_id = idx.get_group_linear_id();
    //   sycl::ext::oneapi::experimental::printf("local_id:%d\n", local_id);

      auto offset = group_id * kvLen + local_id * VL;
      simd<float, VL> max_arr = std::numeric_limits<float>::lowest();
      
      simd<half, VL> src_arr_half(src + offset);
      simd<float, VL> src_arr = convert<float, half, VL>(src_arr_half);
    //   if (remainder != 0 && local_id == GROUPSIZE - 1) {
    //     simd<unsigned short, VL> mask = 0;
    //     for (int i = 0; i < remainder; i++) {
    //         mask[i] = 1;
    //     }
    //     src_arr.merge(src_arr, max_arr, mask);
    //   }
      max_arr = sycl::ext::intel::esimd::max<float, VL>(src_arr, max_arr);
      slm_scalar_store<float>(local_id * 4, hmax<float>(max_arr));

      auto max_val = std::numeric_limits<float>::lowest();
      auto slm_max_arr = slm_block_load<float, 16>(0);
      max_val = std::max(max_val, hmax<float>(slm_max_arr));
// #pragma unroll
//       for (int i = 0; i < GROUPSIZE; ++i) {
//         max_val = sycl::ext::intel::esimd::max<float>(
//             max_val, slm_scalar_load<float>(i * 4));
//       }

      src_arr = sycl::ext::intel::esimd::exp<float, VL>(src_arr - max_val);
      slm_scalar_store<float>(local_id * 4,
                              reduce<float>(src_arr, std::plus<>()));

      auto sum_val = 0.f;
      auto slm_partial_sum_arr = slm_block_load<float, 16>(0);
      sum_val = reduce<float>(slm_partial_sum_arr, std::plus<>());
// #pragma unroll
//       for (int i = 0; i < GROUPSIZE; ++i) {
//         sum_val += slm_scalar_load<float>(i * 4);
//       }

      src_arr = src_arr / sum_val;
      simd<half, VL> dst_arr = convert<half, float, VL>(src_arr);
      dst_arr.copy_to(src + offset);
    });
  });

  return softmax;
}

auto softmax_float(queue &main_queue, float *src, int batchSize,
                  int kvLen) {
  unsigned batch_collapse = batchSize;
  constexpr unsigned VL = 256;
  unsigned GROUPSIZE = kvLen / VL;
  // unsigned remainder = kvLen % VL;
  // if (remainder != 0) {
  //   GROUPSIZE += 1;
  // }

  nd_range<1> NDR{range<1>{batch_collapse * GROUPSIZE}, range<1>{GROUPSIZE}};

  event softmax = main_queue.submit([&](handler &h) {
    h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      // slm_init<256>();
      slm_init<1024>();
      int local_id = idx.get_local_linear_id();
      int group_id = idx.get_group_linear_id();
    //   sycl::ext::oneapi::experimental::printf("local_id:%d\n", local_id);

      auto offset = group_id * kvLen + local_id * VL;
      simd<float, VL> max_arr = std::numeric_limits<float>::lowest();
      
      simd<float, VL> src_arr(src + offset);
    //   if (remainder != 0 && local_id == GROUPSIZE - 1) {
    //     simd<unsigned short, VL> mask = 0;
    //     for (int i = 0; i < remainder; i++) {
    //         mask[i] = 1;
    //     }
    //     src_arr.merge(src_arr, max_arr, mask);
    //   }
      max_arr = sycl::ext::intel::esimd::max<float, VL>(src_arr, max_arr);
      slm_scalar_store<float>(local_id * 4, hmax<float>(max_arr));
      idx.barrier();

      auto max_val = std::numeric_limits<float>::lowest();
      auto slm_max_arr = slm_block_load<float, 16>(0);
      // idx.barrier();
      max_val = std::max(max_val, hmax<float>(slm_max_arr));
// #pragma unroll
//       for (int i = 0; i < GROUPSIZE; ++i) {
//         max_val = sycl::ext::intel::esimd::max<float>(
//             max_val, slm_scalar_load<float>(i * 4));
//       }

      src_arr = sycl::ext::intel::esimd::exp<float, VL>(src_arr - max_val);
      slm_scalar_store<float>(local_id * 4,
                              reduce<float>(src_arr, std::plus<>()));

      idx.barrier();
      auto sum_val = 0.f;
      auto slm_partial_sum_arr = slm_block_load<float, 16>(0);
      // idx.barrier();
      sum_val = reduce<float>(slm_partial_sum_arr, std::plus<>());
// #pragma unroll
//       for (int i = 0; i < GROUPSIZE; ++i) {
//         sum_val += slm_scalar_load<float>(i * 4);
//       }
      // idx.barrier();
      // if (group_id == 311 && local_id == 0) {
      //   sycl::ext::oneapi::experimental::printf("sum_val:%f\n", sum_val);
      // }

      src_arr = src_arr / sum_val;
      src_arr.copy_to(src + offset);
    });
  });

  return softmax;
}

auto softmax_float_mask(queue &main_queue, float *src, int batchSize,
                  int kvLen, unsigned short *mask_init, unsigned remainder) {
  unsigned batch_collapse = batchSize;
  constexpr unsigned VL = 32;
  unsigned GROUPSIZE = kvLen / VL;
  // unsigned remainder = kvLen % VL;
  if (remainder != 0) {
    GROUPSIZE += 1;
  }

  nd_range<1> NDR{range<1>{batch_collapse * GROUPSIZE}, range<1>{GROUPSIZE}};

  event softmax = main_queue.submit([&](handler &h) {
    h.parallel_for(NDR, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      slm_init<256>();
      // slm_init<1024>();
      // slm_init<1024>();
      int local_id = idx.get_local_linear_id();
      int group_id = idx.get_group_linear_id();
      // sycl::ext::oneapi::experimental::printf("local_id:%d\n", local_id);

      auto offset = group_id * kvLen + local_id * VL;
      simd<float, VL> max_arr = std::numeric_limits<float>::lowest();
      
      simd<float, VL> src_arr(src + offset);
      simd<unsigned short, VL> mask(mask_init);
      if (remainder != 0 && local_id == GROUPSIZE - 1) {
        // for (int i = 0; i < remainder; i++) {
        //     mask[i] = 1;
        // }
        src_arr.merge(src_arr, max_arr, mask);
      }
      max_arr = sycl::ext::intel::esimd::max<float, VL>(src_arr, max_arr);
      slm_scalar_store<float>(local_id * 4, hmax<float>(max_arr));
      idx.barrier();

      auto max_val = std::numeric_limits<float>::lowest();
      auto slm_max_arr = slm_block_load<float, 16>(0);
      float remainder_max = std::numeric_limits<float>::lowest();
      if (remainder != 0 && local_id == GROUPSIZE - 1) {
        remainder_max = slm_scalar_load<float>(local_id * 4);
      }
      // idx.barrier();
      max_val = std::max(max_val, hmax<float>(slm_max_arr));
      if (remainder != 0) {
        max_val = std::max(max_val, remainder_max);
      }
// #pragma unroll
//       for (int i = 0; i < GROUPSIZE; ++i) {
//         max_val = sycl::ext::intel::esimd::max<float>(
//             max_val, slm_scalar_load<float>(i * 4));
//       }

      src_arr = sycl::ext::intel::esimd::exp<float, VL>(src_arr - max_val);
      if (remainder != 0) {
        simd<float, VL> zero = 0.0f;
        src_arr.merge(src_arr, zero, mask);
      }
      slm_scalar_store<float>(local_id * 4,
                              reduce<float>(src_arr, std::plus<>()));

      idx.barrier();
      auto sum_val = 0.f;
      auto slm_partial_sum_arr = slm_block_load<float, 16>(0);
      float remainder_sum = 0.0f;
      if (remainder != 0 && local_id == GROUPSIZE - 1) {
        remainder_sum = slm_scalar_load<float>(local_id * 4);
      }
      // idx.barrier();
      sum_val = reduce<float>(slm_partial_sum_arr, std::plus<>());
      if (remainder != 0) {
        sum_val = sum_val + remainder_sum;
      }
// #pragma unroll
//       for (int i = 0; i < GROUPSIZE; ++i) {
//         sum_val += slm_scalar_load<float>(i * 4);
//       }
      // idx.barrier();
      // if (group_id == 311 && local_id == 0) {
      //   sycl::ext::oneapi::experimental::printf("sum_val:%f\n", sum_val);
      // }

      src_arr = src_arr / sum_val;
      if (remainder != 0 && local_id == GROUPSIZE - 1) {
        simd_mask<VL> pred(mask_init);
        simd<unsigned, VL> offsets;
        for (int i = 0; i < VL; i++) {
          offsets[i] = i * 4;
        }
        sycl::ext::intel::experimental::esimd::lsc_scatter<float>(src + offset, offsets, src_arr, pred);
        // scatter<float>(src + offset, offsets, src_arr, pred);
      } else {
        src_arr.copy_to(src + offset);
      }
    });
  });

  return softmax;
}

auto softmax_float_1d(queue &main_queue, float *src, int batchSize,
                  int kvLen) {
  unsigned batch_collapse = batchSize;
  constexpr unsigned VL = 16;
  // unsigned GROUPSIZE = kvLen / VL;
  // unsigned remainder = kvLen % VL;
  // if (remainder != 0) {
  //   GROUPSIZE += 1;
  // }

  range<1> size{batch_collapse};

  event softmax = main_queue.submit([&](handler &h) {
    h.parallel_for(size, [=](id<1> idx) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      int global_id = idx[0];

      // auto offset = global_id * kvLen;
      simd<float, VL> max_arr = std::numeric_limits<float>::lowest();
      
      for (int i = 0; i < kvLen; i += VL) {
        simd<float, VL> src_arr(src + global_id * kvLen + i);
        // if (global_id == 311 && i == 0) {
        //   sycl::ext::oneapi::experimental::printf("src_arr:%f\n", src_arr[0]);
        // }
        max_arr = sycl::ext::intel::esimd::max<float, VL>(src_arr, max_arr);
      }
      auto max_val_i = hmax<float>(max_arr);
      max_arr = max_val_i;
      // if (global_id == 311) {
      //   sycl::ext::oneapi::experimental::printf("max_val:%f\n", max_val_i);
      //   // // for (int i = 0; i < VL; i++) {
      //   // sycl::ext::oneapi::experimental::printf("max_arr:%f\n", max_arr[1]);
      //   // // }
      // }

      simd<float, VL> sum_arr = 0.0;
      for (int i = 0; i < kvLen; i += VL) {
        simd<float, VL> src_arr(src + global_id * kvLen + i);
        src_arr = sycl::ext::intel::esimd::exp<float, VL>(src_arr - max_arr);
        sum_arr += src_arr;
      }
      float sum_val = reduce<float>(sum_arr, std::plus<>());
      // if (global_id == 311) {
      //   sycl::ext::oneapi::experimental::printf("sum_val:%f\n", sum_val);
      // }
      
      for (int i = 0; i < kvLen; i += VL) {
        simd<float, VL> src_arr(src + global_id * kvLen + i);
        src_arr = sycl::ext::intel::esimd::exp<float, VL>(src_arr - max_arr) / sum_val;
        src_arr.copy_to(src + global_id * kvLen + i);
      }
    });
  });

  return softmax;
}


int main() {
    int warmup_time = 5;
    constexpr size_t VL = 16;
    property_list properties{property::queue::enable_profiling()};
    queue q(properties);
    auto context = q.get_info<info::queue::context>();
    auto device = q.get_info<info::queue::device>();

    constexpr size_t size_softmax = M * N;
    auto src = alloc_host_and_init<float>(
        size_softmax,
        [](float* data, size_t idx) {
            data[idx] = generate_random<float>();
        });
    auto src_gpu = static_cast<float *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_softmax * sizeof(float), device, context));
    q.memcpy((void *)src_gpu, (void *)src, size_softmax * sizeof(float)).wait();
    
    auto src_gpu_res_on_cpu = alloc_host_and_init<float>(
        size_softmax,
        [](float* data, size_t idx) {
            data[idx] = 0.0f;
        });

    auto mask = alloc_host_and_init<unsigned short>(
        256,
        [](unsigned short* data, size_t idx) {
            data[idx] = 0;
        });
    auto mask_gpu = static_cast<unsigned short *>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, 256 * sizeof(unsigned short), device, context));
    unsigned remainder = N % 256;
    for (int i = 0; i < remainder; i++) {
      mask[i] = 1;
    }
    q.memcpy((void *)mask_gpu, (void *)mask, 256 * sizeof(unsigned short)).wait();

    for (size_t it = 0; it < warmup_time; it++) {
        auto t1 = std::chrono::steady_clock::now();   // Start timing
        softmax_cpu(src, M, N);
        auto t2 = std::chrono::steady_clock::now();   // Stop timing
        auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "warmup iter:" << it << " cpu kernel time: " << exec_time << "us " << std::endl;
    }


    std::cout << "*****vectorization warmup stage*****" << std::endl;
    for (int it = 0; it < warmup_time; it++) {
        auto t1 = std::chrono::steady_clock::now();   // Start timing
        // auto event = softmax_half(q, src_gpu, M, N);
        // auto event = softmax_float(q, src_gpu, M, N);
        // auto event = softmax_float_mask(q, src_gpu, M, N, mask_gpu, remainder);
        auto event = softmax_float_mask_lw(q, src_gpu, M, N, mask_gpu, remainder);
        // auto event = softmax_float_1d(q, src_gpu, M, N);
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

    q.memcpy((void *)src_gpu_res_on_cpu, (void *)src_gpu, size_softmax * sizeof(float)).wait();

    float total_var = 0.0f;
    // validate result
    for (size_t i = 0; i < M; i++) {
        float v1 = src[i];
        float v2 = src_gpu_res_on_cpu[i];
        auto delt = v1 - v2;
        total_var += delt * delt;
    }
    std::cout << "total error:" << total_var << std::endl;

    free(src_gpu, q);
    free(src);
    free(src_gpu_res_on_cpu);

    return 0;
}
