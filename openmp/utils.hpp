#pragma once

#include <iostream>
#include <random>
#include <omp.h>
#include <immintrin.h>
#include <chrono>

#define DEVICE_MEM_ALIGNMENT (64)

template <typename rtype>
inline rtype generate_random(rtype a = 0.0, rtype b = 1.0) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
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