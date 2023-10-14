#include <cmath>
#include <cstdint>
#include <array>
#include <iostream>
#include <string>

#pragma once

template<size_t N, typename T>
constexpr auto powi(T&& val) {
    if constexpr (N == 0) {
        return T(1);
    }
    else if constexpr (N == 1) {
        return val;
    }
    else {
        auto half = powi<N/2>(val);
        if constexpr (N % 2 == 0) {
            return half * half;
        }
        else {
            return half * half * val;
        }
    }
}

inline bool my_isnan(float val) {
    union { float f; uint32_t x; } u = {val};
    return (u.x << 1) > (0x7f800000u << 1);
}

inline bool my_isnan(double val) {
    union { double f; uint64_t x; } u = {val};
    return (u.x << 1) > (0x7ff0000000000000u << 1);
}

template<typename T, size_t N>
inline void check_bounds(const std::array<T, N>& idx, const std::array<size_t, N>& shape, const std::string& msg) {
    for (size_t i = 0; i < N; ++i) {
        if (idx[i] >= shape[i] || idx[i] < 0) [[unlikely]] {
            std::cerr << msg << std::endl <<
                "Index " << idx[i] << " out of bounds for dimension " << i <<
                " (shape = " << shape[i] << "). Explicitly check bound." << std::endl;
            exit(1);
        }
    }
}
