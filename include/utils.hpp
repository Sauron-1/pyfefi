#include <cmath>

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
