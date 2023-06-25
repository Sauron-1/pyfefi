#include <iostream>
#include <array>
#include <type_traits>
#include <cstdint>
#include <limits>
#include <vectorclass/vectorclass.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ndarray.hpp>
#include <simd.hpp>

#pragma once

template<typename iSimd>
static INLINE iSimd limit_index(iSimd idx, size_t idx_max) {
    idx = select(idx < 0, 0, idx);
    idx = select(idx >= idx_max, idx_max - 1, idx);
    return idx;
}

template<typename Simd, typename Float>
static INLINE auto to_idx_weights(Simd target, Float scale, Float lo, size_t idx_max) {
    auto fidx = (target - lo) / scale;
    auto idx = truncatei(fidx);
    auto delta = fidx - convert<Float>(idx);
    std::array<decltype(idx), 2> indices { limit_index(idx, idx_max), limit_index(idx+1, idx_max) };
    std::array<Simd, 2> weights { 1.0 - delta, delta };
    return make_pair(indices, weights);
}

template<typename Simd, typename Float>
static INLINE auto to_idx_weights_2(Simd target, Float scale, Float lo, size_t idx_max) {
    auto fidx = (target - lo) / scale;
    auto idx = roundi(fidx);
    auto delta = fidx - convert<Float>(idx);
    std::array<decltype(idx), 3> indices {
        limit_index(idx-1, idx_max),
        limit_index(idx, idx_max),
        limit_index(idx+1, idx_max)
    };
    std::array<Simd, 3> weights {
        0.5 * (0.5 - delta) * (0.5 - delta),
        0.75 - delta * delta,
        0.5 * (0.5 + delta) * (0.5 + delta)
    };
    return make_pair(indices, weights);
}
