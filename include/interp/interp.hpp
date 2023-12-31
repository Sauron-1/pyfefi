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

template<typename iSimd>
static INLINE iSimd limit_index(iSimd idx, size_t idx_max) {
    idx = select(idx < 0, 0, idx);
    idx = select(idx >= idx_max, idx_max-1, idx);
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

template<size_t interp_order, typename Simd, typename Float>
static INLINE auto interp_one(
        std::array<Simd, 3> target,
        std::array<Float, 3> scale,
        std::array<Float, 3> lo,
        std::array<Float, 3> hi,
        const NdArray<Float, 4> var,
        std::vector<Simd>& results) {
    using iSimd = typename isimd_type<Simd>::type;
    std::array<std::array<iSimd, interp_order+1>, 3> indices;
    std::array<std::array<Simd, interp_order+1>, 3> weights;
    std::array<size_t, 3> idx_max;
    decltype(target[0] > 0) msk = true;

    for (auto i = 0; i < 3; ++i) {
        idx_max[i] = (hi[i] - lo[i]) / scale[i];
        if (idx_max[i] < 1)
            idx_max[i] = 1;
    }

    for (auto i = 0; i < 3; ++i) {
        msk &= target[i] > lo[i] - scale[i]*0.5;
        msk &= target[i] < hi[i] + scale[i]*0.5;
    }

    for (auto i = 0; i < 3; ++i) {
        if constexpr (interp_order == 1) {
            auto [idx, wt] = to_idx_weights(target[i], scale[i], lo[i], idx_max[i]);
            indices[i] = idx;
            weights[i] = wt;
        }
        else if constexpr (interp_order == 2) {
            auto [idx, wt] = to_idx_weights_2(target[i], scale[i], lo[i], idx_max[i]);
            indices[i] = idx;
            weights[i] = wt;
        }
    }

    const size_t num = var.shape(3);

    for (auto i = 0; i < num; ++i) {
        results[i] = 0.0;
        for (auto ix = 0; ix < interp_order+1; ++ix) {
            for (auto iy = 0; iy < interp_order+1; ++iy) {
                for (auto iz = 0; iz < interp_order+1; ++iz) {
                    auto val = var.gather(
                            indices[0][ix],
                            indices[1][iy],
                            indices[2][iz], iSimd(i));
                    results[i] += val * \
                                  weights[0][ix] * \
                                  weights[1][iy] * \
                                  weights[2][iz];
                }
            }
        }
    }
    for (auto i = 0; i < num; ++i) {
        results[i] = select(
                msk, results[i],
                std::numeric_limits<Float>::quiet_NaN());
    }
}
