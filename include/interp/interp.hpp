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

template<typename Simd, typename bSimd, typename Float>
static INLINE auto to_indices_weights(Simd target, bSimd msk, Float scale, Float lo) {
    target = select(msk, target, lo);
    auto fidx = (target - lo) / scale;
    auto idx = truncatei(fidx);
    auto delta = fidx - convert<Float>(idx);
    std::array<decltype(idx), 2> indices { idx, idx + 1 };
    std::array<Simd, 2> weights { 1.0 - delta, delta };
    return make_pair(indices, weights);
}

template<typename Simd, typename bSimd, typename Float>
static INLINE auto to_indices_weights_2(Simd target, bSimd msk, Float scale, Float lo) {
    target = select(msk, target, lo+scale);
    auto fidx = (target - lo) / scale;
    auto idx = roundi(fidx);
    auto delta = fidx - convert<Float>(idx);
    std::array<decltype(idx), 3> indices { idx - 1, idx, idx + 1 };
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

    decltype(target[0] >= lo[0]) msk;
    if constexpr (interp_order == 1) {
        msk =
            (target[0] >= lo[0]) & (target[0] < hi[0]) &
            (target[1] >= lo[1]) & (target[1] < hi[1]) &
            (target[2] >= lo[2]) & (target[2] < hi[2]);
    }
    else if constexpr (interp_order == 2) {
        msk =
            (target[0] >= lo[0] + scale[0]) & (target[0] < hi[0] - scale[0]) &
            (target[1] >= lo[1] + scale[1]) & (target[1] < hi[1] - scale[1]) &
            (target[2] >= lo[2] + scale[2]) & (target[2] < hi[2] - scale[2]);
    }

    for (auto i = 0; i < 3; ++i) {
        if constexpr (interp_order == 1) {
            auto [idx, wt] = to_indices_weights(target[i], msk, scale[i], lo[i]);
            indices[i] = idx;
            weights[i] = wt;
        }
        else if constexpr (interp_order == 2) {
            auto [idx, wt] = to_indices_weights_2(target[i], msk, scale[i], lo[i]);
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
