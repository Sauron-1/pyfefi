#include <interp/utils.hpp>

template<size_t interp_order, typename Simd, typename Float>
static INLINE auto interp_one_bwd(
        std::array<Simd, 3> source,
        std::array<Float, 3> scale,
        std::array<Float, 3> lo,
        std::array<Float, 3> hi,
        const NdArray<Float, 3> var,
        Simd data) {
    using iSimd = typename isimd_type<Simd>::type;
    std::array<std::array<iSimd, interp_order+1>, 3> indices;
    std::array<std::array<Simd, interp_order+1>, 3> weights;
    std::array<size_t, 3> idx_max;
    decltype(source[0] > 0) msk = true;

    for (auto i = 0; i < 3; ++i) {
        idx_max[i] = (hi[i] - lo[i]) / scale[i];
        if (idx_max[i] < 1)
            idx_max[i] = 1;
    }

    for (auto i = 0; i < 3; ++i) {
        msk &= source[i] > lo[i] - scale[i]*0.5;
        msk &= source[i] < hi[i] + scale[i]*0.5;
    }

    for (auto i = 0; i < 3; ++i) {
        if constexpr (interp_order == 1) {
            auto [idx, wt] = to_idx_weights(source[i], scale[i], lo[i], idx_max[i]);
            indices[i] = idx;
            weights[i] = wt;
        }
        else if constexpr (interp_order == 2) {
            auto [idx, wt] = to_idx_weights_2(source[i], scale[i], lo[i], idx_max[i]);
            indices[i] = idx;
            weights[i] = wt;
        }
    }

    Simd tmp;
    for (auto ix = 0; ix < interp_order+1; ++ix) {
        for (auto iy = 0; iy < interp_order+1; ++iy) {
            for (auto iz = 0; iz < interp_order+1; ++iz) {
                tmp = data * \
                      weights[0][ix] * \
                      weights[1][iy] * \
                      weights[2][iz];
                var.template scatterm<1>(
                        tmp, msk,
                        indices[0][ix],
                        indices[1][iy],
                        indices[2][iz] );
            }
        }
    }
}
