#include <array>
#include <type_traits>
#include <vectorclass/vectorclass.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ndarray.hpp>
#include <simd.hpp>

template<typename Simd, typename iSimd, typename...Arr>
static INLINE auto interp_one_2d_2(
        std::array<Simd, 2> pos,
        std::array<iSimd, 2> idx,
        const Arr&...arr) {
    std::array<std::array<iSimd, 2>, 2> indices{
        std::array{ idx[0], idx[0]+1 },
        std::array{ idx[1], idx[1]+1 }
    };
    std::array<std::array<Simd, 2>, 2> weights{
        std::array{ 1-pos[0], pos[0] },
        std::array{ 1-pos[1], pos[1] }
    };

    auto arrs = std::tie(arr...);

    std::array<Simd, sizeof...(arr)> results;
    for (auto ir = 0; ir < sizeof...(arr); ++ir) {
        results[ir] = 0.0;
    }
    for (auto ix = 0; ix < 2; ++ix) {
        for (auto iy = 0; iy < 2; ++iy) {
            auto w = weights[0][ix] * weights[1][iy];
            constexpr_for<0, sizeof...(arr), 1>( [&](auto I) {
                constexpr auto i = decltype(I)::value;
                auto val = std::get<i>(arrs).gather(
                        indices[0][ix], indices[1][iy]);
                results[i] += val * w;
            });
        }
    }

    return results;
}

template<typename Float>
class LicPack {

    public:
        using Simd = native_simd_t<Float>;
        using iSimd = typename isimd_type<Simd>::type;
        using bSimd = typename bsimd_type<Simd>::type;
        using ibSimd = typename bsimd_type<iSimd>::type;

        INLINE LicPack(
                std::array<iSimd, 2> _idx,
                bSimd _msk,
                const NdArray<Float, 2>& _u,
                const NdArray<Float, 2>& _v,
                const NdArray<Float, 2>& _texture,
                const NdArray<Float, 1>& _kernel) :
            u(_u), v(_v), texture(_texture), kernel(_kernel), idx(_idx), msk(_msk) {
                pos[0] = pos[1] = 0.0;
                result = 0.0;
                shape = u.shape();
                limit_idx();
            }

        template<int sign, bool acc_only>
        INLINE void advance(int kidx) {
            Simd vx, vy;
            if constexpr (not acc_only) {
                auto [ui, vi, ti] = interp_one_2d_2(pos, idx, u, v, texture);
                result = if_add(msk, result, ti * kernel(kidx));
                vx = ui * sign;
                vy = vi * sign;
            }
            else {
                auto [ui, vi] = interp_one_2d_2(pos, idx, u, v);
                vx = ui * sign;
                vy = vi * sign;
            }
            Simd bx = select(vx > 0, Simd(1), Simd(0)),
                 by = select(vy > 0, Simd(1), Simd(0));
            auto mskx = abs(vx) > 1e-20,
                 msky = abs(vy) > 1e-20;
            msk &= mskx | msky;
            auto tx = select(mskx, (bx - pos[0]) / vx, 1e21),
                 ty = select(msky, (by - pos[1]) / vy, 1e21);

            auto if_use_tx = tx < ty;
            auto t = select(tx < ty, tx, ty);

            idx[0] = if_add( ibSimd(if_use_tx), idx[0], select(ibSimd(vx > 0), iSimd(1), iSimd(-1)));
            idx[1] = if_add(!ibSimd(if_use_tx), idx[1], select(ibSimd(vy > 0), iSimd(1), iSimd(-1)));

            pos[0] = select( if_use_tx, 1.0 - bx, t * vx + pos[0]);
            pos[1] = select(!if_use_tx, 1.0 - by, t * vy + pos[1]);

            limit_idx();
        }

        INLINE auto limit_idx() {
            for (auto i = 0; i < 2; ++i) {
                msk &= (idx[i] >= 0) & (idx[i] < shape[i] - 1);
                idx[i] = select(ibSimd(msk), idx[i], 0);
            }
        }

        INLINE Simd run() {
            int nk = kernel.shape(0);
            int kmid = nk / 2;

            bSimd msk_ini = msk;
            std::array<iSimd, 2> idx_ini = idx;
            std::array<Simd, 2> pos_ini = pos;
            for (int kidx = kmid; kidx < nk; ++kidx) {
                advance<1, false>(kidx);
            }

            msk = msk_ini;
            idx = idx_ini;
            pos = pos_ini;
            advance<-1, true>(kmid);
            for (int kidx = kmid-1; kidx >= 0; --kidx) {
                advance<-1, false>(kidx);
            }

            return result;
        }

    private:
        const NdArray<Float, 2> &u, &v, &texture;
        const NdArray<Float, 1> &kernel;
        std::array<iSimd, 2> idx;
        std::array<Simd, 2> pos;
        std::array<size_t, 2> shape;
        bSimd msk;
        Simd result;
};

template<int simd_len_init = -1, typename Simd>
void print_simd(Simd val) {
    constexpr size_t simd_len = simd_len_init > 0 ? simd_len_init : simd_length<Simd>;
    std::cout << "(";
    for (auto i = 0; i < simd_len - 1; ++i) {
        std::cout << val[i] << ", ";
    }
    std::cout << val[simd_len-1] << ")";
}

template<typename Float>
static auto lic_kernel(
        const NdArray<Float, 2>& u,
        const NdArray<Float, 2>& v,
        const NdArray<Float, 2>& texture,
        const NdArray<Float, 1>& kernel,
        NdArray<Float, 2>& results) {
    using Simd = native_simd_t<Float>;
    using iSimd = typename isimd_type<Simd>::type;

    NdIndices nd(results.shape());
    while (nd.has_next()) {
        auto [msk, idx] = nd.template next_batch<iSimd>();
        auto result = LicPack(idx, msk, u, v, texture, kernel).run();
        results.scatter(result, idx[0], idx[1]);
    }
}

template<typename Float>
static auto lic_kernel_para(
        const NdArray<Float, 2>& u,
        const NdArray<Float, 2>& v,
        const NdArray<Float, 2>& texture,
        const NdArray<Float, 1>& kernel,
        NdArray<Float, 2>& results) {
    using Simd = native_simd_t<Float>;
    using iSimd = typename isimd_type<Simd>::type;
    using bSimd = typename bsimd_type<Simd>::type;
    using Int = decltype(std::declval<iSimd>()[0]);
    constexpr size_t simd_len = simd_length<Simd>;
    NdIndices nd(results.shape());
    size_t num_packs = nd.size() / simd_len;
#pragma omp parallel for
    for (auto i = 0; i < num_packs; ++i) {
        std::array<Int, simd_len> buf;
        bSimd msk = true;
        iSimd idx;
        for (auto j = 0; j < simd_len; ++j) {
            buf[j] = i * simd_len + j;
        }
        idx.load(buf.data());
        auto indices = nd.i2idx(idx);
        auto result = LicPack(indices, msk, u, v, texture, kernel).run();
        results.scatter(result, indices[0], indices[1]);
    }

    size_t extra = nd.size() % simd_len;
    size_t extra_start = nd.size() - extra;
    if (extra == 0) return;

    std::array<Int, simd_len> buf;
    for (auto i = 0; i < extra; ++i) buf[i] = 1;
    for (auto i = extra; i < simd_len; ++i) buf[i] = 0;
    iSimd msk_tmp; msk_tmp.load(buf.data());
    auto msk = msk_tmp == 1;
    for (auto i = 0; i < extra; ++i) buf[i] = extra_start + i;
    for (auto i = extra; i < simd_len; ++i) buf[i] = nd.size() - 1;
    iSimd idx; idx.load(buf.data());
    auto indices = nd.i2idx(idx);
    auto result = LicPack(indices, msk, u, v, texture, kernel).run();
    results.scatterm(result, msk, indices[0], indices[1]);
}

template<typename Float>
py::array lic(
        py::array_t<Float> u,
        py::array_t<Float> v,
        py::array_t<Float> texture,
        py::array_t<Float> kernel) {
    if (u.ndim() != 2 || v.ndim() != 2 || texture.ndim() != 2 || kernel.ndim() != 1) {
        throw std::runtime_error("Wrong dimensions!");
    }
    std::array<size_t, 2> shape;
    for (auto i = 0; i < 2; ++i) shape[i] = u.shape(i);
    auto result = py::array_t<Float>(shape);
    NdArray<Float, 2> result_arr(result);

    //lic_kernel(
    lic_kernel_para(
            NdArray<Float, 2>(v),
            NdArray<Float, 2>(u),
            NdArray<Float, 2>(texture),
            NdArray<Float, 1>(kernel),
            result_arr);

    return result;
}

PYBIND11_MODULE(lic, m) {
    m.doc() = "Line integral convolution";
    m.def("lic", &lic<float>, "LIC");
    m.def("lic", &lic<double>, "LIC");
}
