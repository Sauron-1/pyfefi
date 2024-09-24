#include <array>
#include <type_traits>
#include <exception>

#include <xsimd/xsimd.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

template<typename Real>
static inline void rotate_kernel(
        const Real Txx,
        const Real Tyy,
        const Real Tzz,
        const Real Txy,
        const Real Tyz,
        const Real Tzx,
        const Real Bx,
        const Real By,
        const Real Bz,
        Real& __restrict Tpara,
        Real& __restrict Tperp) {
    Real x0 = By*By;
    Real x1 = Bz*Bz;
    Real x2 = x0 + x1;
    Real x3 = Real(1)/std::sqrt(Bx*Bx + x2);
    Real x4 = Bx*x3;
    Real x5 = By*x3;
    Real x6 = Txy*x5;
    Real x7 = Bz*x3;
    Real x8 = Tzx*x7;
    Real x9 = (1 - x4)/x2;
    Real x10 = By*Bz*x9;
    Real x11 = x1*x9 + x4;
    Real x12 = x0*x9 + x4;
    Real x13 = x10/2;
    Real x14 = Tyz*x10;
    Tpara = x4*(Txx*x4 + x6 + x8) + x5*(Txy*x4 + Tyy*x5 + Tyz*x7) + x7*(Tyz*x5 + Tzx*x4 + Tzz*x7);
    Tperp = x11*(Tyy*x11 - x14 - x6)/2 + x12*(Tzz*x12 - x14 - x8)/2 - x13*(-Txy*x7 - Tyy*x10 + Tyz*x12) - x13*(Tyz*x11 - Tzx*x5 - Tzz*x10) - x5*(-Txx*x5 + Txy*x11 - Tzx*x10)/2 - x7*(-Txx*x7 - Txy*x10 + Tzx*x12)/2;
}

template<size_t N, typename Real>
static inline void rotate_kernel_v(
        const Real* __restrict _Txx,
        const Real* __restrict _Tyy,
        const Real* __restrict _Tzz,
        const Real* __restrict _Txy,
        const Real* __restrict _Tyz,
        const Real* __restrict _Tzx,
        const Real* __restrict _Bx,
        const Real* __restrict _By,
        const Real* __restrict _Bz,
        Real* __restrict _Tpara,
        Real* __restrict _Tperp) {
    using vec = xsimd::make_sized_batch_t<Real, N>;
    static_assert(not std::is_void_v<vec>);

    auto Txx = vec::load_unaligned(_Txx);
    auto Tyy = vec::load_unaligned(_Tyy);
    auto Tzz = vec::load_unaligned(_Tzz);
    auto Txy = vec::load_unaligned(_Txy);
    auto Tyz = vec::load_unaligned(_Tyz);
    auto Tzx = vec::load_unaligned(_Tzx);
    auto Bx = vec::load_unaligned(_Bx);
    auto By = vec::load_unaligned(_By);
    auto Bz = vec::load_unaligned(_Bz);

    auto x0 = By*By;
    auto x1 = Bz*Bz;
    auto x2 = x0 + x1;
    auto x3 = Real(1)/sqrt(Bx*Bx + x2);
    auto x4 = Bx*x3;
    auto x5 = By*x3;
    auto x6 = Txy*x5;
    auto x7 = Bz*x3;
    auto x8 = Tzx*x7;
    auto x9 = (1 - x4)/x2;
    auto x10 = By*Bz*x9;
    auto x11 = x1*x9 + x4;
    auto x12 = x0*x9 + x4;
    auto x13 = x10/2;
    auto x14 = Tyz*x10;
    auto Tpara = x4*(Txx*x4 + x6 + x8) + x5*(Txy*x4 + Tyy*x5 + Tyz*x7) + x7*(Tyz*x5 + Tzx*x4 + Tzz*x7);
    auto Tperp = x11*(Tyy*x11 - x14 - x6)/2 + x12*(Tzz*x12 - x14 - x8)/2 - x13*(-Txy*x7 - Tyy*x10 + Tyz*x12) - x13*(Tyz*x11 - Tzx*x5 - Tzz*x10) - x5*(-Txx*x5 + Txy*x11 - Tzx*x10)/2 - x7*(-Txx*x7 - Txy*x10 + Tzx*x12)/2;

    Tpara.store_unaligned(_Tpara);
    Tperp.store_unaligned(_Tperp);
}

template<size_t pack_size, typename Real>
static inline void _rotate(
        const Real* __restrict Txx,
        const Real* __restrict Tyy,
        const Real* __restrict Tzz,
        const Real* __restrict Txy,
        const Real* __restrict Tyz,
        const Real* __restrict Tzx,
        const Real* __restrict vx,
        const Real* __restrict vy,
        const Real* __restrict vz,
        Real* __restrict Tpara,
        Real* __restrict Tperp,
        uint64_t size) {
    const uint64_t num_packs = size / pack_size,
                   remains = size % pack_size;
    const uint64_t packed_size = num_packs * pack_size;
    using vec = xsimd::make_sized_batch_t<Real, pack_size>;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_packs; ++i) {
        rotate_kernel_v<pack_size>(
                &Txx[i*pack_size],
                &Tyy[i*pack_size],
                &Tzz[i*pack_size],
                &Txy[i*pack_size],
                &Tyz[i*pack_size],
                &Tzx[i*pack_size],
                &vx[i*pack_size],
                &vy[i*pack_size],
                &vz[i*pack_size],
                &Tpara[i*pack_size],
                &Tperp[i*pack_size]
            );
    }
    for (uint64_t i = 0; i < remains; ++i) {
        rotate_kernel(
                Txx[i+packed_size],
                Tyy[i+packed_size],
                Tzz[i+packed_size],
                Txy[i+packed_size],
                Tyz[i+packed_size],
                Tzx[i+packed_size],
                vx[i+packed_size],
                vy[i+packed_size],
                vz[i+packed_size],
                Tpara[i+packed_size],
                Tperp[i+packed_size]
            );
    }
}

template<typename T0, typename...T>
void assert_same_shape_strides(const py::array_t<T0>& arr0, const py::array_t<T>&...arrs) {
    bool same_dim = (... and (arr0.ndim() == arrs.ndim()));
    if (not same_dim)
        throw std::runtime_error("Array dimensions must be equal");
    for (int i = 0; i < arr0.ndim(); ++i) {
        bool same_shape = (... and (arr0.shape(i) == arrs.shape(i)));
        if (not same_shape)
            throw std::runtime_error("Array shapes must be the same");
    }
    for (int i = 0; i < arr0.ndim(); ++i) {
        bool same_strides = (... and (arr0.strides(i) == arrs.strides(i)));
        if (not same_strides)
            throw std::runtime_error("Array strides must be the same");
    }
}

template<typename Real>
std::vector<py::array_t<Real>> rotate_temp(
        const py::array_t<Real>& Txx,
        const py::array_t<Real>& Tyy,
        const py::array_t<Real>& Tzz,
        const py::array_t<Real>& Txy,
        const py::array_t<Real>& Tyz,
        const py::array_t<Real>& Tzx,
        const py::array_t<Real>& vx,
        const py::array_t<Real>& vy,
        const py::array_t<Real>& vz) {
    constexpr size_t simd_size = xsimd::simd_type<Real>::size;
    std::vector<py::array_t<Real>> result;
    assert_same_shape_strides(Txx, Tyy, Tzz, Txy, Tzx, Tyz, vx, vy, vz);
    std::vector<size_t> shape, strides;
    for (auto i = 0; i < Txx.ndim(); ++i) {
        shape.push_back(Txx.shape(i));
        strides.push_back(Txx.strides(i));
    }
    for (auto i = 0; i < 2; ++i)
        result.emplace_back(shape, strides);
    size_t size = Txx.size();
    _rotate<simd_size>(
            Txx.data(),
            Tyy.data(),
            Tzz.data(),
            Txy.data(),
            Tyz.data(),
            Tzx.data(),
            vx.data(),
            vy.data(),
            vz.data(),
            result[0].mutable_data(),
            result[1].mutable_data(),
            size);
    return result;
}

PYBIND11_MODULE(temperature_rotate, m) {
    m.doc() = "Rotate temperature matrix along a vector";
    m.def("rotate", &rotate_temp<float>,
            py::arg("Txx"), py::arg("Tyy"), py::arg("Tzz"),
            py::arg("Txy"), py::arg("Tyz"), py::arg("Tzx"),
            py::arg("vx"), py::arg("vy"), py::arg("vz"));
    m.def("rotate", &rotate_temp<double>,
            py::arg("Txx"), py::arg("Tyy"), py::arg("Tzz"),
            py::arg("Txy"), py::arg("Tyz"), py::arg("Tzx"),
            py::arg("vx"), py::arg("vy"), py::arg("vz"));
}
