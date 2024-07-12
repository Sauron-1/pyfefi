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
        const Real Txz,
        const Real Tyz,
        const Real Bx,
        const Real By,
        const Real Bz,
        Real& __restrict outxx,
        Real& __restrict outyy,
        Real& __restrict outzz,
        Real& __restrict outxy,
        Real& __restrict outxz,
        Real& __restrict outyz) {
    const Real B2_yz = By*By + Bz*Bz;
    const Real B_norm_inv = 1 / sqrt(Bx*Bx + B2_yz);
    const Real cos_theta = Bx * B_norm_inv,
               sin_theta = max(sqrt(B2_yz) * B_norm_inv, Real(1e-5));
    /*
    if (sin_theta < 1e-5) {
        outxx = Txx;
        outyy = Tyy;
        outzz = Tzz;
        outxy = Txy;
        outxz = Txz;
        outyz = Tyz;
    }
    */
    constexpr Real ux = 0;
    const Real uy = Bz / sin_theta * B_norm_inv,
               uz = -By / sin_theta * B_norm_inv;

    array<Real, 9> R{
        cos_theta + ux*ux*(1-cos_theta),
        ux*uy*(1-cos_theta) - uz*sin_theta,
        ux*uz*(1-cos_theta) + uy*sin_theta,

        uy*ux*(1-cos_theta) + uz*sin_theta,
        cos_theta + uy*uy*(1-cos_theta),
        uy*uz*(1-cos_theta) - ux*sin_theta,

        uz*ux*(1-cos_theta) - uy*sin_theta,
        uz*uy*(1-cos_theta) + ux*sin_theta,
        cos_theta + uz*uz*(1-cos_theta)
    };

    const Real x0 = R[0]*Txx + R[1]*Txy + R[2]*Txz;
    const Real x1 = R[0]*Txy + R[1]*Tyy + R[2]*Tyz;
    const Real x2 = R[0]*Txz + R[1]*Tyz + R[2]*Tzz;
    const Real x3 = R[3]*Txx + R[4]*Txy + R[5]*Txz;
    const Real x4 = R[3]*Txy + R[4]*Tyy + R[5]*Tyz;
    const Real x5 = R[3]*Txz + R[4]*Tyz + R[5]*Tzz;

    outxx = R[0]*x0 + R[1]*x1 + R[2]*x2;
    outyy = R[3]*x3 + R[4]*x4 + R[5]*x5;
    outzz = R[6]*(R[6]*Txx + R[7]*Txy + R[8]*Txz) + R[7]*(R[6]*Txy + R[7]*Tyy + R[8]*Tyz) + R[8]*(R[6]*Txz + R[7]*Tyz + R[8]*Tzz);
    outxy = R[3]*x0 + R[4]*x1 + R[5]*x2;
    outxz = R[6]*x0 + R[7]*x1 + R[8]*x2;
    outyz = R[6]*x3 + R[7]*x4 + R[8]*x5;
}

template<size_t N, typename Real>
static inline void rotate_kernel_v(
        const Real* __restrict _Txx,
        const Real* __restrict _Tyy,
        const Real* __restrict _Tzz,
        const Real* __restrict _Txy,
        const Real* __restrict _Txz,
        const Real* __restrict _Tyz,
        const Real* __restrict _Bx,
        const Real* __restrict _By,
        const Real* __restrict _Bz,
        Real* __restrict outxx,
        Real* __restrict outyy,
        Real* __restrict outzz,
        Real* __restrict outxy,
        Real* __restrict outxz,
        Real* __restrict outyz) {
    using vec = xsimd::make_sized_batch_t<Real, N>;
    static_assert(not std::is_void_v<vec>);

    auto Txx = vec::load_unaligned(_Txx);
    auto Tyy = vec::load_unaligned(_Tyy);
    auto Tzz = vec::load_unaligned(_Tzz);
    auto Txy = vec::load_unaligned(_Txy);
    auto Txz = vec::load_unaligned(_Txz);
    auto Tyz = vec::load_unaligned(_Tyz);
    auto Bx = vec::load_unaligned(_Bx);
    auto By = vec::load_unaligned(_By);
    auto Bz = vec::load_unaligned(_Bz);

    const vec B2_yz = By*By + Bz*Bz;
    const vec B_norm_inv = Real(1.0) / sqrt(Bx*Bx + B2_yz);
    const vec cos_theta = Bx * B_norm_inv,
               sin_theta = max(sqrt(B2_yz) * B_norm_inv, vec(1e-5));

    constexpr Real ux = 0;
    const vec uy = Bz / sin_theta * B_norm_inv,
               uz = -By / sin_theta * B_norm_inv;

    array<vec, 9> R{
        cos_theta + ux*ux*(1-cos_theta),
        ux*uy*(1-cos_theta) - uz*sin_theta,
        ux*uz*(1-cos_theta) + uy*sin_theta,

        uy*ux*(1-cos_theta) + uz*sin_theta,
        cos_theta + uy*uy*(1-cos_theta),
        uy*uz*(1-cos_theta) - ux*sin_theta,

        uz*ux*(1-cos_theta) - uy*sin_theta,
        uz*uy*(1-cos_theta) + ux*sin_theta,
        cos_theta + uz*uz*(1-cos_theta)
    };

    const vec x0 = R[0]*Txx + R[1]*Txy + R[2]*Txz;
    const vec x1 = R[0]*Txy + R[1]*Tyy + R[2]*Tyz;
    const vec x2 = R[0]*Txz + R[1]*Tyz + R[2]*Tzz;
    const vec x3 = R[3]*Txx + R[4]*Txy + R[5]*Txz;
    const vec x4 = R[3]*Txy + R[4]*Tyy + R[5]*Tyz;
    const vec x5 = R[3]*Txz + R[4]*Tyz + R[5]*Tzz;

    const vec out1 = R[0]*x0 + R[1]*x1 + R[2]*x2;
    const vec out2 = R[3]*x3 + R[4]*x4 + R[5]*x5;
    const vec out3 = R[6]*(R[6]*Txx + R[7]*Txy + R[8]*Txz) + R[7]*(R[6]*Txy + R[7]*Tyy + R[8]*Tyz) + R[8]*(R[6]*Txz + R[7]*Tyz + R[8]*Tzz);
    const vec out4 = R[3]*x0 + R[4]*x1 + R[5]*x2;
    const vec out5 = R[6]*x0 + R[7]*x1 + R[8]*x2;
    const vec out6 = R[6]*x3 + R[7]*x4 + R[8]*x5;

    out1.store_unaligned(outxx);
    out2.store_unaligned(outyy);
    out3.store_unaligned(outzz);
    out4.store_unaligned(outxy);
    out5.store_unaligned(outxz);
    out6.store_unaligned(outyz);
}

template<size_t pack_size, typename Real>
static inline void _rotate(
        const Real* __restrict Txx,
        const Real* __restrict Tyy,
        const Real* __restrict Tzz,
        const Real* __restrict Txy,
        const Real* __restrict Txz,
        const Real* __restrict Tyz,
        const Real* __restrict vx,
        const Real* __restrict vy,
        const Real* __restrict vz,
        Real* __restrict outxx,
        Real* __restrict outyy,
        Real* __restrict outzz,
        Real* __restrict outxy,
        Real* __restrict outxz,
        Real* __restrict outyz,
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
                &Txz[i*pack_size],
                &Tyz[i*pack_size],
                &vx[i*pack_size],
                &vy[i*pack_size],
                &vz[i*pack_size],
                &outxx[i*pack_size],
                &outyy[i*pack_size],
                &outzz[i*pack_size],
                &outxy[i*pack_size],
                &outxz[i*pack_size],
                &outyz[i*pack_size]
            );
    }
    for (uint64_t i = 0; i < remains; ++i) {
        rotate_kernel(
                Txx[i+packed_size],
                Tyy[i+packed_size],
                Tzz[i+packed_size],
                Txy[i+packed_size],
                Txz[i+packed_size],
                Tyz[i+packed_size],
                vx[i+packed_size],
                vy[i+packed_size],
                vz[i+packed_size],
                outxx[i+packed_size],
                outyy[i+packed_size],
                outzz[i+packed_size],
                outxy[i+packed_size],
                outxz[i+packed_size],
                outyz[i+packed_size]
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
        const py::array_t<Real>& Txz,
        const py::array_t<Real>& Tyz,
        const py::array_t<Real>& vx,
        const py::array_t<Real>& vy,
        const py::array_t<Real>& vz) {
    constexpr size_t simd_size = xsimd::simd_type<Real>::size;
    std::vector<py::array_t<Real>> result;
    assert_same_shape_strides(Txx, Tyy, Tzz, Txy, Txz, Tyz, vx, vy, vz);
    std::vector<size_t> shape, strides;
    for (auto i = 0; i < Txx.ndim(); ++i) {
        shape.push_back(Txx.shape(i));
        strides.push_back(Txx.strides(i));
    }
    for (auto i = 0; i < 6; ++i)
        result.emplace_back(shape, strides);
    size_t size = Txx.size();
    _rotate<simd_size>(
            Txx.data(),
            Tyy.data(),
            Tzz.data(),
            Txy.data(),
            Txz.data(),
            Tyz.data(),
            vx.data(),
            vy.data(),
            vz.data(),
            result[0].mutable_data(),
            result[1].mutable_data(),
            result[2].mutable_data(),
            result[3].mutable_data(),
            result[4].mutable_data(),
            result[5].mutable_data(),
            size);
    return result;
}

PYBIND11_MODULE(temperature_rotate, m) {
    m.doc() = "Rotate temperature matrix along a vector";
    m.def("rotate", &rotate_temp<float>,
            py::arg("Txx"), py::arg("Tyy"), py::arg("Tzz"),
            py::arg("Txy"), py::arg("Txz"), py::arg("Tyz"),
            py::arg("vx"), py::arg("vy"), py::arg("vz"));
    m.def("rotate", &rotate_temp<double>,
            py::arg("Txx"), py::arg("Tyy"), py::arg("Tzz"),
            py::arg("Txy"), py::arg("Txz"), py::arg("Tyz"),
            py::arg("vx"), py::arg("vy"), py::arg("vz"));
}
