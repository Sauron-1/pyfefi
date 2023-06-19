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
#include <interp/interp.hpp>

using namespace std;

#ifndef PYFEFI_INTERP_ORDER
#   define PYFEFI_INTERP_ORDER 2
#endif

template<typename Simd, typename Float>
static auto interp_kernel(
        array<const Float*, 3> targets, uint64_t num_targets,
        array<Float, 3> scale, array<Float, 3> lo, array<Float, 3> hi,
        NdArray<Float, 4> vars,
        NdArray<Float, 2> results) {
    constexpr size_t interp_order = PYFEFI_INTERP_ORDER;

    array<Simd, 3> target;
    constexpr size_t simd_len = simd_length<Simd>;
    const size_t num = vars.shape(3);
    auto num_simd = num_targets / simd_len;
    auto num_remain = num_targets % simd_len;
#pragma omp parallel
    {
        vector<Simd> buffer(num);
#pragma omp for
        for (auto i = 0; i < num_simd; ++i) {
            for (auto j = 0; j < 3; ++j) {
                target[j].load(targets[j] + i * simd_len);
            }
            interp_one<interp_order>(target, scale, lo, hi, vars, buffer);
            for (auto s = 0; s < simd_len; ++s) {
                for (auto j = 0; j < num; ++j) {
                    results(i*simd_len+s, j) = buffer[j][s];
                }
            }
        }
    }

    array<Float, simd_len> buf;
    for (auto j = 0; j < 3; ++j) {
        for (auto i = 0; i < num_remain; ++i) {
            buf[i] = targets[j][num_simd * simd_len + i];
        }
        for (auto i = num_remain; i < simd_len; ++i) {
            buf[i] = lo[j];
        }
        target[j].load(buf.data());
    }
    vector<Simd> buffer(num);
    interp_one<interp_order>(target, scale, lo, hi, vars, buffer);
    for (auto i = 0; i < num_remain; ++i) {
        for (auto s = 0; s < simd_len; ++s) {
            for (auto j = 0; j < num; ++j) {
                results(i*simd_len+s, j) = buffer[j][s];
            }
        }
    }
}

template<typename Float>
py::array interp(
        py::list coords, py::list pqw, const py::array_t<Float> var) {
    // extract p, q, w from coords
    auto p = coords[0].cast<py::array_t<Float>>();
    auto q = coords[1].cast<py::array_t<Float>>();
    auto w = coords[2].cast<py::array_t<Float>>();
    // p, q, w must be 1-D
    if (p.ndim() != 1 || q.ndim() != 1 || w.ndim() != 1) {
        throw std::runtime_error("p, q, w must be 1-D");
    }
    // lo, hi: first and last value of p, q, w
    array<Float, 3> lo, hi, scale;
    lo[0] = p.data()[0];
    lo[1] = q.data()[0];
    lo[2] = w.data()[0];
    hi[0] = p.data()[p.size()-1];
    hi[1] = q.data()[q.size()-1];
    hi[2] = w.data()[w.size()-1];
    scale[0] = p.data()[1] - p.data()[0];
    scale[1] = q.data()[1] - q.data()[0];
    scale[2] = w.data()[1] - w.data()[0];
    // extract p1, q1, w1 from pqw
    auto p1 = pqw[0].cast<py::array_t<Float>>();
    auto q1 = pqw[1].cast<py::array_t<Float>>();
    auto w1 = pqw[2].cast<py::array_t<Float>>();
    // p1, q1, w1 must have same shape and stride, while any dim is ok.
    if (p1.ndim() != q1.ndim() || q1.ndim() != w1.ndim()) {
        throw std::runtime_error("p1, q1, w1 must have same shape");
    }
    for (auto i = 0; i < p1.ndim(); ++i) {
        if (p1.shape(i) != q1.shape(i) || q1.shape(i) != w1.shape(i)) {
            throw std::runtime_error("p1, q1, w1 must have same shape");
        }
        if (p1.strides(i) != q1.strides(i) || q1.strides(i) != w1.strides(i)) {
            throw std::runtime_error("p1, q1, w1 must have same stride");
        }
    }

    auto var_dim = var.ndim();
    if (var_dim < 3) {
        throw std::runtime_error("var must have at least 3 dimensions");
    }

    // Create result array with shape (p1.size(), new_shape[3])
    const NdArray<Float, 4> var_arr(var);
    auto result_shape = array<size_t, 2>{size_t(p1.size()), var_arr.shape(3)};
    auto result = py::array_t<Float>(result_shape);
    NdArray<Float, 2> result_arr(result);

    // build target coords
    auto num_target = p1.size();
    array<const Float*, 3> targets {p1.data(), q1.data(), w1.data()};

    // invoke interp_kernel
    interp_kernel<typename native_simd_type<Float>::type, Float>(targets, num_target, scale, lo, hi, var_arr, result_arr);

    vector<size_t> result_shape1;
    for (auto i = 0; i < p1.ndim(); ++i) {
        result_shape1.push_back(p1.shape(i));
    }
    auto var_shape = var.shape();
    for (auto i = 3; i < var_dim; ++i) {
        result_shape1.push_back(var_shape[i]);
    }

    return result.reshape(result_shape1);
}

// export interp to python
PYBIND11_MODULE(interp, m) {
    m.doc() = "Interpolation";
    m.def("interp", &interp<float>, "Interpolation");
    m.def("interp", &interp<double>, "Interpolation");
}
