/**
 * @file cartesian_mod.cpp
 * @brief Modified cartesian coordinate system
 * @details This file contains the implementation coordinates conversion
 *         between cartesian and modified cartesian coordinate system.
 */
#include "ndarray.hpp"
#include <pybind11/pytypes.h>
#include <vector>
#include <cmath>
#include <iostream>

template<typename Real>
static inline Real mytanh(Real x, Real x0, Real d, Real a1, Real a2) {
    return 0.5 * (a2 + a1) + 0.5 * (a2 - a1) * tanh((x-x0) / d);
}

template<typename Real, typename FnLeft, typename FnRight>
static inline std::vector<Real> integral(Real dx0, Real xlo, Real xhi, FnLeft&& fnleft, FnRight&& fnright) {
    std::vector<Real> xs(4000);

    size_t num_left = 0,
           idx = 0;

    Real x = dx0,
         dx = dx0;
    for (auto i = 0; i < 2000; ++i) {
        x -= dx;
        dx *= fnleft(x);
        xs[idx] = x;
        ++idx;
        ++num_left;
        if (x <= xlo - 4*dx) break;
    }

    x = 0;
    dx = dx0;
    for (auto i = 0; i < 2000; ++i) {
        x += dx;
        dx *= fnright(x);
        xs[idx] = x;
        ++idx;
        if (x >= xhi + 4*dx) break;
    }

    std::vector<Real> p2x(idx);
    for (auto i = 0; i < num_left; ++i) {
        p2x[i] = xs[num_left-i-1];
    }
    for (auto i = num_left; i < idx; ++i) {
        p2x[i] = xs[i];
    }

    return p2x;
}

template<typename Real>
class CMAxis {

    public:
        template<typename FnLeft, typename FnRight>
        CMAxis(Real dx0, Real _xlo, Real _xhi, FnLeft&& fnleft, FnRight&& fnright) {
            init(dx0, _xlo, _xhi, std::forward<FnLeft>(fnleft), std::forward<FnRight>(fnright));
        }

        CMAxis() = default;

        template<typename FnLeft, typename FnRight>
        void init(Real dx0, Real _xlo, Real _xhi, FnLeft&& fnleft, FnRight&& fnright) {
            p2x_ = integral(dx0, _xlo, _xhi, fnleft, fnright);
            int extra = std::max(int(dx0*20), 4);
            size_t len = p2x_.size();
            Real xlo = p2x_[4],
                 xhi = p2x_[len-5];
            int idx_min = int(xlo*100) - extra,
                idx_max = int(xhi*100) + extra;
            int num = idx_max - idx_min + 1;

            x2p_.resize(num);
            for (auto& val : x2p_) val = 0.0;

            int i0 = int(xlo*100)-2;
            int i1 = int(xhi*100)+2;
            int _j = 4;
            for (int i = i0; i < i1+1; ++i) {
                int idx = i - idx_min;
                Real x = Real(i) / 100.;
                for (int j = _j; j < len; ++j) {
                    if (x < p2x_[j]) {
                        _j = j - 1;
                        x2p_[idx] = Real(j-4) - (p2x_[j] - x) / (p2x_[j] - p2x_[j-1]);
                        break;
                    }
                }
            }
            idx_min_ = idx_min;
        }

        Real get(Real p) const {
            int i = int(p);
            Real w1 = p - i;
            //Real w0 = 1 - w1;
            i += 4;
            return p2x_[i] + (p2x_[i+1] - p2x_[i])*w1 + 0.5*(p2x_[i+1]-2*p2x_[i]+p2x_[i-1])*w1*w1 + \
                (p2x_[i+2] - 3*p2x_[i+1] + 3*p2x_[i] - p2x_[i-1]) * w1*w1*w1 / 6;
        }

        Real getr(Real x) const {
            size_t len = p2x_.size();
            Real xlo = p2x_[4],
                 xhi = p2x_[len-5];
            x = std::max(xlo, std::min(x, xhi)) * 100;
            int i = int(x);
            Real w1 = x - i;
            //Real w0 = 1 - w1;
            i -= idx_min_;
            return x2p_[i] + (x2p_[i+1] - x2p_[i])*w1 + 0.5*(x2p_[i+1]-2*x2p_[i]+x2p_[i-1])*w1*w1 + \
                (x2p_[i+2] - 3*x2p_[i+1] + 3*x2p_[i] - x2p_[i-1]) * w1*w1*w1 / 6;
        }

        size_t num_grids() const {
            return p2x_.size() - 8;
        }

    private:
        std::vector<Real> p2x_, x2p_;
        int idx_min_;
};

template<typename Real>
class CartesianModCore {

    public:
        CartesianModCore(const py::array_t<Real> diff, const py::array_t<Real> lims) :
            xaxis(diff.at(0), lims.at(0, 0), lims.at(0, 1),
                    [](Real x) { return mytanh<Real>(x, -2, 1, 1.005, 1); },
                    [](Real x) { return mytanh<Real>(x, 1.2, 1, 1, 1.005); }),
            yaxis(diff.at(1), lims.at(1, 0), lims.at(1, 1),
                    [](Real x) { return mytanh<Real>(x, -1.5, 1, 1.01, 1); },
                    [](Real x) { return mytanh<Real>(x, 1.5, 1, 1, 1.01); }),
            zaxis(diff.at(2), lims.at(2, 0), lims.at(2, 1),
                    [](Real x) { return mytanh<Real>(x, -1.5, 1, 1.01, 1); },
                    [](Real x) { return mytanh<Real>(x, 1.5, 1, 1, 1.01); }) {}

        CartesianModCore(const py::array_t<Real> diff, const py::array_t<Real> lims, const py::array_t<Real> conf) {
            std::array<std::array<Real, 4>, 6> args;
            for (auto i = 0; i < 6; ++i)
                for (auto j = 0; j < 4; ++j)
                    args[i][j] = conf.at(i, j);
            xaxis.init(diff.at(0), lims.at(0, 0), lims.at(0, 1),
                    [=](Real x) { return mytanh<Real>(x, args[0][0], args[0][1], args[0][2], args[0][3]); },
                    [=](Real x) { return mytanh<Real>(x, args[1][0], args[1][1], args[1][2], args[1][3]); });
            yaxis.init(diff.at(1), lims.at(1, 0), lims.at(1, 1),
                    [=](Real x) { return mytanh<Real>(x, args[2][0], args[2][1], args[2][2], args[2][3]); },
                    [=](Real x) { return mytanh<Real>(x, args[3][0], args[3][1], args[3][2], args[3][3]); });
            zaxis.init(diff.at(2), lims.at(2, 0), lims.at(2, 1),
                    [=](Real x) { return mytanh<Real>(x, args[4][0], args[4][1], args[4][2], args[4][3]); },
                    [=](Real x) { return mytanh<Real>(x, args[5][0], args[5][1], args[5][2], args[5][3]); });
        }

        py::array grid_sizes() const {
            py::array_t<size_t> result(std::array<size_t, 1>{3});
            auto ptr = result.mutable_data();
            ptr[0] = xaxis.num_grids();
            ptr[1] = yaxis.num_grids();
            ptr[2] = zaxis.num_grids();
            return result;
        }

        py::tuple to_cartesian(py::array_t<Real> p, py::array_t<Real> q, py::array_t<Real> w) const {
            size_t size = p.size();
            if (!has_same_shape(p, q, w))
                throw std::runtime_error("p, q, w must have same shape");
            if (!has_same_stride(p, q, w))
                throw std::runtime_error("p, q, w must have same stride");
            py::array_t<Real>
                x(get_shape(p), get_strides(p)),
                y(get_shape(q), get_strides(q)),
                z(get_shape(w), get_strides(w));
            const Real *p_ptr = p.data(),
                       *q_ptr = q.data(),
                       *w_ptr = w.data();
            Real *x_ptr = x.mutable_data(),
                 *y_ptr = y.mutable_data(),
                 *z_ptr = z.mutable_data();
            size_t p_st = get_min_stride(p),
                   q_st = get_min_stride(q),
                   w_st = get_min_stride(w);
#pragma omp parallel for
            for (auto i = 0; i < size; ++i) {
                x_ptr[i] = xaxis.get(p_ptr[i*p_st]);
                y_ptr[i] = yaxis.get(q_ptr[i*q_st]);
                z_ptr[i] = zaxis.get(w_ptr[i*w_st]);
            }
            return py::make_tuple(x, y, z);
        }

        py::tuple from_cartesian(py::array_t<Real> x, py::array_t<Real> y, py::array_t<Real> z) const {
            size_t size = x.size();
            if (!has_same_shape(x, y, z))
                throw std::runtime_error("x, y, z must have same shape");
            if (!has_same_stride(x, y, z))
                throw std::runtime_error("x, y, z must have same stride");
            py::array_t<Real>
                p(get_shape(x), get_strides(x)),
                q(get_shape(y), get_strides(y)),
                w(get_shape(z), get_strides(z));
            const Real *x_ptr = x.data(),
                       *y_ptr = y.data(),
                       *z_ptr = z.data();
            Real *p_ptr = p.mutable_data(),
                 *q_ptr = q.mutable_data(),
                 *w_ptr = w.mutable_data();
            size_t x_st = get_min_stride(x),
                   y_st = get_min_stride(y),
                   z_st = get_min_stride(z);
#pragma omp parallel for
            for (auto i = 0; i < size; ++i) {
                p_ptr[i] = xaxis.getr(x_ptr[i*x_st]);
                q_ptr[i] = yaxis.getr(y_ptr[i*y_st]);
                w_ptr[i] = zaxis.getr(z_ptr[i*z_st]);
            }
            return py::make_tuple(p, q, w);
        }

    private:
        CMAxis<Real> xaxis, yaxis, zaxis;
};

PYBIND11_MODULE(cartesian_mod, m) {

    py::class_<CartesianModCore<double>>(m, "CartesianModCore")
        .def(py::init<const py::array_t<double>, const py::array_t<double>>())
        .def(py::init<const py::array_t<double>, const py::array_t<double>, const py::array_t<double>>())
        .def("to_cartesian", &CartesianModCore<double>::to_cartesian)
        .def("from_cartesian", &CartesianModCore<double>::from_cartesian)
        .def("grid_sizes", &CartesianModCore<double>::grid_sizes);

    py::class_<CartesianModCore<float>>(m, "CartesianModCoref")
        .def(py::init<const py::array_t<float>, const py::array_t<float>>())
        .def(py::init<const py::array_t<float>, const py::array_t<float>, const py::array_t<float>>())
        .def("to_cartesian", &CartesianModCore<float>::to_cartesian)
        .def("from_cartesian", &CartesianModCore<float>::from_cartesian)
        .def("grid_sizes", &CartesianModCore<float>::grid_sizes);

}
