#include <cmath>
#include <numbers>

#include <pybind11/pytypes.h>

#include <ndarray.hpp>

template<typename Real>
FORCE_INLINE static Real fgrid(Real p) {
    return (0.11 * p - std::erf((p-10.0)/12.0) - erf((p+10.0)/12.0)) * 100.0*2.0/4.0/0.44;
}

template<typename Real>
FORCE_INLINE static Real fgrid_diff(Real p) {
    Real e1 = (p - 10.0) / 12.0,
         e2 = (p + 10.0) / 12.0;
    return 100.0*2.0/4.0/0.44 * (0.11 - 1.0/6.0/std::sqrt(std::numbers::pi_v<Real>) *
            (std::exp(-e1*e1) + std::exp(-e2*e2)));
}

template<typename Real>
FORCE_INLINE static Real solve_grid(Real x) {
    Real tol = 1e-6;
    int max_step = 100;
    Real p = 2;
    for (int i = 0; i < max_step; ++i) {
        Real xp = fgrid(p);
        Real delta = xp - x;
        if (fabs(delta) < tol)
            break;
        Real diff = fgrid_diff(p);
        p -= delta / diff;
    }
    return p;
}

template<typename Real>
class SphereModCore {
    public:
        py::tuple to_cartesian(
                py::array_t<Real> p,
                py::array_t<Real> q,
                py::array_t<Real> w) const {
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
                Real r = fgrid(p_ptr[i*p_st]),
                     theta = q_ptr[i*q_st],
                     phi = w_ptr[i*w_st];

                Real rho = r * sin(theta);

                y_ptr[i] = r * cos(theta);
                x_ptr[i] = rho * cos(phi);
                z_ptr[i] = -rho * sin(phi);
            }
            return py::make_tuple(x, y, z);
        }

        py::tuple from_cartesian(
                py::array_t<Real> x,
                py::array_t<Real> y,
                py::array_t<Real> z) const {
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
#pragma omp parallel for schedule(guided)
            for (auto i = 0; i < size; ++i) {
                Real _x = x_ptr[i*x_st],
                     _y = y_ptr[i*y_st],
                     _z = z_ptr[i*z_st];
                Real rho = std::hypot(_x, _z);
                Real r = std::hypot(rho, _y);

                p_ptr[i] = solve_grid(r);
                q_ptr[i] = std::atan2(rho, _y);
                w_ptr[i] = std::atan2(-_z, _x);
            }
            return py::make_tuple(p, q, w);
        }
};

PYBIND11_MODULE(sphere_mod, m) {

    py::class_<SphereModCore<double>>(m, "SphereModCore")
        .def(py::init<>())
        .def("to_cartesian", &SphereModCore<double>::to_cartesian)
        .def("from_cartesian", &SphereModCore<double>::from_cartesian);

    py::class_<SphereModCore<float>>(m, "SphereModCoref")
        .def(py::init<>())
        .def("to_cartesian", &SphereModCore<float>::to_cartesian)
        .def("from_cartesian", &SphereModCore<float>::from_cartesian);

    m.def("solve_grid", &solve_grid<float>, "Solve p value from r");
    m.def("solve_grid", &solve_grid<double>, "Solve p value from r");

}
