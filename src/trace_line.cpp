#include <array>
#include <vector>

#include <farray.hpp>
#include <simd.hpp>
#include <tuple_algebra/tuple_algebra.hpp>
#include <rk.hpp>

//using T = double;
//constexpr size_t N = 3;

template<typename T, size_t N>
class LineTracer {

    public:
        template<typename...Arr>
            requires( sizeof...(Arr) == N )
        LineTracer(py::array_t<Arr>...arrs) {
            std::assign(data, std::make_tuple(FArray<T, N>(arrs)...));
            for (auto i = 0; i < N; ++i)
                delta[i] = 1;
            for (auto i = 0; i < N; ++i)
                start[i] = 0;
            scale = std::sqrt(std::dot(delta, delta)) / T(N);
        }

        template<typename...Arr>
            requires( sizeof...(Arr) == N+2 )
        LineTracer(py::array_t<Arr>...arrs) {
            auto args = std::make_tuple(arrs...);
            auto args1 = std::firstN<N>(args);
            std::assign(data,
                    std::apply_unary_op([](auto val) { return FArray<T, N>(val); },
                        args1));
            for (auto i = 0; i < N; ++i)
                for (auto i = 0; i < N; ++i) {
                    delta[i] = std::get<N>(args).at(i);
                    start[i] = std::get<N+1>(args).at(i);
                }
            scale = std::sqrt(std::dot(delta, delta)) / T(N);
        }

        auto eval(std::array<T, N> coord) const {
            std::array<T, N> ret;
            std::assign(coord, convert(coord));
            for (auto i = 0; i < N; ++i) {
                ret[i] = data[i](coord);
            }
            return ret;
        }

        auto operator()([[maybe_unused]] T s, std::array<T, N> coord) const {
            std::array<T, N> ret = eval(coord);
            T norm = std::sqrt(std::dot(ret, ret));
            std::assign(ret, ret / norm);
            return ret;
        }

        template<bool neg>
        auto step(std::array<T, N>& coord, T step, T tol, T tol_rel, T max_step, T min_step) const {
            auto stepper = make_runge_kutta<T, N, 4>(*this);
            auto [result, real_step, err] = stepper.template step_adaptive<neg>(0, coord, step, tol, tol_rel, max_step, min_step);
            return std::make_pair(result, real_step);
        }

        template<bool neg = false>
        std::vector<std::array<T, N>> trace(
                std::array<T, N> init,
                T step_size, T tol, T tol_rel,
                T max_step, T min_step, size_t max_iter,
                T min_dist, T term_val) const {
            std::vector<std::array<T, N>> result;
            result.push_back(init);
            while (not terminate(result.back(), term_val) and --max_iter > 0) {
                auto [res, real_step] = step<neg>(result.back(), step_size, tol, tol_rel, max_step, min_step);
                step_size = real_step;
                result.push_back(res);
                if (min_dist > 0) {
                    auto d = dist(init, result.back(), result[result.size()-2]);
                    if (result.size() > 10 and d < min_dist)
                        break;
                }
            }
            return result;
        }

        //using Real = double;
        template<typename Real>
        py::array bidir_trace(
                py::array_t<T, py::array::forcecast> init,
                Real step_size, Real tol, Real tol_rel, Real max_step, Real min_step, size_t max_iter, T min_dist, Real term_val) {
            std::array<T, N> coord;
            for (auto i = 0; i < N; ++i) {
                coord[i] = init.at(i);
            }
            if (step_size < min_dist)
                step_size = min_dist * 2;
            auto res_neg = trace<true>(coord, step_size, tol, tol_rel, max_step, min_step, max_iter, min_dist, term_val);
            auto res_pos = trace<false>(coord, step_size, tol, tol_rel, max_step, min_step, max_iter, min_dist, term_val);
            size_t len_neg = res_neg.size(),
                   len_pos = res_pos.size();
            size_t length = len_neg + len_pos - 1;
            std::array<size_t, 2> result_shape{ length, N };
            py::array_t<T> result(result_shape);
            T* ptr = result.mutable_data();
            for (auto i = 0; i < len_neg; ++i) {
                for (auto j = 0; j < N; ++j) {
                    ptr[i*N + j] = res_neg[len_neg-i-1][j];
                }
            }
            for (auto i = 0; i < len_pos-1; ++i) {
                for (auto j = 0; j < N; ++j) {
                    ptr[(i+len_neg)*N + j] = res_pos[i+1][j];
                }
            }
            return result;
        }

        bool terminate(std::array<T, N> coord, T term_val) const {
            std::array<T, N> coord_real;
            std::assign(coord_real, convert(coord));
            if (data[0].is_out(coord_real)) {
                //std::cerr << "Terminating: out of bounds" << std::endl;
                return true;
            }
            auto val = eval(coord);
            T norm = std::sqrt(std::dot(val, val));
            /*
            if (norm < term_val)
                std::cerr << "Terminating: zero field vector" << std::endl;
            */
            return norm < term_val;
        }

        T dist(std::array<T, N> init, std::array<T, N> p1, std::array<T, N> p2) const {
            auto diff1 = init - p1;
            auto diff2 = init - p2;
            auto diff = p1 - p2;
            auto d_i_p1_2 = std::dot(diff1, diff1);
            if (std::dot(diff, diff1) < 0 and std::dot(diff, diff2) > 0) {
                auto diff_norm = diff / std::sqrt(std::dot(diff, diff));
                auto project = std::abs(std::dot(diff_norm, diff1));
                return std::sqrt(d_i_p1_2 - project*project);
            }
            return std::sqrt(d_i_p1_2);
        }

    private:
        std::array<FArray<T, N>, N> data;
        std::array<T, N> delta, start;
        T scale;

        auto convert(std::array<T, N> coord) const {
            std::array<T, N> result;
            std::assign(result, (coord - start) / delta);
            return result;
        }

};

#define DEFAULTS \
    py::arg("init"),  \
    py::arg("step_size") = 0.01,  \
    py::arg("tol") = 1e-6,  \
    py::arg("tol_rel") = 1e-6,  \
    py::arg("max_step") = 1,  \
    py::arg("min_step") = 1e-4, \
    py::arg("min_iter") = 5000, \
    py::arg("min_dist") = 0.0, \
    py::arg("term_val") = 1e-2

PYBIND11_MODULE(line_tracer, m) {

    py::class_<LineTracer<float, 2>>(m, "LineTracer2")
        .def(py::init<const py::array_t<float>, const py::array_t<float>>(),
                py::arg("vx"), py::arg("vy"))
        .def(py::init<const py::array_t<float>, const py::array_t<float>, const py::array_t<float>, const py::array_t<float>>(),
                py::arg("vx"), py::arg("vy"), py::arg("delta"), py::arg("start"))
        .def("trace", &LineTracer<float, 2>::bidir_trace<double>, DEFAULTS)
        .def("trace", &LineTracer<float, 2>::bidir_trace<float>, DEFAULTS);

    py::class_<LineTracer<float, 3>>(m, "LineTracer3")
        .def(py::init<const py::array_t<float>, const py::array_t<float>, const py::array_t<float>, const py::array_t<float>, const py::array_t<float>>(),
                py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("delta"), py::arg("start"))
        .def("trace", &LineTracer<float, 3>::bidir_trace<double>, DEFAULTS)
        .def("trace", &LineTracer<float, 3>::bidir_trace<float>, DEFAULTS);

}
