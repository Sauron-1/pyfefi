#include <array>
#include <vector>

#include <farray.hpp>
#include <simd.hpp>
#include <tuple_algebra/tuple_algebra.hpp>
#include <rk.hpp>

#include <maybe_omp.hpp>

//using T = double;
//constexpr size_t N = 3;

#define TRACE_LINE_OMP_SHEDULE dynamic

template<typename T>
struct TraceConfig {
    T step_size,
      tol,
      tol_rel,
      max_step,
      min_step,
      min_dist,
      term_val;
    size_t max_iter;

    void print() const {
        std::cout << "step_size: " << step_size << std::endl
                  << "tol: " << tol << std::endl
                  << "tol_rel: " << tol_rel << std::endl
                  << "max_step: " << max_step << std::endl
                  << "min_step: " << min_step << std::endl
                  << "min_dist: " << min_dist << std::endl
                  << "term_val: " << term_val << std::endl
                  << "max_iter: " << max_iter << std::endl;
    }
};

template<typename T, size_t N>
class LineTracer {

    public:
        template<typename...Arr>
            requires( sizeof...(Arr) == N )
        LineTracer(const py::array_t<Arr, py::array::forcecast>&...arrs) {
            std::assign(data, std::make_tuple(FArray<T, N>(arrs)...));
            for (auto i = 0; i < N; ++i)
                delta[i] = 1;
            for (auto i = 0; i < N; ++i)
                start[i] = 0;
            scale = std::sqrt(std::dot(delta, delta)) / T(N);
        }

        template<typename...Arr>
            requires( sizeof...(Arr) == N+2 )
        LineTracer(const py::array_t<Arr, py::array::forcecast>&...arrs) {
            auto args = std::tie(arrs...);
            auto args1 = std::firstN<N>(args);
            std::assign(data,
                    std::apply_unary_op([](auto& val) { return FArray<T, N>(val); },
                        args1));
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
        auto step(std::array<T, N>& coord, TraceConfig<T> cfg) const {
            auto stepper = make_runge_kutta<T, N, 4>(*this);
            auto [result, real_step, err] = stepper.template step_adaptive<neg>(
                    0, coord, cfg.step_size, cfg.tol, cfg.tol_rel, cfg.max_step, cfg.min_step);
            return std::make_pair(result, real_step);
        }

        template<bool neg = false>
        std::vector<std::array<T, N>> trace(
                std::array<T, N> init,
                TraceConfig<T> cfg) const {
            std::vector<std::array<T, N>> result;
            result.push_back(init);
            while (not terminate(result.back(), cfg.term_val) and --cfg.max_iter > 0) {
                auto [res, real_step] = step<neg>(result.back(), cfg);
                cfg.step_size = real_step;
                result.push_back(res);
                if (cfg.min_dist > 0) {
                    auto d = dist(init, result.back(), result[result.size()-2]);
                    if (result.size() > 10 and d < cfg.min_dist)
                        break;
                }
            }
            return result;
        }

        std::vector<std::array<T, N>> bidir_trace(
                std::array<T, N> init,
                TraceConfig<T> cfg) const {
            auto res_neg = trace<true>(init, cfg);
            auto res_pos = trace<false>(init, cfg);
            size_t len_neg = res_neg.size(),
                   len_pos = res_pos.size();
            size_t length = len_neg + len_pos - 1;
            res_pos.resize(length);
            // move the positive part to the end of the vector
            std::move_backward(res_pos.begin(), res_pos.begin() + len_pos, res_pos.end());
            // reverse the negative part and copy them to the beginning of the vector
            std::reverse(res_neg.begin(), res_neg.end());
            std::copy(res_neg.begin(), res_neg.end(), res_pos.begin());
            return res_pos;
        }

        template<typename Real, int dir=0>
        py::array trace_one_py(
                py::array_t<T, py::array::forcecast> init,
                Real step_size, Real tol, Real tol_rel, Real max_step, Real min_step, size_t max_iter, T min_dist, Real term_val) const {
            TraceConfig<T> cfg {
                .step_size = T(step_size),
                .tol = T(tol),
                .tol_rel = T(tol_rel),
                .max_step = T(max_step),
                .min_step = T(min_step),
                .min_dist = T(min_dist),
                .term_val = T(term_val),
                .max_iter = max_iter,
            };
            cfg.print();
            std::array<T, N> coords;
            for (auto i = 0; i < N; ++i)
                coords[i] = init.at(i);
            if constexpr (dir == 0) {
                auto result = bidir_trace(coords, cfg);
                return to_numpy(result);
            }
            else if constexpr (dir == 1) {
                auto result = trace<false>(coords, cfg);
                return to_numpy(result);
            }
            else if constexpr (dir == -1) {
                auto result = trace<true>(coords, cfg);
                return to_numpy(result);
            }
        }

        template<typename Real>
        py::list trace_many(
                py::array_t<T, py::array::forcecast> inits,
                Real step_size, Real tol, Real tol_rel, Real max_step, Real min_step, size_t max_iter, T min_dist, Real term_val) const {
            if (inits.ndim() != 2 or inits.shape(1) != N) {
                throw std::runtime_error("inits must be a (num_points, dim) array");
            }
            TraceConfig<T> cfg {
                .step_size = T(step_size),
                .tol = T(tol),
                .tol_rel = T(tol_rel),
                .max_step = T(max_step),
                .min_step = T(min_step),
                .min_dist = T(min_dist),
                .term_val = T(term_val),
                .max_iter = max_iter,
            };
            size_t num_points = inits.shape(0);
            std::vector<std::array<T, N>> coords(num_points);
            for (auto i = 0; i < num_points; ++i)
                for (auto j = 0; j < N; ++j)
                    coords[i][j] = inits.at(i, j);

            std::vector<py::array> results(num_points);
#pragma omp parallel for schedule(TRACE_LINE_OMP_SHEDULE)
            for (auto i = 0; i < num_points; ++i) {
                auto result = bidir_trace(coords[i], cfg);
                results[i] = to_numpy(result);
            }

            py::list res;
            for (auto& a : results) {
                res.append(a);
            }

            return res;
        }

        template<typename Real>
        py::array find_roots(
                py::array_t<T, py::array::forcecast> inits,
                Real step_size, Real tol, Real tol_rel, Real max_step, Real min_step, size_t max_iter, T min_dist, Real term_val) const {
            if (inits.ndim() != 2 or inits.shape(1) != N) {
                throw std::runtime_error("inits must be a (num_points, dim) array");
            }
            TraceConfig<T> cfg {
                .step_size = T(step_size),
                .tol = T(tol),
                .tol_rel = T(tol_rel),
                .max_step = T(max_step),
                .min_step = T(min_step),
                .min_dist = T(min_dist),
                .term_val = T(term_val),
                .max_iter = max_iter,
            };
            size_t num_points = inits.shape(0);
            std::vector<std::array<T, N>> coords(num_points);
            for (auto i = 0; i < num_points; ++i)
                for (auto j = 0; j < N; ++j)
                    coords[i][j] = inits.at(i, j);

            std::array<size_t, 3> result_shape{ num_points, 4, N };
            py::array_t<T> result(result_shape);
            T* ptr = result.mutable_data();
#pragma omp parallel for schedule(TRACE_LINE_OMP_SHEDULE)
            for (auto i = 0; i < num_points; ++i) {
                auto pos = trace<false>(coords[i], cfg);
                auto neg = trace<true>(coords[i], cfg);
                for (auto j = 0; j < 2; ++j)
                    if (j < neg.size())
                        for (auto d = 0; d < N; ++d)
                            ptr[i*4*N + j*N + d] = neg[neg.size()-j-1][d];
                for (auto j = 0; j < 2; ++j)
                    if (j < pos.size())
                        for (auto d = 0; d < N; ++d)
                            ptr[i*4*N + (1-j)*N + d] = pos[pos.size()-j-1][d];
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
            //if (norm < term_val) {
            //    std::cerr << "Terminating: zero field vector" << std::endl;
            //    for (auto i = 0; i < N; ++i)
            //        std::cerr << coord[i] << std::endl;
            //}
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

        py::array to_numpy(std::vector<std::array<T, N>> coords) const {
            std::array<size_t, 2> shape{ coords.size(), N };
            py::array_t<T> result(shape);
            T* ptr = result.mutable_data();
            for (auto i = 0; i < coords.size(); ++i) {
                for (auto j = 0; j < N; ++j) {
                    ptr[i*N + j] = coords[i][j];
                }
            }
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
    py::arg("max_iter") = 5000, \
    py::arg("min_dist") = 0.0, \
    py::arg("term_val") = 1e-2

using arg_t = py::array_t<float, py::array::forcecast>;

PYBIND11_MODULE(line_tracer, m) {

    py::class_<LineTracer<float, 2>>(m, "LineTracer2")
        .def(py::init<const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"))
        .def(py::init<const arg_t, const arg_t, const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"), py::arg("delta"), py::arg("start"))
        .def("trace", &LineTracer<float, 2>::trace_one_py<double, 0>, DEFAULTS)
        .def("trace", &LineTracer<float, 2>::trace_one_py<float, 0>, DEFAULTS)
        .def("trace_many", &LineTracer<float, 2>::trace_many<float>, DEFAULTS)
        .def("trace_many", &LineTracer<float, 2>::trace_many<double>, DEFAULTS)
        .def("find_roots", &LineTracer<float, 2>::find_roots<float>, DEFAULTS)
        .def("find_roots", &LineTracer<float, 2>::find_roots<double>, DEFAULTS);

    py::class_<LineTracer<float, 3>>(m, "LineTracer3")
        .def(py::init<const arg_t, const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"), py::arg("vz"))
        .def(py::init<const arg_t, const arg_t, const arg_t,
                const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("delta"), py::arg("start"))
        .def("trace", &LineTracer<float, 3>::trace_one_py<double, 0>, DEFAULTS)
        .def("trace", &LineTracer<float, 3>::trace_one_py<float, 0>, DEFAULTS)
        .def("trace_many", &LineTracer<float, 3>::trace_many<float>, DEFAULTS)
        .def("trace_many", &LineTracer<float, 3>::trace_many<double>, DEFAULTS)
        .def("find_roots", &LineTracer<float, 3>::find_roots<float>, DEFAULTS)
        .def("find_roots", &LineTracer<float, 3>::find_roots<double>, DEFAULTS);

}
