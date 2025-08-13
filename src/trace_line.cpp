#include <pyfefi.hpp>
#include <ndarray.hpp>
#include <utils.hpp>
#include <array>
#include <vector>

#include <simd.hpp>
#include <rk.hpp>
#include <tuple_arithmetic.hpp>
#include <algorithm>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <maybe_omp.hpp>
#include <picinterp.hpp>
#include "trace_grid.hpp"

namespace py = pybind11;

#define TRACE_LINE_OMP_SHEDULE dynamic
#define TRACE_LINE_INTERP_ORDER 2

template<typename T>
struct TraceConfig {
    T step_size,
      tol,
      tol_rel,
      max_step,
      min_step,
      min_dist,
      term_val;
    int max_iter;

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
        LineTracer(const py::array_t<Arr, py::array::forcecast>&...arrs) : m_data{NdArray<T, N>(arrs)...} {
            for (auto i = 0; i < N; ++i)
                delta[i] = 1;
            for (auto i = 0; i < N; ++i)
                start[i] = 0;
            scale = std::sqrt(tpa::dot(delta, delta)) / T(N);

            const auto& arr0 = std::get<0>(std::tie(arrs...));
            for (auto i = 0; i < N; ++i)
                m_shape[i] = arr0.shape(i);
            m_size = tpa::prod(m_shape);
        }

        template<typename...Arr>
            requires( sizeof...(Arr) == N + 2 )
        LineTracer(const py::array_t<Arr, py::array::forcecast>&...arrs) {
            auto args = std::tie(arrs...);
            auto args1 = tpa::firstN<N>(args);
            simd::constexpr_for<0, N, 1>([this, &args1](auto I) {
                m_data[I] = NdArray<T, N>(std::get<I>(args1));
            });

            for (auto i = 0; i < N; ++i) {
                delta[i] = std::get<N>(args).at(i);
                start[i] = std::get<N+1>(args).at(i);
            }
            scale = std::sqrt(tpa::dot(delta, delta)) / T(N);

            const auto& arr0 = std::get<0>(args1);
            for (auto i = 0; i < N; ++i)
                m_shape[i] = arr0.shape(i);
            m_size = tpa::prod(m_shape);
        }

        auto eval(std::array<T, N> coord) const {
            std::array<T, N> ret;
            tpa::assign(coord, convert(coord));
            limit_bound(coord);
            check_bounds(coord, m_shape, "In eval, checking float");
            Interp interp(coord);
            for (auto i = 0; i < N; ++i) {
                ret[i] = interp.gather(m_data[i]);
            }
            return ret;
        }

        auto operator()([[maybe_unused]] T s, std::array<T, N> coord) const {
            std::array<T, N> ret = eval(coord);
            T norm = std::max(std::sqrt(tpa::dot(ret, ret)), T(1e-10));
            tpa::assign(ret, ret / norm);
            return ret;
        }

        template<bool neg>
        auto step(std::array<T, N>& coord, TraceConfig<T> cfg) const {
            auto stepper = make_runge_kutta<T, N, 4>(*this);
            auto [result, real_step, err] = stepper.template step_adaptive<neg>(
                    0, coord, cfg.step_size, cfg.tol, cfg.tol_rel, cfg.max_step, cfg.min_step);
            return std::make_pair(result, real_step);
        }

        template<bool neg = false, typename TermFn>
        std::vector<std::array<T, N>> trace(
                std::array<T, N> init,
                TraceConfig<T> cfg, TermFn&& terminate) const {
            std::vector<std::array<T, N>> result;
            result.push_back(init);
            while (not terminate(result.back()) and --cfg.max_iter > 0) {
                auto [res, real_step] = step<neg>(result.back(), cfg);
                bool has_nan = false;
                for (auto r : res) has_nan |= my_isnan(r);
                if (has_nan) break;
                cfg.step_size = real_step;
                result.push_back(res);
                if (cfg.min_dist > 0 and result.size() > 10) {
                    auto d = dist(init, result.back(), result[result.size()-2]);
                    if (d < cfg.min_dist)
                        break;
                }
            }
            return result;
        }

        template<typename TermFn>
        std::vector<std::array<T, N>> bidir_trace(
                std::array<T, N> init,
                TraceConfig<T> cfg, TermFn&& terminate) const {
            auto res_neg = trace<true>(init, cfg, std::forward<TermFn>(terminate));
            auto res_pos = trace<false>(init, cfg, std::forward<TermFn>(terminate));
            size_t len_neg = res_neg.size(),
                   len_pos = res_pos.size();
            size_t length = len_neg + len_pos - 1;
            res_pos.resize(length);
            // move the positive part to the end of the vector
            if (len_neg == 1)
                return res_pos;
            std::move_backward(res_pos.begin(), res_pos.begin() + len_pos, res_pos.end());
            // reverse the negative part and copy them to the beginning of the vector
            std::reverse(res_neg.begin(), res_neg.end());
            std::copy(res_neg.begin(), res_neg.end(), res_pos.begin());
            return res_pos;
        }

        template<typename Real, int dir=0>
        py::array trace_one_py(
                py::array_t<T, py::array::forcecast> init,
                Real step_size, Real tol, Real tol_rel, Real max_step, Real min_step, int max_iter, T min_dist, Real term_val) const {
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
            std::array<T, N> coords;
            for (auto i = 0; i < N; ++i)
                coords[i] = init.at(i);
            if constexpr (dir == 0) {
                auto result = bidir_trace(coords, cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                return to_numpy(result);
            }
            else if constexpr (dir == 1) {
                auto result = trace<false>(coords, cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                return to_numpy(result);
            }
            else if constexpr (dir == -1) {
                auto result = trace<true>(coords, cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                return to_numpy(result);
            }
        }

        template<typename Real>
        py::list trace_many(
                py::array_t<T, py::array::forcecast> inits,
                Real step_size, Real tol, Real tol_rel, Real max_step, Real min_step, int max_iter, T min_dist, Real term_val) const {
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

            std::vector<py::array_t<T>> results(num_points);
#pragma omp parallel for schedule(TRACE_LINE_OMP_SHEDULE) default(shared)
            for (auto i = 0; i < num_points; ++i) {
                int tid = omp_get_thread_num();
                int my_num;
                auto result = bidir_trace(coords[i], cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                //results[i] = to_numpy(result);
                copy_to_numpy(result, results[i]);
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
                Real step_size, Real tol, Real tol_rel, Real max_step, Real min_step, int max_iter, T min_dist, Real term_val) const {
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
#pragma omp parallel for schedule(TRACE_LINE_OMP_SHEDULE) default(shared)
            for (auto i = 0; i < num_points; ++i) {
                auto pos = trace<false>(coords[i], cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                auto neg = trace<true>(coords[i], cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
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

        template<typename Real>
        py::array find_open_close(
                py::array_t<uint8_t, py::array::forcecast> ends_flags,
                py::array_t<uint8_t, py::array::forcecast> trace_flags,
                Real step_size, Real tol, Real tol_rel, Real max_step, Real min_step, int max_iter, T min_dist, Real term_val,
                size_t report_num) const {
            constexpr uint16_t initial_flag = ~uint16_t(0);
            if (ends_flags.ndim() != N)
                throw std::invalid_argument("ends_flags must have dimension " + std::to_string(N));
            const std::array<size_t, N> shape = m_shape;
            for (auto i = 0u; i < N; ++i)
                if (ends_flags.shape(i) != shape[i])
                    throw std::invalid_argument("size of dimension " + std::to_string(i) + " of ends_flags does not match.");
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

            auto trace_flags_arr = trace_flags.unchecked<N>();
            auto flags = ends_flags.unchecked<N>();
            py::array_t<uint16_t> result(shape);
            auto result_arr = result.unchecked<N>();
            TraceGrid<uint16_t, T, N> trace_grid(
                    result.mutable_data(),
                    start, delta, shape);

            auto result_ptr = result.mutable_data();
            for (auto i = 0; i < result.size(); ++i)
                result_ptr[i] = initial_flag;

            auto term_fn = [this, &flags, term_val, &shape, &trace_flags_arr](auto& pos) {
                bool term_zero = terminate(pos, term_val);
                if (term_zero)
                    return true;
                auto ipos = tpa::cast<int>(convert(pos));
                check_bounds(ipos, shape, "In term_fn");
                return std::apply(flags, ipos) > 0 || std::apply(trace_flags_arr, ipos) == 0;
            };

            auto get_flag = [this, &flags](auto& pos) -> uint16_t {
                auto ipos = tpa::cast<int>(convert(pos));
                return std::apply(flags, ipos);
            };

            NdIndices<N> indices(shape-4);
            size_t total_points = indices.size();

            std::vector<size_t> indices_sf(total_points);
            for (size_t i = 0; i < total_points; ++i) indices_sf[i] = i;
            std::shuffle(indices_sf.begin(), indices_sf.end(), std::mt19937(std::random_device()()));

            if (report_num == 0) report_num = total_points;

            std::vector<size_t> traced_num_thread(omp_get_max_threads());
            for (auto& tn : traced_num_thread) tn = 0;

            for (auto seg = 0u; seg < total_points; seg += report_num) {
#pragma omp parallel for schedule(TRACE_LINE_OMP_SHEDULE) default(shared)
                for (int i = seg; i < seg+report_num; ++i) {
                    if (i >= total_points) continue;
                    auto idx = indices.i2idx(indices_sf[i])+2;
                    check_bounds(idx, shape, "Accesing result_arr");
                    if (
                            std::apply(result_arr, idx) != initial_flag or
                            std::apply(trace_flags_arr, idx) == 0 or
                            std::apply(flags, idx) != 0 )
                        continue;

                    int tid = omp_get_thread_num();
                    traced_num_thread[tid] += 1;

                    auto seed = (tpa::cast<T>(idx) + T(0.5)) * delta + start;
                    auto line = bidir_trace(seed, cfg, term_fn);
                    std::array<size_t, 2> skips{0, 0};
                    size_t line_len = line.size();

                    if (line_len < 4)
                        continue;

                    while (not in_bound(convert(line[skips[0]])) and skips[0] < line_len)
                        skips[0] += 1;
                    while (not in_bound(convert(line[line_len-skips[1]-1])) and skips[1] < line_len)
                        skips[1] += 1;

                    if (skips[0] >= line_len or skips[1] >= line_len or skips[0]+skips[1] >= line_len-1)
                        continue;
                    uint16_t result_flag = (get_flag(line[skips[0]]) << 8) | get_flag(line[line_len-skips[1]-1]);
                    trace_grid.set_lines(
                            result_flag,
                            line, skips);
                }
                if (report_num < total_points) {
                    size_t traced_num = 0;
                    for (auto tn : traced_num_thread) traced_num += tn;
                    int print_len = int(std::log10(total_points)+1);
                    printf("%*ld / %*ld, %*ld points set, %*ld lines traced.\n",
                            print_len, std::min(seg+report_num, total_points),
                            print_len, total_points,
                            print_len, trace_grid.get_total(),
                            print_len, traced_num);
                }
            }

            return result;
        }

        bool terminate(std::array<T, N> coord, T term_val) const {
            std::array<T, N> coord_real;
            tpa::assign(coord_real, convert(coord));
            if (not in_bound(coord_real))
                return true;
            auto val = eval(coord);
            T norm = std::sqrt(tpa::dot(val, val));
            return norm <= term_val;
        }

        T dist(std::array<T, N> init, std::array<T, N> p1, std::array<T, N> p2) const {
            auto diff1 = init - p1;
            auto diff2 = init - p2;
            auto diff = p1 - p2;
            auto d_i_p1_2 = tpa::dot(diff1, diff1);
            if (tpa::dot(diff, diff1) < 0 and tpa::dot(diff, diff2) > 0) {
                auto diff_norm = diff / std::max(std::sqrt(tpa::dot(diff, diff)), T(1e-10));
                auto project = std::abs(tpa::dot(diff_norm, diff1));
                return std::sqrt(d_i_p1_2 - project*project);
            }
            return std::sqrt(d_i_p1_2);
        }

    private:
        using data_t = NdArray<T, N>;
        using Interp = picinterp::InterpolatorV<0, N, TRACE_LINE_INTERP_ORDER, T, int>;
        std::array<data_t, N> m_data;
        std::array<T, N> delta, start;
        std::array<size_t, N> m_shape;
        T scale;
        size_t m_size;

        template<typename Pos>
        FORCE_INLINE bool in_bound(const Pos& arr) const {
            constexpr int buf = TRACE_LINE_INTERP_ORDER + 1;
            bool in = true;
            for (auto i = 0; i < N; ++i)
                in = in and arr[i] >= buf and arr[i] < m_shape[i] - buf and not my_isnan(arr[i]);
            return in;
        }

        template<typename P>
        FORCE_INLINE auto limit_bound(std::array<P, N>& arr) const {
            constexpr int buf = TRACE_LINE_INTERP_ORDER + 1;
            for (auto i = 0; i < N; ++i) {
                if (int(arr[i]) < buf) arr[i] = buf;
                if (int(arr[i]) >= m_shape[i]-buf) arr[i] = m_shape[i] - buf - 1;
            }
            return arr;
        }

        FORCE_INLINE auto convert(std::array<T, N> coord) const {
            std::array<T, N> result;
            tpa::assign(result, (coord - start) / delta);
            return result;
        }

        py::array to_numpy(const std::vector<std::array<T, N>>& coords) const {
            std::array<size_t, 2> shape{ coords.size(), N };
            py::array_t<T> result(shape);
            T* ptr = result.mutable_data();
            for (auto i = 0; i < coords.size(); ++i)
                for (auto j = 0; j < N; ++j)
                    ptr[i*N + j] = coords[i][j];
            return result;
        }

        void copy_to_numpy(const std::vector<std::array<T, N>>& coords, py::array_t<T>& out) const {
            std::vector<size_t> shape{coords.size(), N};
            out.resize(shape);
            auto out_uc = out.template mutable_unchecked<2>();
            for (auto i = 0; i < coords.size(); ++i)
                for (auto j = 0; j < N; ++j)
                    out_uc(i, j) = coords[i][j];
        }

};

#define CFG_DEFAULTS \
    py::arg("step_size") = 0.01,  \
    py::arg("tol") = 1e-6,  \
    py::arg("tol_rel") = 1e-6,  \
    py::arg("max_step") = 1,  \
    py::arg("min_step") = 1e-4, \
    py::arg("max_iter") = 5000, \
    py::arg("min_dist") = 0.0, \
    py::arg("term_val") = 1e-2

#define DEFAULTS \
    py::arg("init"),  \
    CFG_DEFAULTS

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
        .def("find_roots", &LineTracer<float, 2>::find_roots<double>, DEFAULTS)
        .def("find_open_close", &LineTracer<float, 2>::find_open_close<float>, py::arg("ends_flags"), py::arg("trace_flags"), CFG_DEFAULTS, py::arg("report_num")=0u)
        .def("find_open_close", &LineTracer<float, 2>::find_open_close<double>, py::arg("ends_flags"), py::arg("trace_flags"), CFG_DEFAULTS, py::arg("report_num")=0u);

    py::class_<LineTracer<float, 3>>(m, "LineTracer3")
        .def(py::init<const arg_t, const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"), py::arg("vz"))
        .def(py::init<const arg_t, const arg_t, const arg_t,
                const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("delta"), py::arg("start"))
        .def("trace", &LineTracer<float, 3>::trace_one_py<float, 0>, DEFAULTS)
        .def("trace", &LineTracer<float, 3>::trace_one_py<double, 0>, DEFAULTS)
        .def("trace_many", &LineTracer<float, 3>::trace_many<float>, DEFAULTS)
        .def("trace_many", &LineTracer<float, 3>::trace_many<double>, DEFAULTS)
        .def("find_roots", &LineTracer<float, 3>::find_roots<float>, DEFAULTS)
        .def("find_roots", &LineTracer<float, 3>::find_roots<double>, DEFAULTS)
        .def("find_open_close", &LineTracer<float, 3>::find_open_close<float>, py::arg("ends_flags"), py::arg("trace_flags"), CFG_DEFAULTS, py::arg("report_num")=0u)
        .def("find_open_close", &LineTracer<float, 3>::find_open_close<double>, py::arg("ends_flags"), py::arg("trace_flags"), CFG_DEFAULTS, py::arg("report_num")=0u)
        ;

}
