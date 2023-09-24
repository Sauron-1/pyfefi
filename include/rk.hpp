#include <pyfefi.hpp>
#include <array>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <tuple_arithmetic.hpp>

using std::size_t;

template<typename T, size_t O> struct RungeKuttaArgs {};

template<typename T> struct RungeKuttaArgs<T, 2> {
    static constexpr std::array<T, 3> c {
        1./2., 3./4., 1.
    };
    static constexpr std::array<T, 4> b {
        2./9., 1./3., 4./9., 0.
    };
    static constexpr std::array<T, 4> bs {
        7./24., 1./4., 1./3., 1./8.
    };
    static constexpr std::array<std::array<T, 3>, 3> a {
        std::array<T, 3>{ 1./2., 0.,    0.   },
        std::array<T, 3>{ 0.,    3./4., 0.   },
        std::array<T, 3>{ 2./9., 1./3., 4./9.}
    };
};

template<typename T> struct RungeKuttaArgs<T, 4> {
    static constexpr std::array<T, 5> c {
        1./4., 3./8., 12./13., 1., 1./2.
    };
    static constexpr std::array<T, 6> b {
        16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.
    };
    static constexpr std::array<T, 6> bs {
        25./216., 0., 1408./2565., 2197./4104., -1./5., 0.
    };
    static constexpr std::array<std::array<T, 5>, 5> a {
        std::array<T, 5>{ 1./4.,        0.,           0.,           0.,           0.},
        std::array<T, 5>{ 3./32.,       9./32.,       0.,           0.,           0.},
        std::array<T, 5>{ 1932./2197., -7200./2197.,  7296./2197.,  0.,           0.},
        std::array<T, 5>{ 439./216.,   -8.,           3680./513.,  -845./4104.,   0.},
        std::array<T, 5>{-8./27.,       2,           -3544./2565.,  1859./4104., -11./40.}
    };
};

template<typename Fn, typename T, size_t N, size_t O>
class RungeKutta {

    public:
        constexpr static size_t dim = N;
        constexpr static size_t order = O;

        FORCE_INLINE RungeKutta(const Fn& fn) : fn(fn) {}

        auto step(T t, std::array<T, N> y, T h) {
            std::array<std::array<T, N>, order+2> ks;
            std::array<T, N> tmp;
            ks[0] = fn(t, y);
            for (auto i = 1; i < order+2; ++i) {
                tpa::assign(tmp, 0);
                for (auto j = 0; j < i; ++j)
                    tpa::assign(tmp, tmp + args.a[i-1][j] * ks[j]);
                tpa::assign(tmp, tmp * h + y);
                tpa::assign(ks[i], fn(t + args.c[i-1] * h, tmp));
            }
            std::array<T, N> result, err;
            tpa::assign(result, 0);
            tpa::assign(err, 0);
            for (auto i = 0;  i < order+2; ++i) {
                tpa::assign(result, result + args.b[i] * ks[i]);
                tpa::assign(err, err + (args.b[i] - args.bs[i]) * ks[i]);
            }
            tpa::assign(result, result * h + y);
            T max_err = 0;
            T max_err_rel = 0;
            for (auto i = 0; i < N; ++i)
                if (fabs(err[i]) > max_err) {
                    max_err = fabs(err[i]);
                    max_err_rel = fabs(err[i]) / std::max(T(fabs(result[i])), T(1e-10));
                }
            return std::make_tuple(result, max_err, max_err_rel);
        }

        template<bool neg = false>
        auto step_adaptive(T t, std::array<T, N> y, T h, T tol, T tol_rel, T max_step, T min_step) {
            const int max_try = 10;
            int i = 0;
            while (true) {
                T h_use = h;
                if constexpr (neg)
                    h_use = -h_use;
                auto [result, err, err_rel] = step(t, y, h_use);
                auto [err_use, tol_use] = get_err_tol(err, err_rel, tol, tol_rel);
                auto [change, new_h] = estimate_step_size(h, err_use, tol_use, max_step, min_step);
                if (not change || ++i >= max_try-1)
                    return std::make_tuple(result, h, err);
                h = new_h;
            }
        }

    private:
        const Fn& fn;
        RungeKuttaArgs<T, order> args;

        /**
         * If err < 0.6*tol or err > tol, use linear approximate to change step
         * so that err is esitmated to 0.8 * tol.
         */
        auto estimate_step_size(T h, T err, T tol, T max_step, T min_step) {
            bool need_change = false;
            if (err < 0.6 * tol and h < max_step)
                need_change = true;
            if (err > tol and h > min_step)
                need_change = true;
            if (need_change) {
                h = h * pow(0.8 * tol / err, 1./(order+1));
                if (h < min_step) h = min_step;
                if (h > max_step) h = max_step;
            }
            return std::make_pair(need_change, h);
        }

        auto estimate_step_size1(T h, T err, T tol, T max_step, T min_step) {
            bool need_change = false;
            if (err < 0.6 * tol and h < max_step) {
                need_change = true;
                h = h * 1.2;
            }
            if (err > tol and h > min_step) {
                need_change = true;
                h = h / 1.2;
            }
            if (h < min_step) h = min_step;
            if (h > max_step) h = max_step;
            return std::make_pair(need_change, h);
        }

        /**
         * Prefer relative tolerance. Use absolute one if:
         * 1. absolute one is satisfied while relative one is not.
         * 2. err_rel too large, i.e. y is close to zero.
         */
        auto get_err_tol(T err, T err_rel, T tol, T tol_rel) {
            if (err < tol and err_rel > tol_rel) {
                return std::make_pair(err, tol);
            }
            if (err_rel / tol_rel > 1e2 * (err / tol)) {
                return std::make_pair(err, tol);
            }
            return std::make_pair(err_rel, tol_rel);
        }

};

template<typename T, size_t N, size_t O, typename Fn>
FORCE_INLINE auto make_runge_kutta(const Fn& fn) {
    return RungeKutta<Fn, T, N, O>(fn);
}
