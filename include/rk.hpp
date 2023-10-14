#include <pyfefi.hpp>
#include <array>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <tuple_arithmetic.hpp>
#include <tuple_math.hpp>

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
            return std::make_pair(result, tpa::abs(err));
        }

        template<bool neg = false>
        auto step_adaptive(T t, std::array<T, N> y, T h, T tol, T tol_rel, T max_step, T min_step) {
            int max_try = 10;
            while (true) {
                T h_use = h;
                if constexpr (neg)
                    h_use = -h_use;
                auto [result, err] = step(t, y, h_use);
                auto tol_use = tpa::abs(result + y) * (T(0.5)*tol_rel) + tol;
                T err_over_tol = tpa::reduce_max(tpa::max(err, tol_use*T(0.2)) / tol_use);
                auto new_h = get_new_step_size(h, err_over_tol, max_step, min_step);
                if (new_h < h and max_try-- > 0)
                    h = new_h;
                else {
                    return std::make_tuple(result, new_h, err);
                }
            }
        }

    private:
        const Fn& fn;
        RungeKuttaArgs<T, order> args;


        auto get_new_step_size(T h, T err_over_tol, T max_step, T min_step) {
            if ((err_over_tol < 0.6 and h < max_step) or
                (err_over_tol > 1.0 and h > min_step)) {
                h = h * pow(0.8 / std::max(err_over_tol, T(0.2)), 1./(order+1));
                if (h < min_step) h = min_step;
                if (h > max_step) h = max_step;
            }
            return h;
        }


};

template<typename T, size_t N, size_t O, typename Fn>
FORCE_INLINE auto make_runge_kutta(const Fn& fn) {
    return RungeKutta<Fn, T, N, O>(fn);
}
