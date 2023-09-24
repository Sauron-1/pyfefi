#include <pyfefi.hpp>
#include <vector>
#include <iostream>
#include <array>
#include <tuple_arithmetic.hpp>
#include <cmath>

using std::size_t;

template<typename T, typename Float, size_t N>
class TraceGrid {

    public:
        static constexpr size_t dim = N;
        using farr = std::array<Float, N>;
        using sarr = std::array<size_t, N>;
        using iarr = std::array<int, N>;
        using Scalar = T;

        TraceGrid(T *ptr, farr start, farr scale, sarr _shape) :
            ptr(ptr), start(start), scale(scale) {
                shape = tpa::cast<int>(_shape);
                for (auto i = 0; i < N; ++i)
                    start[i] -= scale[i] * 0.5;
                strides[N-1] = 1;
                for (auto i = 1; i < N; ++i)
                    strides[N-i-1] = strides[N-i] * shape[N-i];
            }

        template<typename Pos>
        int to_idx(Pos&& pos) const {
            auto idx = tpa::cast<int>((pos - start) / scale);
            return tpa::dot(idx, strides);
        }

        farr i2pos(int idx) const {
            iarr ipos;
            for (auto i = 0; i < N; ++i) {
                ipos[N-i-1] = idx % shape[N-i-1];
                idx /= shape[N-i-1];
            }
            farr pos = (ipos + 0.5) * scale + start;
            return pos;
        }

        size_t size() const {
            auto ret = 1;
            for (auto s : shape)
                ret *= s;
            return ret;
        }

        template<typename...Idx>
            requires(sizeof...(Idx) == N)
        T& operator()(Idx...idx) {
            return ptr[to_idx(std::make_tuple(idx...))];
        }
        template<typename...Idx>
            requires(sizeof...(Idx) == N)
        T operator()(Idx...idx) const {
            return ptr[to_idx(std::make_tuple(idx...))];
        }

        template<typename Idx>
        T& operator()(std::array<Idx, N> idx) {
            return ptr[to_idx(idx)];
        }
        template<typename Idx>
        T operator()(std::array<Idx, N> idx) const {
            return ptr[to_idx(idx)];
        }

        void set_line(T val, farr s, farr e) {
            s = (s - start) / scale;
            e = (e - start) / scale;

            farr dir = e - s;
            dir = dir / std::sqrt(tpa::dot(dir, dir));
            farr cur = s;
            iarr dir_sign = tpa::apply_unary_op(
                    [](auto a) { return a < 0.0 ? -1 : 1; }, dir);
            auto dir_is_zero = std::abs(dir_sign) < 1e-10;
            while (tpa::dot(e - cur, dir) > 0) {
                ptr[to_idx1(cur)] = val;
                auto next_i = tpa::cast<int>(cur + dir_sign);
                auto t = select(dir_is_zero, 1e20, (next_i - cur) / dir);
                Float min_t = tpa::reduce(
                        [](auto a, auto b) { return std::min(Float(a), Float(b)); },
                        t);
                cur = cur + dir * min_t;
            }
        }

    private:
        T *ptr;
        farr start, scale;
        iarr shape, strides;

        template<typename Pos>
        int to_idx1(Pos&& pos) const {
            auto idx = tpa::cast<int>(pos);
            return tpa::dot(idx, strides);
        }

};
