#include <pyfefi.hpp>
#include <vector>
#include <iostream>
#include <array>
#include <tuple_arithmetic.hpp>
#include <tuple_math.hpp>
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

                m_size = 1;
                for (auto s : shape)
                    m_size *= s;
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

        size_t size() const { return m_size; }

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
            using tpa::operator<;
            using tpa::operator<=;
            s = (s - start) / scale;
            e = (e - start) / scale;
            double zero = 1e-10;

            farr dir = e - s;
            auto len = std::sqrt(tpa::dot(dir, dir));
            if (len <= zero) return;
            dir = dir / len;
            farr cur = s;
            iarr dir_sign = tpa::apply_unary_op(
                    [zero](auto a) { return a < -zero ? -1 : ( a > zero ? 1 : 0 ); }, dir);
            auto dir_is_zero = tpa::abs(dir_sign) <= zero;
            while (tpa::dot(e - cur, dir) > -1) {
                auto idx = to_idx1(cur);
                if (idx >= 0 || idx < m_size)
                    ptr[idx] = val;

                auto next_i = tpa::cast<int>(cur + dir_sign);
                auto t = select(dir_is_zero, tpa::repeat_as(1e20, next_i), (next_i - cur) / dir);
                Float min_t = tpa::reduce(
                        [](auto a, auto b) { return std::min(Float(a), Float(b)); },
                        t);
                cur = cur + dir * min_t;
            }
        }

        void set_lines(T val, std::vector<farr> line, std::array<size_t, 2> skips=std::array<size_t, 2>{0, 0}) {
            for (auto i = skips[0]; i < line.size()-skips[1]-1; ++i) {
                set_line(val, line[i], line[i+1]);
            }
        }

        size_t get_total() const {
            size_t total = 0;
            for (auto i = 0; i < m_size; ++i)
                if (ptr[i] != 0) ++total;
            return total;
        }

    private:
        T *ptr;
        farr start, scale;
        iarr shape, strides;
        size_t m_size;

        template<typename Pos>
        int to_idx1(Pos&& pos) const {
            auto idx = tpa::cast<int>(pos);
            return tpa::dot(idx, strides);
        }

};
