#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cmath>

#include <ndarray.hpp>
#include <type_traits>
#include <utils.hpp>

#pragma once

namespace detail {
    template<typename F1, typename...F> struct max_float {};
    template<typename F1> struct max_float<F1> { using type = F1; };
    template<typename...F> struct max_float<double, F...> { using type = double; };
    template<typename...F> struct max_float<float, F...> { using type = typename max_float<F...>::type; };
    template<typename F1, typename...F> using max_float_t = typename max_float<F1, F...>::type;

    /*
    template<size_t N, size_t dim>
    constexpr auto cartesian_prod() {
        constexpr size_t size = powi<dim>(N);
        std::array<std::array<size_t, dim>, size> result;
        size_t block_len = size;
        for (auto d = 0; d < dim; ++d) {
            block_len /= N;
            auto num_blocks = size / N / block_len;
            auto block_space = size / num_blocks;
            for (auto b = 0; b < num_blocks; ++b) {
                auto start = block_space * b;
                for (auto i = 0; i < N; ++i) {
                    for (auto idx = 0; idx < block_len; ++idx) {
                        auto index = i * block_len + idx + start;
                        result[index][d] = i;
                    }
                }
            }
        }
        return result;
    }
    */

    template<size_t N, size_t dim>
    constexpr auto cartesian_prod_impl() {
        constexpr size_t size = powi<dim>(N);
        std::array<std::array<size_t, dim>, size> result;
        size_t count = 0;
        for (auto i = 0; i < N; ++i) {
            if constexpr (dim == 1)
                result[count++] = {1};
            else {
                auto sub = cartesian_prod_impl<N, dim-1>();
                for (const auto& sub_arr : sub) {
                    result[count][0] = i;
                    for (auto d = 1; d < dim; ++d) {
                        result[count][d] = sub_arr[d-1];
                    }
                    ++count;
                }
            }
        }
        return result;
    }

    template<size_t N, size_t dim>
    static constexpr auto cartesian_prod = cartesian_prod_impl<N, dim>();
}

template<typename T, size_t N>
class FArray : public NdArray<T, N> {

    public:
        using Super = NdArray<T, N>;

        FArray() : Super() {
        }

        FArray(const py::array_t<T> arr) : Super(arr) {
        }

        template<typename...F>
            requires( std::is_floating_point_v<F> || ...)
        INLINE T operator()(F...coords) const {
            return this->operator()(std::array<T, N>{ T(coords)... });
        }

        INLINE T operator()(std::array<T, N> coords) const {
            auto [wt, idx] = to_weight_idx(coords);
            T ret = T(0);
            constexpr auto cp = detail::cartesian_prod<3, N>;
            std::array<size_t, N> indices;
            for (auto iN : cp) {
                T w = T(1);
                for (auto i = 0; i < N; ++i) {
                    w *= wt[i][iN[i]];
                    indices[i] = idx[i][iN[i]];
                }
                ret += Super::get(indices) * w;
            }
            return ret;
        }

        INLINE auto is_out(std::array<T, N> val) const {
            for (auto i = 0; i < N; ++i) {
                if (is_out_one(val[i], shape_[i]))
                    return true;
            }
            return false;
        }

    protected:
        using Super::shape_;

        INLINE auto to_weight_idx(std::array<T, N> coords) const {
            std::array<std::array<T, 3>, N> weights;
            std::array<std::array<int, 3>, N> indices;
            for (auto i = 0; i < N; ++i) {
                auto [wt, idx] = to_weight_idx_one(coords[i], shape_[i]);
                weights[i] = wt;
                indices[i] = idx;
            }
            return std::make_pair(weights, indices);
        }

        INLINE auto to_weight_idx_one(T coord, int size) const {
            int idx = std::round(coord);
            T delta = coord - T(idx);
            std::array<T, 3> weights {
                T(0.5 * (0.5 - delta) * (0.5 - delta)),
                T(0.75  - delta * delta),
                T(0.5 * (0.5 + delta) * (0.5 + delta))
            };
            std::array<int, 3> indices {
                limit(idx-1, size),
                limit(idx, size),
                limit(idx+1, size),
            };
            return std::make_pair(weights, indices);
        }

        INLINE auto limit(int val, int size) const {
            return val < 0
                ? 0
                : val >= size
                    ? size - 1
                    : val;
        }

        INLINE auto is_out_one(T val, int size) const {
            return val < 1 || val >= size-2;
        }

};
