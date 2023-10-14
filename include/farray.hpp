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

template<typename T, size_t N, size_t interp_order=2>
class FArray : public NdArray<T, N> {

    public:
        using Super = NdArray<T, N>;

        FArray() : Super() {
        }

        FArray(const py::array_t<T, py::array::forcecast> arr) : Super(arr) {
        }

        template<typename...F>
            requires( std::is_floating_point_v<F> || ...)
        INLINE T operator()(F...coords) const {
            return this->operator()(std::array<T, N>{ T(coords)... });
        }

        INLINE T operator()(std::array<T, N> coords) const {
            auto [wt, idx] = to_weight_idx(coords);
            T ret = T(0);
            constexpr auto cp = detail::cartesian_prod<interp_order+1, N>;
            std::array<size_t, N> indices;
            for (auto iN : cp) {
                T w = T(1);
                for (auto i = 0; i < N; ++i) {
                    w *= wt[i][iN[i]];
                    indices[i] = idx[i][iN[i]];
                }
                //check_bounds(indices, shape_, "Inside FArray");
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

        INLINE auto& operator()(FArray<T, N, interp_order>& other) {
            Super::operator=(static_cast<Super>(other));
            return *this;
        }

    protected:
        using Super::shape_;

        INLINE auto to_weight_idx(std::array<T, N> coords) const {
            std::array<std::array<T, interp_order+1>, N> weights;
            std::array<std::array<int, interp_order+1>, N> indices;
            for (auto i = 0; i < N; ++i) {
                auto [wt, idx] = to_weight_idx_one(coords[i], shape_[i]);
                weights[i] = wt;
                indices[i] = idx;
            }
            return std::make_pair(weights, indices);
        }

        INLINE auto to_weight_idx_one(T coord, int size) const {
            if constexpr (interp_order == 1) {
                int idx = std::floor(coord);
                T delta = coord - T(idx);
                std::array<T, 2> weights {
                    T(1.0 - delta),
                    T(delta)
                };
                std::array<int, 2> indices {
                    limit(idx, size),
                    limit(idx+1, size),
                };
                return std::make_pair(weights, indices);
            }
            else {
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
        }

        INLINE auto limit(int val, int size) const {
            return val < 0
                ? 0
                : val >= size
                    ? size - 1
                    : val;
        }

        INLINE auto is_out_one(T val, int size) const {
            return my_isnan(val) || val < 1 || val >= size-2;
        }

};
