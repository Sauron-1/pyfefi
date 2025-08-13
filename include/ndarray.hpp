#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <type_traits>

#include <iostream>

#include "simd.hpp"

#pragma once

namespace py = pybind11;

/**
 * @brief A class for N-dimensional arrays.
 *
 * Works like pybind11::array_t, but without bounds checking.
 */
template<typename T, size_t N>
class NdArray {

    public:
        using Scalar = T;
        static constexpr size_t ndim = N;

        NdArray() {
            data_ = nullptr;
            for (auto i = 0; i < ndim; ++i) {
                shape_[i] = 0;
                strides_[i] = 0;
            }
        }

        template<int extra_flags>
        NdArray(py::array_t<T, extra_flags>& arr) {
            setup_shape_stride(arr);
            if (arr.writeable())
                data_ = (Scalar*) arr.mutable_data();
            else
                data_ = (Scalar*) arr.data();
            allocated_data_ = false;
        }

        template<int extra_flags>
        NdArray(const py::array_t<T, extra_flags>& arr) {
            setup_shape_stride(arr);
            data_ = (Scalar*) arr.data();
            allocated_data_ = false;
        }

        NdArray(py::array& arr) {
            setup_shape_stride(arr);
            if (arr.writeable())
                data_ = (Scalar*) arr.mutable_data();
            else
                data_ = (Scalar*) arr.data();
            allocated_data_ = false;
        }

        NdArray(const py::array& arr) {
            setup_shape_stride(arr);
            data_ = (Scalar*) arr.data();
            allocated_data_ = false;
        }

        template<typename Shape, typename Strides>
        NdArray(T* data, Shape&& sp, Strides&& st) {
            allocated_data_ = false;
            data_ = data;
            for (auto i = 0; i < N; ++i) {
                shape_[i] = sp[i];
                strides_[i] = st[i] / sizeof(Scalar);
            }
        }

        NdArray(const std::array<size_t, N>& shape) : shape_(shape) {
            size_t size = 1;
            for (auto i = 0; i < N; ++i) {
                size *= shape_[i];
            }
            data_ = new Scalar[size];
            allocated_data_ = true;
            strides_[N - 1] = 1;
            for (auto i = 1; i < N; ++i) {
                strides_[N - i - 1] = strides_[N - i] * shape_[N - i];
            }
        }

        NdArray(const NdArray<T, N>& other) {
            data_ = other.data();
            shape_ = other.shape();
            strides_ = other.strides();
            allocated_data_ = false;
        }

        auto& operator=(const NdArray<T, N>& other) {
            data_ = other.data();
            shape_ = other.shape();
            strides_ = other.strides();
            allocated_data_ = false;
            return *this;
        }

        ~NdArray() {
            if (allocated_data_) {
                delete[] data_;
            }
        }

        template<typename...I>
            requires( std::is_integral_v<I> && ...)
        FORCE_INLINE T& operator()(I...indices) {
            return data_[index(indices...)];
        }

        template<typename...I>
            requires( std::is_integral_v<I> && ...)
        FORCE_INLINE const T& operator()(I...indices) const {
            return data_[index(indices...)];
        }

        FORCE_INLINE T& get(std::array<size_t, N> arr) {
            return data_[index_impl(arr)];
        }
        FORCE_INLINE const T& get(std::array<size_t, N> arr) const {
            return data_[index_impl(arr)];
        }

        template<typename...I>
            requires( std::is_integral_v<I> && ...)
        FORCE_INLINE size_t index(I...indices) const {
            static_assert(sizeof...(I) == N, "Wrong number of indices");
            return index_impl(std::array<size_t, N>{size_t(indices)...});
        }

        FORCE_INLINE size_t index_impl(std::array<size_t, N> indices) const {
#if defined(BOUNDSCHECK)
            check_bounds(indices);
#endif
            size_t idx = 0;
            for (size_t i = 0; i < N; ++i) {
                idx += indices[i] * strides_[i];
            }
            return idx;
        }

        /**
         * SIMD version of operator().
         */
        template<typename...Isimd>
        FORCE_INLINE auto gather(Isimd...indices) const {
            static_assert(sizeof...(Isimd) == N, "Wrong number of indices");
            using first_type = std::tuple_element_t<0, std::tuple<Isimd...>>;
            // Check that all indices are convertible to first_type
            static_assert((std::is_convertible_v<Isimd, first_type> && ...),
                    "All indices must be convertible to the first index type");
            return gather_impl(std::array<first_type, N>{first_type(indices)...});
        }

        /**
         * SIMD version of operator().
         */
        template<typename Isimd>
        FORCE_INLINE auto gather_impl(std::array<Isimd, N> indices) const {
#if defined(BOUNDSCHECK)
            check_bounds(indices);
#endif
            constexpr size_t simd_len = sizeof(Isimd) / sizeof(decltype(indices[0][0]));
            using simd_type = simd::vec<Scalar, simd_len>;
            simd_type result;
            Isimd idx = 0;
            for (size_t i = 0; i < N; ++i) {
                idx += indices[i] * strides_[i];
            }
            for (size_t i = 0; i < simd_len; ++i) {
                result[i] = data_[idx[i]];
            }
            return result;
        }

        FORCE_INLINE const auto& shape() const {
            return shape_;
        }
        FORCE_INLINE size_t shape(size_t i) const {
            return shape_[i];
        }
        FORCE_INLINE const auto& strides() const { return strides_; }
        FORCE_INLINE auto data() const { return data_; }

        FORCE_INLINE size_t size() const {
            return shape_[0] * strides_[0];
        }

        template<int op_flag = 0, typename Simd, typename...Isimd>
        FORCE_INLINE auto scatter(Simd values, Isimd...indices) {
            static_assert(sizeof...(Isimd) == N, "Wrong number of indices");
            using first_type = std::tuple_element_t<0, std::tuple<Isimd...>>;
            static_assert((std::is_convertible_v<Isimd, first_type> && ...),
                    "All indices must be convertible to the first index type");
            return scatter_impl<op_flag>(values, std::array<first_type, N>{first_type(indices)...});
        }

        template<int op_flag, typename Simd, typename Isimd>
        FORCE_INLINE auto scatter_impl(Simd values, std::array<Isimd, N> indices) {
#if defined(BOUNDSCHECK)
            check_bounds(indices);
#endif
            constexpr size_t simd_len = sizeof(Isimd) / sizeof(decltype(indices[0][0]));
            Isimd idx = 0;
            for (size_t i = 0; i < N; ++i) {
                idx += indices[i] * strides_[i];
            }
            for (size_t i = 0; i < simd_len; ++i) {
                if constexpr (op_flag == 0)
                    data_[idx[i]] = values[i];
                else if constexpr (op_flag == 1)
                    data_[idx[i]] += values[i];
            }
        }

        template<int op_flag = 0, typename Simd, typename Bsimd, typename...Isimd>
        FORCE_INLINE auto scatterm(Simd values, Bsimd msk, Isimd...indices) {
            static_assert(sizeof...(Isimd) == N, "Wrong number of indices");
            using first_type = std::tuple_element_t<0, std::tuple<Isimd...>>;
            static_assert((std::is_convertible_v<Isimd, first_type> && ...),
                    "All indices must be convertible to the first index type");
            return scatterm_impl<op_flag>(values, msk, std::array<first_type, N>{first_type(indices)...});
        }

        template<int op_flag, typename Simd, typename Bsimd, typename Isimd>
        FORCE_INLINE auto scatterm_impl(Simd values, Bsimd msk, std::array<Isimd, N> indices) {
#if defined(BOUNDSCHECK)
            check_bounds(indices);
#endif
            constexpr size_t simd_len = sizeof(Isimd) / sizeof(decltype(indices[0][0]));
            Isimd idx = 0;
            for (size_t i = 0; i < N; ++i) {
                idx += indices[i] * strides_[i];
            }
            for (size_t i = 0; i < simd_len; ++i) {
                if (msk[i]) {
                    if constexpr (op_flag == 0)
                        data_[idx[i]] = values[i];
                    else if constexpr (op_flag == 1)
                        data_[idx[i]] += values[i];
                }
            }
        }

    protected:
        std::array<size_t, N> shape_;
        std::array<size_t, N> strides_;
        bool allocated_data_;
        T* data_;

        void check_bounds(std::array<size_t, N> idx) const {
            for (size_t i = 0; i < N; ++i) {
                if (idx[i] >= shape_[i] || idx[i] < 0) [[unlikely]] {
                    std::cerr << "Index " << idx[i] << " out of bounds for dimension " << i << " (shape = " << shape_[i] << ") when accessing with scalar" << std::endl;
                    exit(1);
                }
            }
        }

        template<typename iSimd>
        void check_bounds(std::array<iSimd, N> idx) const {
            constexpr size_t simd_len = iSimd::width;
            for (size_t i = 0; i < N; ++i) {
                auto msk = idx[i] >= shape_[i] || idx[i] < 0;
                if (horizontal_or(msk)) [[unlikely]] {
                    for (size_t j = 0; j < simd_len; ++j) {
                        if (msk[j]) {
                            std::cerr << "Index " << idx[i][j] << " out of bounds for dimension " << i << " (shape = " << shape_[i] << ") when accessing with vector" << std::endl;
                            exit(1);
                        }
                    }
                }
            }
        }
        
        template<typename Arr>
        void setup_shape_stride(const Arr& arr) {
            size_t arr_dim = arr.ndim();
            for (size_t i = 0; i < N; ++i) {
                if (i < arr_dim) {
                    shape_[i] = arr.shape(i);
                    strides_[i] = arr.strides(i) / sizeof(Scalar);
                }
                else {
                    shape_[i] = 1;
                    strides_[i] = 0;
                }
            }
        }
};

template<size_t N>
class NdIndices {

    public:
        FORCE_INLINE NdIndices(std::array<size_t, N> shape): shape_(shape) {
            size_ = 1;
            index_ = 0;
            for (auto i = 0; i < N; ++i) {
                state_[i] = 0;
                size_ *= shape_[i];
            }
        }

        FORCE_INLINE size_t size() const {
            return size_;
        }

        FORCE_INLINE std::array<size_t, N> next() {
            index_ += 1;
            std::array<size_t, N> ret = state_;
            for (auto i = 0; i < N; ++i) {
                auto idx = N - i - 1;
                auto val = state_[idx] + 1;
                if (val < shape_[idx]) {
                    state_[idx] = val;
                    return ret;
                }
                state_[idx] = 0;
            }
            return std::array<size_t, N>{0};
        }

        template<typename iSimd>
        FORCE_INLINE auto next_batch() {
            constexpr size_t simd_len = iSimd::width;
            using Int = decltype(std::declval<iSimd>()[0]);
            std::array<std::array<Int, simd_len>, N> buf;
            std::array<Int, simd_len> msk_buf;
            for (auto i = 0; i < simd_len; ++i) {
                msk_buf[i] = index_ < size_;
                if (msk_buf[i]) {
                    auto res = next();
                    for (auto j = 0; j < N; ++j) {
                        buf[j][i] = res[j];
                    }
                }
                else {
                    for (auto j = 0; j < N; ++j) buf[j][i] = 0;
                }
            }

            std::array<iSimd, N> result;
            iSimd msk;
            msk.load(msk_buf.data());
            for (auto i = 0; i < N; ++i) result[i].load(buf[i].data());
            return std::make_pair(msk != 0, result);
        }

        FORCE_INLINE bool has_next() {
            return index_ < size_;
        }

        template<typename iSimd>
        FORCE_INLINE bool has_next_full() {
            constexpr size_t simd_len = iSimd::width;
            return index_ + simd_len < size_;
        }

        FORCE_INLINE auto i2idx(size_t i) {
            std::array<size_t, N> ret;
            for (auto j = 0; j < N; ++j) {
                auto idx = N - j - 1;
                ret[idx] = i % shape_[idx];
                i /= shape_[idx];
            }
            return ret;
        }

        template<typename iSimd>
            requires(not std::is_scalar_v<iSimd>)
        FORCE_INLINE auto i2idx(iSimd i) {
            std::array<iSimd, N> ret;
            for (auto j = 0; j < N; ++j) {
                auto idx = N - j - 1;
                auto [div, mod] = div_mod(i, shape_[idx]);
                ret[idx] = mod;
                i = div;
            }
            return ret;
        }

    protected:
        std::array<size_t, N> shape_;
        std::array<size_t, N> state_;
        size_t size_, index_;

        template<typename iSimd,  typename OpType>
        FORCE_INLINE auto div_mod(iSimd i, OpType op) {
            using Int = decltype(i[0]);
            constexpr size_t simd_len = iSimd::width;
            iSimd div, mod;
            div = i / op;
            mod = i % op;
            return std::make_pair(div, mod);
        }
};


template<typename T, int flags>
static inline std::vector<size_t> get_shape(const py::array_t<T, flags> arr) {
    size_t ndim = arr.ndim();
    std::vector<size_t> shape(ndim);
    for (auto i = 0; i < ndim; ++i) {
        shape[i] = arr.shape(i);
    }
    return shape;
}

template<typename T, int flags>
static inline std::vector<size_t> get_strides(const py::array_t<T, flags> arr) {
    size_t ndim = arr.ndim();
    std::vector<size_t> strides(ndim);
    for (auto i = 0; i < ndim; ++i) {
        strides[i] = arr.strides(i);
    }
    return strides;
}

template<typename T, int flags>
static inline size_t get_min_stride(const py::array_t<T, flags> arr) {
    size_t ndim = arr.ndim();
    size_t stride = sizeof(T);
    for (auto i = 0; i < ndim; ++i) {
        auto s = arr.strides(i);
        if (s < stride) stride = s;
    }
    return stride / sizeof(T);
}

template<typename Arr0, typename...Arr>
static inline bool has_same_dim(const Arr0& arr0, const Arr&...arr) {
    size_t ndim = arr0.ndim();
    return ((arr.ndim() == ndim) && ...);
}

template<typename Arr0, typename...Arr>
static inline bool has_same_shape(const Arr0& arr0, const Arr&...arr) {
    if (!has_same_dim(arr...)) return false;
    size_t ndim = arr0.ndim();
    for (auto i = 0; i < ndim; ++i) {
        if (((arr.shape(i) != arr0.shape(i)) || ...)) return false;
    }
    return true;
}

template<typename Arr0, typename...Arr>
static inline bool has_same_stride(const Arr0& arr0, const Arr&...arr) {
    if (!has_same_dim(arr...)) return false;
    size_t ndim = arr0.ndim();
    for (auto i = 0; i < ndim; ++i) {
        if (((arr.strides(i) != arr0.strides(i)) || ...)) return false;
    }
    return true;
}
