#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vectorclass/vectorclass.h>

namespace py = pybind11;

template<typename Float, size_t N> struct simd_type {};
template<> struct simd_type<float, 4> { using type = Vec4f; };
template<> struct simd_type<float, 8> { using type = Vec8f; };
template<> struct simd_type<float, 16> { using type = Vec16f; };
template<> struct simd_type<double, 2> { using type = Vec2d; };
template<> struct simd_type<double, 4> { using type = Vec4d; };
template<> struct simd_type<double, 8> { using type = Vec8d; };

template<typename Simd> struct scalar_type {
    using type = decltype(std::declval<Simd>()[0]);
    constexpr static size_t simd_len = sizeof(Simd) / sizeof(type);
};

template<typename T, size_t N>
using simd_type_t = typename simd_type<T, N>::type;

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

        template<int extra_flags>
        NdArray(py::array_t<T, extra_flags>& arr) {
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
            data_ = (Scalar*) arr.mutable_data();
        }

        NdArray(py::array& arr) {
            if (arr.ndim() != N) {
                throw std::runtime_error("Wrong number of dimensions");
            }
            for (size_t i = 0; i < N; ++i) {
                shape_[i] = arr.shape(i);
                strides_[i] = arr.strides(i) / sizeof(Scalar);
            }
            data_ = (Scalar*) arr.mutable_data();
        }

        template<typename...I>
        T& operator()(I...indices) {
            return data_[index(indices...)];
        }

        template<typename...I>
        const T& operator()(I...indices) const {
            return data_[index(indices...)];
        }

        template<typename...I>
        size_t index(I...indices) const {
            static_assert(sizeof...(I) == N, "Wrong number of indices");
            return index_impl(std::array<size_t, N>{size_t(indices)...});
        }

        template<typename...I>
        size_t index_impl(std::array<size_t, N> indices) const {
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
        auto gather(Isimd...indices) const {
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
        auto gather_impl(std::array<Isimd, N> indices) const {
            constexpr size_t simd_len = sizeof(Isimd) / sizeof(decltype(indices[0][0]));
            using simd_type = simd_type_t<Scalar, simd_len>;
            simd_type result;
            std::array<Scalar, simd_len> tmp;
            Isimd idx = 0;
            for (size_t i = 0; i < N; ++i) {
                idx += indices[i] * strides_[i];
            }
            for (size_t i = 0; i < simd_len; ++i) {
                tmp[i] = data_[idx[i]];
            }
            result.load(tmp.data());
            return result;
        }

        size_t shape(size_t i) const {
            return shape_[i];
        }

        size_t size() const {
            return shape_[0] * strides_[0];
        }

    private:
        std::array<size_t, N> shape_;
        std::array<size_t, N> strides_;
        T* data_;
};
