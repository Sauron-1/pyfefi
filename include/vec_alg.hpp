#include <Kokkos_Core.hpp>
#include <type_traits>

namespace small_vector {

#if defined(__cpp_lib_remove_cvref)
    using std::remove_cvref_t;
#else
    template<typename T>
    using remove_cvref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
#endif


template<typename T, size_t N>
class Vec {
    public:
        using Scalar = T;
        static constexpr size_t dim = N;

        T data[N];

        KOKKOS_FORCEINLINE_FUNCTION Vec() = default;

        template<typename Tother, typename = std::enable_if_t<std::is_convertible_v<Tother, T>, bool>>
        KOKKOS_FORCEINLINE_FUNCTION Vec(Tother val) {
            for (size_t i = 0; i < N; ++i)
                data[i] = val;
        }

        template<typename...Ts>
        KOKKOS_FORCEINLINE_FUNCTION Vec(Ts...vs) {
            data = {T(vs)...};
        }

        KOKKOS_FORCEINLINE_FUNCTION Vec& operator=(const Vec& val) = default;

        template<typename Tother, typename = std::enable_if_t<std::is_convertible_v<Tother, T>, bool>>
        KOKKOS_FORCEINLINE_FUNCTION Vec& operator=(Tother val) {
            for (size_t i = 0; i < N; ++i)
                data[i] = val;
            return *this;
        }

        KOKKOS_FORCEINLINE_FUNCTION T& operator()(size_t i) { return data[i]; }
        KOKKOS_FORCEINLINE_FUNCTION const T& operator()(size_t i) const { return data[i]; }
};

template<typename T> struct is_vec : std::false_type {};
template<typename T, size_t N> struct is_vec<Vec<T, N>> : std::true_type {};
template<typename T> constexpr bool is_vec_v = is_vec<T>::value;

template<typename...T> struct TypeList;

template<typename...T> struct contains_vec;
template<> struct contains_vec<> : std::false_type{};
template<typename T1, typename...T> struct contains_vec<T1, T...>:
    std::conditional_t<is_vec_v<T1>, std::true_type, contains_vec<T...>> {};

namespace detail {
    template<typename Op, size_t N, typename Ts, typename Tv> struct vec_invoke_result_impl;

    template<typename Op, size_t N, typename...T>
    struct vec_invoke_result_impl<Op, N, TypeList<T...>, TypeList<>> {
        using type = Vec<remove_cvref_t<std::invoke_result_t<Op, T...>>, N>;
    };

    template<typename Op, size_t N, typename...Ts, typename Tv0, typename...Tv>
    struct vec_invoke_result_impl<Op, N, TypeList<Ts...>, TypeList<Tv0, Tv...>> {
        using type = std::conditional_t<
            is_vec_v<Tv0>,
            std::enable_if_t<
                N==0 or Tv0::dim==N,
                typename vec_invoke_result_impl<Op, Tv0::dim, TypeList<Ts..., typename Tv0::Scalar>, TypeList<Tv...>>::type>,
            typename vec_invoke_result_impl<Op, N, TypeList<Ts..., Tv0>, TypeList<Tv...>>::type>;
    };
}
template<typename Op, typename...T> struct vec_invoke_result;

template<typename To, typename T, size_t N>
std::enable_if_t<std::is_convertible_v<T, To>, Vec<To, N>>
KOKKOS_FORCEINLINE_FUNCTION cast(const Vec<T, N>& vec) {
    Vec<To, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = To(vec[i]);
    return ret;
}

template<typename T, size_t N>
KOKKOS_FORCEINLINE_FUNCTION auto operator-(const Vec<T, N>& vec) {
    Vec<T, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = -vec[i];
    return ret;
}

#define SMALL_VEC_BINARY_OP(OPNAME, OP)

}
