#include <type_traits>
#include <cstdint>
#include <utility>
#include <xsimd/xsimd.hpp>
#include <functional>
#include <cmath>

#if defined(__CUDACC__)
#include <cuda/std/array>
#endif

#pragma once

#ifndef FORCE_INLINE
#  if defined(_MSC_VER)
#    define FORCE_INLINE __forceinline
#  else
#    define FORCE_INLINE [[ gnu::always_inline ]] inline
#  endif
#endif

#define INLINE FORCE_INLINE

namespace simd {

template<int start, int end, int inc, typename Functor>
FORCE_INLINE constexpr void constexpr_for(Functor&& functor) {
    if constexpr (start < end) {
        functor(std::integral_constant<int, start>{});
        constexpr_for<start+inc, end, inc>(std::forward<Functor>(functor));
    }
}   

// Math functions
template<typename T>
requires(std::is_arithmetic_v<T>)
FORCE_INLINE constexpr T rsqrt(const T& val) {
    return T(1) / sqrt(val);
}

template<typename T>
requires(std::is_arithmetic_v<T>)
FORCE_INLINE constexpr T sign(const T& val) {
    return val > 0 ? T(1) : val < 0 ? T(-1) : T(0);
}

template<typename T, typename T1, typename T2>
requires(std::is_arithmetic_v<T> and std::is_convertible_v<T1, T> and std::is_convertible_v<T2, T>)
FORCE_INLINE constexpr T clip(const T& val, T1 lo, T2 hi) {
    return val < lo ? lo : val > hi ? hi : val;
}

/*
 * Array operations
 */
template<size_t N, typename Functor, typename...Ts>
    requires(std::is_invocable_v<Functor, Ts...>)
FORCE_INLINE constexpr auto array_map(Functor&& fn, const std::array<Ts, N>...arrs) {
    using result_scalar = std::invoke_result_t<Functor, Ts...>;
    std::array<result_scalar, N> result;
    for (size_t i = 0; i < N; ++i)
        result[i] = std::invoke(std::forward<Functor>(fn), arrs[i]...);
    return result;
}

template<size_t N, typename T, typename Functor>
    requires(std::is_invocable_v<Functor, T, T>)
FORCE_INLINE constexpr auto array_reduce(Functor&& fn, const std::array<T, N> arr) {
    if constexpr (N == 1)
        return arr[0];
    else {
        auto v0 = std::invoke(std::forward<Functor>(fn), arr[0], arr[1]);
        if constexpr (N > 2)
            for (size_t i = 2; i < N; ++i)
                v0 = std::invoke(std::forward<Functor>(fn), v0, arr[i]);
        return v0;
    }
}

#ifndef __CUDACC__
template<typename T, size_t N>
static constexpr bool has_simd_v = not std::is_void_v<xsimd::make_sized_batch_t<std::remove_cvref_t<T>, N>>;
#else
template<typename T, size_t N>
static constexpr bool has_simd_v = false;
#endif

template<typename T>
static constexpr size_t simd_width_v =
    has_simd_v<T, 16> ? 16 :
    has_simd_v<T,  8> ?  8 :
    has_simd_v<T,  4> ?  4 :
    has_simd_v<T,  2> ?  2 : 1;

struct Empty {
    using batch_bool_type = void;

    struct arch_type {
        static constexpr size_t alignment() { return 0; }
    };
};

template<typename T, typename = void>
struct int_of { using type = int; };
template<typename T>
struct int_of<T, std::enable_if_t<sizeof(T)==1>> { using type = int8_t; };
template<typename T>
struct int_of<T, std::enable_if_t<sizeof(T)==2>> { using type = int16_t; };
template<typename T>
struct int_of<T, std::enable_if_t<sizeof(T)==4>> { using type = int32_t; };
template<typename T>
struct int_of<T, std::enable_if_t<sizeof(T)==8>> { using type = int64_t; };

template<typename T>
using int_of_t = typename int_of<T>::type;

template<typename T, typename = void>
struct uint_of { using type = int; };
template<typename T>
struct uint_of<T, std::enable_if_t<sizeof(T)==1>> { using type = uint8_t; };
template<typename T>
struct uint_of<T, std::enable_if_t<sizeof(T)==2>> { using type = uint16_t; };
template<typename T>
struct uint_of<T, std::enable_if_t<sizeof(T)==4>> { using type = uint32_t; };
template<typename T>
struct uint_of<T, std::enable_if_t<sizeof(T)==8>> { using type = uint64_t; };

template<typename T>
using uint_of_t = typename uint_of<T>::type;

template<typename T, typename = void>
struct float_of { using type = float; };
template<typename T>
struct float_of<T, std::enable_if_t<sizeof(T)==4>> { using type = float; };
template<typename T>
struct float_of<T, std::enable_if_t<sizeof(T)==8>> { using type = double; };

template<typename T>
using float_of_t = typename float_of<T>::type;

template<typename T, size_t N> struct vec_bool;

template<typename T, size_t N>
struct vec {
    static constexpr size_t width = N;
    static constexpr bool is_simd = has_simd_v<T, N>;
    using scalar_t = T;
    using array_t = std::array<scalar_t, width>;
    using batch_t = std::conditional_t<
        is_simd,
        xsimd::make_sized_batch_t<T, N>,
        Empty>;
    using vec_bool_t = vec_bool<T, N>;
    static constexpr size_t align = is_simd ? batch_t::arch_type::alignment() : alignof(T);

    union {
        array_t arr;
        batch_t batch;
    };

    FORCE_INLINE constexpr vec() = default;
    FORCE_INLINE constexpr vec(const batch_t& batch) : batch(batch) {}
    FORCE_INLINE constexpr vec(const array_t& arr) : arr(arr) {}

    FORCE_INLINE constexpr vec(const scalar_t& val) {
        if constexpr (is_simd)
            batch = batch_t(val);
        else
            arr = array_map<width>([val]() { return val; });
    }

    template<typename...Ts>
        requires(sizeof...(Ts) == width)
    FORCE_INLINE constexpr vec(Ts...vals) {
        if constexpr (is_simd)
            batch = batch_t(scalar_t(vals)...);
        else
            arr = array_t{scalar_t(vals)...};
    }

    FORCE_INLINE static constexpr vec loada(const T* ptr) {
        if constexpr (is_simd)
            return batch_t::load_aligned(ptr);
        else {
            vec result;
            for (size_t i = 0; i < N; ++i)
                result.arr[i] = ptr[i];
            return result;
        }
    }
    FORCE_INLINE static constexpr vec loadu(const T* ptr) {
        if constexpr (is_simd)
            return batch_t::load_unaligned(ptr);
        else {
            vec result;
            for (size_t i = 0; i < N; ++i)
                result.arr[i] = ptr[i];
            return result;
        }
    }

    FORCE_INLINE constexpr void storea(T* ptr) {
        if constexpr (is_simd)
            batch.store_aligned(ptr);
        else {
            for (size_t i = 0; i < N; ++i)
                ptr[i] = arr[i];
        }
    }
    FORCE_INLINE constexpr void storeu(T* ptr) {
        if constexpr (is_simd)
            batch.store_unaligned(ptr);
        else {
            for (size_t i = 0; i < N; ++i)
                ptr[i] = arr[i];
        }
    }

    FORCE_INLINE constexpr operator batch_t() {
        return batch;
    }

    FORCE_INLINE constexpr scalar_t operator[](int i) const {
        return arr[i];
    }
    FORCE_INLINE constexpr scalar_t& operator[](int i) {
        return arr[i];
    }


    FORCE_INLINE constexpr vec& operator+=(const vec& other);
    FORCE_INLINE constexpr vec& operator-=(const vec& other);
    FORCE_INLINE constexpr vec& operator*=(const vec& other);
    FORCE_INLINE constexpr vec& operator/=(const vec& other);
    FORCE_INLINE constexpr vec& operator%=(const vec& other);

    FORCE_INLINE constexpr vec& operator&=(const vec& other);
    FORCE_INLINE constexpr vec& operator|=(const vec& other);
    FORCE_INLINE constexpr vec& operator^=(const vec& other);

    FORCE_INLINE constexpr vec& operator>>=(const vec& other);
    FORCE_INLINE constexpr vec& operator<<=(const vec& other);

};
template<typename T, typename A>
FORCE_INLINE auto to_vec(const xsimd::batch<T, A>& b) {
    return vec<T, xsimd::batch<T, A>::size>(b);
}

template<typename T, size_t N>
struct vec_bool {
    static constexpr size_t width = N;
    static constexpr bool is_simd = has_simd_v<T, N>;
    using scalar_t = T;
    using vec_t = vec<T, N>;
    using data_t = std::conditional_t<
        is_simd,
        typename vec_t::batch_t::batch_bool_type,
        std::array<bool, N>>;

    data_t value;

    FORCE_INLINE constexpr vec_bool() = default;
    FORCE_INLINE constexpr vec_bool(const data_t& val) : value(val) {}
    FORCE_INLINE constexpr vec_bool(bool m) {
        if constexpr (is_simd) {
            uint64_t mask = 0;
            if (m) mask = ~mask;
            value = data_t::from_mask(mask);
        }
        else {
            for (auto& val : value) val = m;
        }
    }

    FORCE_INLINE constexpr scalar_t operator[](int i) const {
        if constexpr (is_simd)
            return value.get(i);
        else
            return value[i];
    }

};


/*
 * Unary functions and operations
 */
#define SIMD_MAP_UNARY_FN(FNNAME) \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec<T, N> FNNAME (const vec<T, N>& v) { \
        if constexpr (vec<T, N>::is_simd) \
            return FNNAME (v.batch); \
        else \
            return array_map([](T val) { return FNNAME (val); }, v.arr); \
    }
#define SIMD_MAP_STD_UNARY_FN(FNNAME) \
    using std::FNNAME; \
    SIMD_MAP_UNARY_FN(FNNAME)

SIMD_MAP_UNARY_FN(sign);
SIMD_MAP_UNARY_FN(rsqrt);

SIMD_MAP_STD_UNARY_FN(abs);
SIMD_MAP_STD_UNARY_FN(fabs);

SIMD_MAP_STD_UNARY_FN(exp);
SIMD_MAP_STD_UNARY_FN(exp2);
SIMD_MAP_STD_UNARY_FN(expm1);
SIMD_MAP_STD_UNARY_FN(log);
SIMD_MAP_STD_UNARY_FN(log10);
SIMD_MAP_STD_UNARY_FN(log2);
SIMD_MAP_STD_UNARY_FN(log1p);

SIMD_MAP_STD_UNARY_FN(sqrt);
SIMD_MAP_STD_UNARY_FN(cbrt);

SIMD_MAP_STD_UNARY_FN(sin);
SIMD_MAP_STD_UNARY_FN(cos);
SIMD_MAP_STD_UNARY_FN(tan);
SIMD_MAP_STD_UNARY_FN(asin);
SIMD_MAP_STD_UNARY_FN(acos);
SIMD_MAP_STD_UNARY_FN(atan);

SIMD_MAP_STD_UNARY_FN(sinh);
SIMD_MAP_STD_UNARY_FN(cosh);
SIMD_MAP_STD_UNARY_FN(tanh);
SIMD_MAP_STD_UNARY_FN(asinh);
SIMD_MAP_STD_UNARY_FN(acosh);
SIMD_MAP_STD_UNARY_FN(atanh);

SIMD_MAP_STD_UNARY_FN(erf);
SIMD_MAP_STD_UNARY_FN(erfc);
SIMD_MAP_STD_UNARY_FN(tgamma);
SIMD_MAP_STD_UNARY_FN(lgamma);

SIMD_MAP_STD_UNARY_FN(ceil);
SIMD_MAP_STD_UNARY_FN(floor);
SIMD_MAP_STD_UNARY_FN(trunc);
SIMD_MAP_STD_UNARY_FN(round);
SIMD_MAP_STD_UNARY_FN(nearbyint);
SIMD_MAP_STD_UNARY_FN(rint);

#define SIMD_MAP_UNARY_BOOL_FN(FNNAME) \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec_bool<T, N> FNNAME (const vec<T, N>& v) { \
        if constexpr (vec<T, N>::is_simd) \
            return FNNAME (v.batch); \
        else \
            return array_map([](T val) { return std::FNNAME (val); }, v.arr); \
    }
SIMD_MAP_UNARY_BOOL_FN(isfinite);
SIMD_MAP_UNARY_BOOL_FN(isnan);

#define SIMD_MAP_UNARY_OP(OPNAME, OP) \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec<T, N> OPNAME (const vec<T, N>& v) { \
        if constexpr (vec<T, N>::is_simd) \
            return OP (v.batch); \
        else \
            return array_map([](T val) { return OP (val); }, v.arr); \
    }
SIMD_MAP_UNARY_OP(operator+, +);
SIMD_MAP_UNARY_OP(operator-, -);
SIMD_MAP_UNARY_OP(operator~, ~);

template<typename T, size_t N>
FORCE_INLINE constexpr vec_bool<T, N> operator! (const vec_bool<T, N>& v) {
    if constexpr (vec_bool<T, N>::is_simd)
        return ! (v.value);
    else
        return array_map([](T val) { return ! (val); }, v.value);
}


namespace detail {
    template<size_t I, size_t Idx0, size_t...Idx, typename T>
    FORCE_INLINE constexpr void array_permute_impl(T& tgt, const T& src) {
        tgt[I] = src[Idx0];
        if constexpr (sizeof...(Idx) > 0)
            array_permute_impl<I+1, Idx...>(tgt, src);
    }
}
template<size_t...Idx, typename T, size_t N>
FORCE_INLINE constexpr vec<T, N> permute(const vec<T, N>& v) {
    using vec_t = vec<T, N>;
    if constexpr (vec_t::is_simd) {
        return xsimd::swizzle(v.batch,
                xsimd::batch_constant<uint_of_t<T>,
                    typename vec_t::batch_t::arch_type,
                    (uint_of_t<T>)(Idx)...>{});
    }
    else {
        vec_t result;
        detail::array_permute_impl<0, Idx...>(result, v);
        return result;
    }
}


/*
 * Binary operations and functions
 */
#define SIMD_MAP_BINARY_FN(FNNAME) \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec<T, N> FNNAME (const vec<T, N>& v1, const vec<T, N>& v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return FNNAME (v1.batch, v2.batch); \
        else \
            return array_map([](T val1, T val2) { return std::FNNAME (val1, val2); }, v1.arr, v2.arr); \
    } \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec<T, N> FNNAME (const vec<T, N>& v1, typename vec<T, N>::scalar_t v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return FNNAME (v1.batch, typename vec<T, N>::batch_t(v2)); \
        else \
            return array_map([v2](T val1) { return std::FNNAME (val1, v2); }, v1.arr); \
    } \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec<T, N> FNNAME (typename vec<T, N>::scalar_t v1, const vec<T, N>& v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return FNNAME (typename vec<T, N>::batch_t(v1), v2.batch); \
        else \
            return array_map([v1](T val2) { return std::FNNAME (v1, val2); }, v2.arr); \
    }
SIMD_MAP_BINARY_FN(pow);
SIMD_MAP_BINARY_FN(atan2);

SIMD_MAP_BINARY_FN(fdim);
SIMD_MAP_BINARY_FN(fmin);
SIMD_MAP_BINARY_FN(fmax);
SIMD_MAP_BINARY_FN(min);
SIMD_MAP_BINARY_FN(max);

SIMD_MAP_BINARY_FN(hypot);
SIMD_MAP_BINARY_FN(fmod);
SIMD_MAP_BINARY_FN(remainder);

#define SIMD_MAP_BINARY_OP(OPNAME, OP) \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec<T, N> OPNAME (const vec<T, N>& v1, const vec<T, N>& v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return (v1.batch OP v2.batch); \
        else \
            return array_map([](T val1, T val2) { return (val1 OP val2); }, v1.arr, v2.arr); \
    } \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec<T, N> OPNAME (const vec<T, N>& v1, typename vec<T, N>::scalar_t v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return (v1.batch OP typename vec<T, N>::batch_t(v2)); \
        else \
            return array_map([v2](T val1) { return (val1 OP v2); }, v1.arr); \
    } \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec<T, N> OPNAME (typename vec<T, N>::scalar_t v1, const vec<T, N>& v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return (typename vec<T, N>::batch_t(v1) OP v2.batch); \
        else \
            return array_map([v1](T val2) { return (v1 OP val2); }, v2.arr); \
    }
SIMD_MAP_BINARY_OP(operator+, +);
SIMD_MAP_BINARY_OP(operator-, -);
SIMD_MAP_BINARY_OP(operator*, *);
SIMD_MAP_BINARY_OP(operator/, /);
SIMD_MAP_BINARY_OP(operator%, %);
SIMD_MAP_BINARY_OP(operator>>, >>);
SIMD_MAP_BINARY_OP(operator<<, <<);
SIMD_MAP_BINARY_OP(operator|, |);
SIMD_MAP_BINARY_OP(operator&, &);
SIMD_MAP_BINARY_OP(operator^, ^);

#define SIMD_MAP_BINARY_BOOL_OP(OPNAME, OP) \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec_bool<T, N> OPNAME (const vec<T, N>& v1, const vec<T, N>& v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return (v1.batch OP v2.batch); \
        else \
            return array_map([](T val1, T val2) { return (val1 OP val2); }, v1.arr, v2.arr); \
    } \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec_bool<T, N> OPNAME (const vec<T, N>& v1, typename vec<T, N>::scalar_t v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return (v1.batch OP typename vec<T, N>::batch_t(v2)); \
        else \
            return array_map([v2](T val1) { return (val1 OP v2); }, v1.arr); \
    } \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec_bool<T, N> OPNAME (typename vec<T, N>::scalar_t v1, const vec<T, N>& v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return (typename vec<T, N>::batch_t(v1) OP v2.batch); \
        else \
            return array_map([v1](T val2) { return (v1 OP val2); }, v2.arr); \
    }
SIMD_MAP_BINARY_BOOL_OP(operator>, >);
SIMD_MAP_BINARY_BOOL_OP(operator<, <);
SIMD_MAP_BINARY_BOOL_OP(operator==, ==);
SIMD_MAP_BINARY_BOOL_OP(operator>=, >=);
SIMD_MAP_BINARY_BOOL_OP(operator<=, <=);
SIMD_MAP_BINARY_BOOL_OP(operator!=, !=);

#define SIMD_MAP_BOOL_BINARY_OP(OPNAME, OP) \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec_bool<T, N> OPNAME (const vec_bool<T, N>& v1, const vec_bool<T, N>& v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return (v1.value OP v2.value); \
        else \
            return array_map([](T val1, T val2) { return (val1 OP val2); }, v1.value, v2.value); \
    } \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec_bool<T, N> OPNAME (const vec_bool<T, N>& v1, bool v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return (v1.value OP v2); \
        else \
            return array_map([v2](T val1) { return (val1 OP v2); }, v1.value); \
    } \
    template<typename T, size_t N> \
    FORCE_INLINE constexpr vec_bool<T, N> OPNAME (bool v1, const vec_bool<T, N>& v2) { \
        if constexpr (vec<T, N>::is_simd) \
            return (v1 OP v2.value); \
        else \
            return array_map([v1](T val2) { return (v1 OP val2); }, v2.value); \
    }
SIMD_MAP_BOOL_BINARY_OP(operator&&, &&);
SIMD_MAP_BOOL_BINARY_OP(operator||, ||);
SIMD_MAP_BOOL_BINARY_OP(operator^, ^);


/*
 * Clip
 */
template<typename T, size_t N>
FORCE_INLINE constexpr vec<T, N> clip(const vec<T, N>& v1, const vec<T, N>& v2, const vec<T, N>& v3) {
    if constexpr (vec<T, N>::is_simd)
        return clip(v1.batch, v2.batch, v3.batch);
    else
        return array_map([](T val1, T val2, T val3) {
                return clip(val1, val2, val3);
            }, v1.arr, v2.arr, v3.arr);
}
template<typename T, size_t N>
FORCE_INLINE constexpr vec<T, N> clip(const vec<T, N>& v1, typename vec<T, N>::scalar_t v2, const vec<T, N>& v3) {
    if constexpr (vec<T, N>::is_simd)
        return clip(v1.batch, typename vec<T, N>::batch_t(v2), v3.batch);
    else
        return array_map([v2](T val1, T val3) {
                return clip(val1, v2, val3);
            }, v1.arr, v3.arr);
}
template<typename T, size_t N>
FORCE_INLINE constexpr vec<T, N> clip(const vec<T, N>& v1, const vec<T, N>& v2, typename vec<T, N>::scalar_t v3) {
    if constexpr (vec<T, N>::is_simd)
        return clip(v1.batch, v2.batch, typename vec<T, N>::batch_t(v3));
    else
        return array_map([v3](T val1, T val2) {
                return clip(val1, val2, v3);
            }, v1.arr, v2.arr);
}
template<typename T, size_t N>
FORCE_INLINE constexpr vec<T, N> clip(const vec<T, N>& v1, typename vec<T, N>::scalar_t v2, typename vec<T, N>::scalar_t v3) {
    if constexpr (vec<T, N>::is_simd)
        return clip(v1.batch, typename vec<T, N>::batch_t(v2), typename vec<T, N>::batch_t(v3));
    else
        return array_map([v2, v3](T val1) {
                return clip(val1, v2, v3);
            }, v1.arr);
}

/*
 * Reductions
 */
template<typename T, size_t N, typename Fn>
FORCE_INLINE constexpr T reduce(Fn&& f, const vec<T, N>& x) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::reduce(std::forward<Fn>(f), x.batch);
    else
        return array_reduce(std::forward<Fn>(f), x.arr);
}
template<typename T, size_t N>
FORCE_INLINE constexpr T reduce_add(const vec<T, N>& x) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::reduce_add(x.batch);
    else
        return array_reduce(std::plus<T>{}, x.arr);
}
template<typename T, size_t N>
FORCE_INLINE constexpr T reduce_min(const vec<T, N>& x) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::reduce_min(x.batch);
    else
        return array_reduce([](T v1, T v2) { return std::min(v1, v2); }, x.arr);
}
template<typename T, size_t N>
FORCE_INLINE constexpr T reduce_max(const vec<T, N>& x) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::reduce_max(x.batch);
    else
        return array_reduce([](T v1, T v2) { return std::max(v1, v2); }, x.arr);
}

template<typename T, typename Fn>
    requires(std::is_arithmetic_v<T>)
FORCE_INLINE constexpr T reduce(Fn&&, const T& val) { return val; }

template<typename T>
    requires(std::is_arithmetic_v<T>)
FORCE_INLINE constexpr T reduce_add(const T& val) { return val; }

template<typename T>
    requires(std::is_arithmetic_v<T>)
FORCE_INLINE constexpr T reduce_min(const T& val) { return val; }

template<typename T>
    requires(std::is_arithmetic_v<T>)
FORCE_INLINE constexpr T reduce_max(const T& val) { return val; }

template<typename T, size_t N>
FORCE_INLINE constexpr bool all(const vec_bool<T, N>& x) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::all(x.value);
    else {
        for (bool v : x.value) {
            if (!v)
                return false;
        }
        return true;
    }
}
template<typename T, size_t N>
FORCE_INLINE constexpr bool any(const vec_bool<T, N>& x) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::any(x.value);
    else {
        for (bool v : x.value) {
            if (v)
                return true;
        }
        return false;
    }
}
template<typename T, size_t N>
FORCE_INLINE constexpr bool none(const vec_bool<T, N>& x) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::none(x.value);
    else {
        for (bool v : x.value) {
            if (v)
                return false;
        }
        return true;
    }
}

FORCE_INLINE constexpr bool all(bool b) { return b; }
FORCE_INLINE constexpr bool any(bool b) { return b; }
FORCE_INLINE constexpr bool non(bool b) { return not b; }


/*
 * Select
 */
template<typename T, size_t N>
FORCE_INLINE constexpr vec<T, N> select(const vec_bool<T, N>& cond, const vec<T, N>& true_br, const vec<T, N>& false_br) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::select(cond.value, true_br.batch, false_br.batch);
    else
        return array_map([](bool c, T t, T f) { return c ? t : f; }, cond.value, true_br.arr, false_br.arr);
}
template<typename T, size_t N>
FORCE_INLINE constexpr vec<T, N> select(const vec_bool<T, N>& cond, typename vec<T, N>::scalar_t true_br, const vec<T, N>& false_br) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::select(cond.value, typename vec<T, N>::batch_t(true_br), false_br.batch);
    else
        return array_map([true_br](bool c, T f) { return c ? true_br : f; }, cond.value, false_br.arr);
}
template<typename T, size_t N>
FORCE_INLINE constexpr vec<T, N> select(const vec_bool<T, N>& cond, const vec<T, N>& true_br, typename vec<T, N>::scalar_t false_br) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::select(cond.value, true_br.batch, typename vec<T, N>::batch_t(false_br));
    else
        return array_map([false_br](bool c, T t) { return c ? t : false_br; }, cond.value, true_br.arr);
}
template<typename T, size_t N>
FORCE_INLINE constexpr vec<T, N> select(const vec_bool<T, N>& cond, typename vec<T, N>::scalar_t true_br, typename vec<T, N>::scalar_t false_br) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::select(cond.value, typename vec<T, N>::batch_t(true_br), typename vec<T, N>::batch_t(false_br));
    else
        return array_map([true_br, false_br](bool c) { return c ? true_br : false_br; }, cond.value);
}

template<typename T1, typename T2>
requires(std::is_arithmetic_v<T1> and std::is_arithmetic_v<T2>)
FORCE_INLINE constexpr auto select(bool cond, const T1& t, const T2& f) {
    return cond ? t : f;
}


/*
 * Casting
 */
template<typename To, typename T, size_t N>
FORCE_INLINE constexpr auto cast(const vec<T, N>& v) {
    if constexpr (std::is_same_v<To, bool>)
        return (v == 0);
    else {
        if constexpr (vec<T, N>::is_simd and vec<To, N>::is_simd)
            return vec<To, N>(xsimd::batch_cast<To>(v.batch));
        else
            return vec<To, N>(array_map([](T v) { return To(v); }, v.arr));
    }
}
template<typename To, typename T, size_t N>
FORCE_INLINE constexpr auto cast(const vec_bool<T, N>& v) {
    if constexpr (std::is_same_v<To, bool>)
        return v;
    else {
        if constexpr (vec<T, N>::is_simd) {
            vec<To, N> result;
            for (int i = 0; i < N; ++i)
                result[i] = To(v.value.get(i));
            return result;
        }
        else
            return vec<To, N>(array_map([](bool b) { return To(b); }, v.value));
    }
}
template<typename To, typename T>
requires(std::is_arithmetic_v<T>)
FORCE_INLINE constexpr auto cast(const T& v) { return To(v); }

template<typename T, size_t N>
FORCE_INLINE constexpr vec<int_of_t<T>, N> to_int(const vec<T, N>& v) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::to_int(v.batch);
    else
        return array_map([](T v) { return int_of_t<T>(v); }, v.arr);
}
template<typename T>
requires(std::is_arithmetic_v<T>)
FORCE_INLINE constexpr auto to_int(const T& v) {
    return int_of_t<T>(v);
}

template<typename T, size_t N>
FORCE_INLINE constexpr vec<float_of_t<T>, N> to_float(const vec<T, N>& v) {
    if constexpr (vec<T, N>::is_simd)
        return cast<float_of_t<T>>(v);
    else
        return array_map([](T v) { return float_of_t<T>(v); }, v.arr);
}
template<typename T>
requires(std::is_arithmetic_v<T>)
FORCE_INLINE constexpr auto to_float(const T& v) {
    return float_of_t<T>(v);
}

template<typename T, size_t N>
FORCE_INLINE constexpr vec_bool<xsimd::as_integer_t<T>, N> to_intb(const vec_bool<T, N>& v) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::batch_bool_cast<xsimd::as_integer_t<T>>(v.value);
    else
        return v.value;
}
FORCE_INLINE constexpr bool to_intb(bool b) { return b; }

template<typename T, size_t N>
FORCE_INLINE constexpr vec_bool<xsimd::as_float_t<T>, N> to_floatb(const vec_bool<T, N>& v) {
    if constexpr (vec<T, N>::is_simd)
        return xsimd::batch_bool_cast<xsimd::as_float_t<T>>(v.value);
    else
        return v.value;
}
FORCE_INLINE constexpr bool to_floatb(bool b) { return b; }


/*
 * Assign operators
 */
#define SIMD_DEFINE_MEMBER_ASSIGN( OPNAME, OP ) \
template<typename T, size_t N> \
FORCE_INLINE constexpr vec<T, N>& vec<T, N>::OPNAME (const vec<T, N>& other) { \
    return *this = *this OP other; \
}
SIMD_DEFINE_MEMBER_ASSIGN(operator+=, +);
SIMD_DEFINE_MEMBER_ASSIGN(operator-=, -);
SIMD_DEFINE_MEMBER_ASSIGN(operator*=, *);
SIMD_DEFINE_MEMBER_ASSIGN(operator/=, /);
SIMD_DEFINE_MEMBER_ASSIGN(operator%=, %);

SIMD_DEFINE_MEMBER_ASSIGN(operator&=, &);
SIMD_DEFINE_MEMBER_ASSIGN(operator|=, |);
SIMD_DEFINE_MEMBER_ASSIGN(operator^=, ^);

SIMD_DEFINE_MEMBER_ASSIGN(operator>>=, >>);
SIMD_DEFINE_MEMBER_ASSIGN(operator<<=, <<);

} // namespace simd
