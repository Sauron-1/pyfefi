#if defined(_MSC_VER)
#   define INLINE __forceinline
#else
#   define INLINE inline __attribute__((always_inline))
#   ifndef __GNUC__
#       pragma STDC FP_CONTRACT ON
#   endif
#endif

#include <utility>
#include <type_traits>
#include <functional>

#include <vectorclass/vectorclass.h>

#pragma once


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

template<typename Simd>
struct isimd_type {
    using type = decltype(roundi(std::declval<Simd>()));
};

template<typename Simd>
struct bsimd_type {
    using type = decltype(std::declval<Simd>() == std::declval<Simd>());
};

template<typename Simd>
constexpr static size_t simd_length = sizeof(Simd) / sizeof(std::declval<Simd>()[0]);

template<typename Float>
struct native_simd_type {
    using type = simd_type_t<Float, (INSTRSET >= 9 ? 512 : 256) / sizeof(Float) / 8>;
};

template<typename Float>
using native_simd_t = typename native_simd_type<Float>::type;

template<typename Float, typename Simd>
static auto INLINE convert(Simd i) {
    if constexpr (std::is_same_v<Float, double>) {
        return to_double(i);
    }
    else {
        return to_float(i);
    }
}

#ifndef CONSTEXPR_FOR
#define CONSTEXPR_FOR
template<auto Start, auto End, auto Inc, typename Fn, typename...Args>
    requires( Start <= End or std::invocable<Fn, std::integral_constant<decltype(Start), Start>, Args...> )
INLINE constexpr void constexpr_for(Fn&& fn, Args&&...args) {
    if constexpr (Start < End) {
        std::invoke(std::forward<Fn>(fn), std::integral_constant<decltype(Start), Start>{}, std::forward<Args>(args)...);
        constexpr_for<Start+Inc, End, Inc>(std::forward<Fn>(fn), std::forward<Args>(args)...);
    }
}
#endif
