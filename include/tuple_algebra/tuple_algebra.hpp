/**
 * This header file provides arithmetic operations for tuple-like
 * objects in std namespace, including +, -, *, /, and dot and cross
 * production, sum and prod reduce.
 * Assign operator must be explicitly called, while alg_tuple::alg_tuple
 * provides overloaded operator=.
 *
 * The header requires C++20 to compile.
 */
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <functional>

#pragma once

namespace std {

#ifndef FORCE_INLINE
#   define FORCE_INLINE inline __attribute__((always_inline))
#endif

// Basic concepts and utilities

/**
 * If get<N>(t) is valid for T.
 */
template<typename T, size_t N>
concept has_tuple_element = requires(std::remove_cvref_t<T> t) {
    typename std::tuple_element_t<N, decltype(t)>;
    { get<N>(t) } -> convertible_to<const std::tuple_element_t<N, decltype(t)>&>;
};

/**
 * Concept for tuple-like objects.
 * Requirements:
 * - std::tuple_size<T>::type is integral constant
 * - tuple element type can be accessed via std::tuple_element_t<N, T>
 * - get<N>(t) is valid for N in [0, std::tuple_size_v<T>)
 */
template<typename T>
concept tuple_like = requires(std::remove_cvref_t<T> t) {
    typename std::tuple_size<decltype(t)>::type;
    requires std::derived_from<
        std::tuple_size<decltype(t)>,
        std::integral_constant<std::size_t, std::tuple_size_v<decltype(t)>>
    >;
} && []<std::size_t...N>(std::index_sequence<N...>) {
    return (has_tuple_element<T, N> && ...);
}(std::make_index_sequence<std::tuple_size_v<remove_cvref_t<T>>>{});

/**
 * For-loop for constexpr context.
 */
#ifndef CONSTEXPR_FOR
#define CONSTEXPR_FOR
template<auto Start, auto End, auto Inc, typename Fn, typename...Args>
    requires( Start >= End or invocable<Fn, integral_constant<decltype(Start), Start>, Args...> )
FORCE_INLINE constexpr void constexpr_for(Fn&& fn, Args&&...args) {
    if constexpr (Start < End) {
        std::invoke(std::forward<Fn>(fn), integral_constant<decltype(Start), Start>{}, std::forward<Args>(args)...);
        constexpr_for<Start+Inc, End, Inc>(std::forward<Fn>(fn), std::forward<Args>(args)...);
    }
}
#endif

/**
 * ref_tuple is a tuple-like object that holds lvalue references.
 */
template<typename T>
concept ref_tuple_like = tuple_like<T> &&
    []<std::size_t...N>(std::index_sequence<N...>) {
        return (is_lvalue_reference<tuple_element_t<N, remove_cvref_t<T>>>::value || ...);
    }(std::make_index_sequence<std::tuple_size_v<remove_cvref_t<T>>>{});

/**
 * same_type_tuple is a tuple-like object that satisfies:
 * - tuple size is 1, or
 * - all tuple elements are of the same type
 */
template<typename T>
concept same_type_tuple = tuple_like<T> &&
    (( std::tuple_size_v<std::remove_cvref_t<T>> == 1 ) ||
     []<std::size_t...N>(std::index_sequence<N...>) {
        return (std::is_same_v<tuple_element_t<N, remove_cvref_t<T>>,
                               tuple_element_t<0, remove_cvref_t<T>>> && ...);
     }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{}));

// Const tuple for promotion
// A const_tuple works like a tuple with N same elements.
template<typename T, size_t N>
struct const_tuple {
    T value;
};

template<typename T, size_t N>
struct tuple_size<const_tuple<T, N>> : public std::integral_constant<size_t, N> {};

template<size_t idx, typename T, size_t N>
struct tuple_element<idx, const_tuple<T, N>> {
    using type = T;
};

template<size_t idx, typename T, size_t N>
    requires (idx < N)
FORCE_INLINE constexpr T get(const const_tuple<T, N>& ct) {
    return ct.value;
}

namespace detail {
    /**
     * Get the tuple size of the first tuple-like object in Ts.
     */
    template<typename T, typename...Ts>
    constexpr size_t tuple_size_of() {
        using T_raw = std::remove_cvref_t<T>;
        if constexpr (tuple_like<T_raw>) {
            return std::tuple_size_v<T_raw>;
        }
        else if constexpr (sizeof...(Ts) > 0) {
            return tuple_size_of<Ts...>();
        }
        else {
            return 0;
        }
    }

    /**
     * Promote scalar to tuple (const_tuple<T, N>), N
     * is decided by Trefs.
     */
    template<typename...Tref, typename T>
        requires(std::tuple_like<T> || (... || std::tuple_like<Tref>))
    FORCE_INLINE constexpr decltype(auto) promote(T&& t) {
        using T_raw = std::remove_cvref_t<T>;
        if constexpr (tuple_like<T_raw>) {
            return std::forward<T>(t);
        }
        else {
            return const_tuple<T_raw, tuple_size_of<Tref...>()>{t};
        }
    }
}



// Special tuples

// repeated_tuple: repeat a type N times.
// TODO: may be it should be replaced by const_tuple?
namespace detail {
template<typename T, size_t N, typename...Ts>
struct repeated_tuple_impl {
    using type = typename repeated_tuple_impl<T, N-1, T, Ts...>::type;
};
template<typename T, typename...Ts>
struct repeated_tuple_impl<T, 0, Ts...> {
    using type = std::tuple<Ts...>;
};
}
template<typename T, size_t N>
struct repeated_tuple {
    using type = typename detail::repeated_tuple_impl<T, N>::type;
};
template<typename T, size_t N>
using repeated_tuple_t = typename repeated_tuple<T, N>::type;


// Assignment.
// Assignment statement must be called explicitly because
// we can't overload operator=.
namespace detail {
    // tuple-tuple assignment: same tuple size
    template<typename T1, typename T2>
    concept can_assign_tp_tp = tuple_like<T1> and tuple_like<T2>
        and tuple_size_v<std::remove_reference_t<T1>> == tuple_size_v<std::remove_reference_t<T2>>;

    // tuple-value assignment: a tuple and a scalar
    template<typename T1, typename T2>
    concept can_assign_tp_val = tuple_like<T1> and not tuple_like<T2>;

    // value-value assignment: two scalars with valid operator=
    template<typename T1, typename T2>
    concept can_assign_val_val = not tuple_like<T1> and not tuple_like<T2> and
        requires( T1 v1, T2 v2 ) { v1 = v2; };

    // direct assignment: any two types with valid operator=
    template<typename T1, typename T2>
    concept can_direct_assign = requires (T1 v1, T2 v2) {
        v1 = v2;
    };

    template<typename T1, typename T2>
    concept can_assign =
        can_assign_tp_tp<T1, T2> or
        can_assign_tp_val<T1, T2> or
        can_assign_val_val<T1, T2> or
        can_direct_assign<T1, T2>;
}
/**
 * Assignment operator
 */
template<typename T1, typename T2>
    requires( detail::can_assign<T1, T2> )
FORCE_INLINE constexpr auto assign(T1&& t1, T2&& t2) {
    if constexpr (detail::can_direct_assign<T1, T2>) {
        t1 = t2;
    }
    else if constexpr (detail::can_assign_tp_tp<T1, T2>) {
        constexpr_for<0, tuple_size_v<remove_reference_t<T1>>, 1>( [](auto I, auto&& v1, auto&& v2) {
            assign(get<I>(std::forward<decltype(v1)>(v1)), get<I>(std::forward<decltype(v2)>(v2)));
        }, std::forward<T1>(t1), std::forward<T2>(t2));
    }
    else if constexpr (detail::can_assign_tp_val<T1, T2>) {
        constexpr_for<0, tuple_size_v<remove_reference_t<T1>>, 1>( [](auto I, auto&& v1, auto&& v2) {
            assign(get<I>(std::forward<decltype(v1)>(v1)), std::forward<decltype(v2)>(v2));
        }, std::forward<T1>(t1), std::forward<T2>(t2));
    }
    else {
        t1 = t2;
    }
}


// Casting: construct a all-same-type tuple from a tuple-like object
// by casting each element to the type.
namespace detail {
    template<typename T, tuple_like Tp, size_t...I>
    FORCE_INLINE auto construct_impl(Tp&& tp, std::index_sequence<I...>) {
        return T( std::get<I>(std::forward<Tp>(tp))... );
    }
}
template<typename T, tuple_like Tp>
FORCE_INLINE auto construct(Tp&& tp) {
    return detail::construct_impl<T>(
            std::forward<Tp>(tp),
            std::make_index_sequence<std::tuple_size_v<Tp>> {});
}


// Unary operation
namespace detail {
    template<typename Op, typename Tp, std::size_t...I>
    FORCE_INLINE constexpr auto apply_unary_op(Op&& op, Tp&& tp, std::index_sequence<I...>) {
        return std::make_tuple(op(std::get<I>(tp))...);
    }
}
template<typename Op, tuple_like Tp>
FORCE_INLINE constexpr auto apply_unary_op(Op&& op, Tp&& tp) {
    return detail::apply_unary_op(
            std::forward<Op>(op), std::forward<Tp>(tp),
            std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tp>>::value>{});
}

// (a1, a2, ...) -> (a1[i], a2[i], ...)
template<tuple_like Tp>
FORCE_INLINE constexpr auto index(Tp&& tp, std::ptrdiff_t index) {
    return apply_unary_op(
            [index](auto&& v) { return v[index]; },
            std::forward<Tp>(tp));
}

// (a1, a2, ...) -> (a1(args...), a2(args...), ...)
template<tuple_like Tp, typename...T>
FORCE_INLINE constexpr auto invoke(Tp&& tp, T&&...args) {
    return apply_unary_op(
            [args...](auto&& v) { return v(args...); },
            std::forward<Tp>(tp));
}

// (a1, a2, ...) -> (T(a1), T(a2), ...)
template<typename T, tuple_like Tp>
FORCE_INLINE constexpr auto cast(Tp&& tp) {
    return apply_unary_op(
            [](auto&& v) { return T(v); },
            std::forward<Tp>(tp));
}

// (a1, a2, ...) -> (a1.m, a2.m, ...)
template<typename T, tuple_like Tp, typename Tm>
FORCE_INLINE constexpr auto m_val(Tp&& tp, Tm T::*member) {
    return apply_unary_op(
            [member](auto&& v) { return v.*member; },
            std::forward<Tp>(tp));
}

#define TP_MAKE_UNARY_OP(FN_NAME, EXPR) \
template<std::tuple_like Tp> \
FORCE_INLINE constexpr auto FN_NAME(Tp&& tp) { \
    return std::apply_unary_op( \
            [](auto&& a) { return (EXPR); }, \
            std::forward<Tp>(tp)); \
}

#define TP_MAP_UNARY_FN(FN_NAME) \
    TP_MAKE_UNARY_OP(FN_NAME, FN_NAME(a))

#define TP_MAP_UNARY_STD_FN(FN_NAME) \
    TP_MAKE_UNARY_OP(FN_NAME, std::FN_NAME(a))

#define TP_MAP_METHOD(FN_NAME) \
template<tuple_like Tp, typename...T> \
FORCE_INLINE constexpr auto FN_NAME(Tp&& tp, T&&...args) { \
    return std::apply_unary_op( \
            [args...](auto&& v) { return v.FN_NAME(args...); }, \
            std::forward<Tp>(tp)); \
}

#define TP_MAP_FUNCTION(FN_NAME) \
template<tuple_like Tp, typename...T> \
FORCE_INLINE constexpr auto FN_NAME(Tp&& tp, T&&...args) { \
    return std::apply_unary_op( \
            [args...](auto&& v) { return FN_NAME(v, args...); }, \
            std::forward<Tp>(tp)); \
}


// Binary operation
namespace detail {
    template<typename Op, typename T1, typename T2, std::size_t...I>
    FORCE_INLINE constexpr auto apply_binary_op_impl(Op&& op, T1&& t1, T2&& t2, std::index_sequence<I...>) {
        return std::make_tuple(op(
                    std::get<I>(std::forward<T1>(t1)),
                    std::get<I>(std::forward<T2>(t2)))...);
    }
}
template<typename Op, typename T1, typename T2>
    requires( std::tuple_like<T1> || std::tuple_like<T2> )
FORCE_INLINE constexpr auto apply_binary_op(Op&& op, T1&& v1, T2&& v2) {
    auto tp1 = detail::promote<T2>(std::forward<T1>(v1));
    auto tp2 = detail::promote<T1>(std::forward<T2>(v2));
    return detail::apply_binary_op_impl(
            std::forward<Op>(op), tp1, tp2,
            std::make_index_sequence<tuple_size_v<decltype(tp1)>>{});
}

#define TP_MAKE_BINARY_OP(FN_NAME, EXPR) \
template<typename Tp1, typename Tp2> \
    requires(std::tuple_like<Tp1> || std::tuple_like<Tp2>) \
FORCE_INLINE constexpr auto FN_NAME(Tp1&& tp1, Tp2&& tp2) { \
    return std::apply_binary_op( \
            [](auto&& a, auto&& b) { return (EXPR); }, \
            std::forward<Tp1>(tp1), std::forward<Tp2>(tp2)); \
}

#define TP_MAP_BINARY_FN(FN_NAME) \
    TP_MAKE_BINARY_OP(FN_NAME, FN_NAME(a, b))
#define TP_MAP_BINARY_STD_FN(FN_NAME) \
    TP_MAKE_BINARY_OP(FN_NAME, std::FN_NAME(a, b))


// Ternary operation
namespace detail {
    template<typename Op, typename T1, typename T2, typename T3, std::size_t...I>
    FORCE_INLINE constexpr auto apply_ternary_op_impl(Op&& op, T1&& t1, T2&& t2, T3&& t3, std::index_sequence<I...>) {
        return std::make_tuple(op(std::get<I>(t1), std::get<I>(t2), std::get<I>(t3))...);
    }
}
template<typename Op, typename T1, typename T2, typename T3>
    requires( std::tuple_like<T1> || std::tuple_like<T2> || std::tuple_like<T3> )
FORCE_INLINE constexpr auto apply_ternary_op(Op&& op, T1&& v1, T2&& v2, T3&& v3) {
    decltype(auto) tp1 = detail::promote<T2, T3>(std::forward<T1>(v1));
    decltype(auto) tp2 = detail::promote<T1, T3>(std::forward<T2>(v2));
    decltype(auto) tp3 = detail::promote<T1, T2>(std::forward<T3>(v3));
    return detail::apply_ternary_op_impl(
            std::forward<Op>(op),
            std::forward<decltype(tp1)>(tp1),
            std::forward<decltype(tp2)>(tp2),
            std::forward<decltype(tp3)>(tp3),
            std::make_index_sequence<tuple_size_v<std::remove_cvref_t<decltype(tp1)>>>{});
}

#define TP_MAKE_TERNARY_OP(FN_NAME, EXPR) \
template<typename Tp1, typename Tp2, typename Tp3> \
    requires(std::tuple_like<Tp1> || std::tuple_like<Tp2> || std::tuple_like<Tp3>) \
FORCE_INLINE constexpr auto FN_NAME(Tp1&& tp1, Tp2&& tp2, Tp3&& tp3) { \
    return std::apply_ternary_op( \
            [](auto&& a, auto&& b, auto&& c) { return (EXPR); }, \
            std::forward<Tp1>(tp1), \
            std::forward<Tp2>(tp2), \
            std::forward<Tp3>(tp3)); \
}

#define TP_MAP_TERNARY_FN(FN_NAME) \
    TP_MAKE_TERNARY_OP(FN_NAME, FN_NAME(a, b, c))
#define TP_MAP_TERNARY_STD_FN(FN_NAME) \
    TP_MAKE_TERNARY_OP(FN_NAME, std::FN_NAME(a, b, c))

TP_MAKE_UNARY_OP(operator-, -a);

TP_MAKE_BINARY_OP(operator+, a + b);
TP_MAKE_BINARY_OP(operator-, a - b);
TP_MAKE_BINARY_OP(operator*, a * b);
TP_MAKE_BINARY_OP(operator/, a / b);

TP_MAP_BINARY_STD_FN(min);
TP_MAP_BINARY_STD_FN(max);


TP_MAKE_TERNARY_OP(cond, a ? b : c);

// (a1, a2, ...) -> (a1.m(args...), a2.m(args...), ...)
template<typename T, tuple_like Tp, typename Tm, typename...Targs>
FORCE_INLINE constexpr auto m_fn1(Tp&& tp, Tm T::*member, Targs&&...args) {
    return apply_unary_op(
            [member, args...](auto&& v) { return (v.*member)(args...); },
            std::forward<Tp>(tp));
}
// (a1, a2, ...), (b1, b2, ...) -> (a1.m(b1), a2.m(b2), ...)
template<typename T, tuple_like Tp1, tuple_like Tp2, typename Tm>
FORCE_INLINE constexpr auto m_fn2(Tp1&& tp1, Tm T::*member, Tp2&& tp2) {
    return apply_binary_op(
            [member](auto&& v1, auto&& v2) { return (v1.*member)(v2); },
            std::forward<Tp1>(tp1), std::forward<Tp2>(tp2));
}

// Reduce operation
namespace detail {
    template<typename Op>
    struct ReduceFunction {
        Op m_op;
        ReduceFunction(Op&& op) : m_op(std::move(op)) {}
        template<typename T>
        FORCE_INLINE decltype(auto) operator()(T&& v) {
            return std::forward<T>(v);
        }
        template<typename T1, typename...T>
        FORCE_INLINE auto operator()(T1&& v1, T&&...v) {
            return m_op(std::forward<T1>(v1), operator()(std::forward<T>(v)...));
        }
    };
}

// (a1, a2, ...) -> op(a1, op(a2, ...))
template<typename Op, tuple_like Tp>
    requires( std::tuple_size_v<std::remove_reference_t<Tp>> > 0 )
FORCE_INLINE constexpr auto reduce(Op&& op, Tp&& tp) {
    detail::ReduceFunction<Op> reduce_op(std::forward<Op>(op));
    return std::apply(reduce_op, tp);
}

// (a1, a2, ...) -> a1 + a2 + ...
template<tuple_like Tp>
FORCE_INLINE constexpr auto sum(Tp&& tp) {
    return reduce(
            [](auto&& a, auto&& b) { return a + b; },
            std::forward<Tp>(tp));
}

// (a1, a2, ...) -> a1 * a2 * ...
template<tuple_like Tp>
FORCE_INLINE constexpr auto prod(Tp&& tp) {
    return reduce(
            [](auto&& a, auto&& b) { return a * b; },
            std::forward<Tp>(tp));
}

// (a1, a2, ...) -> a1 || a2, || ...
template<tuple_like Tp>
FORCE_INLINE constexpr auto any(Tp&& tp) {
    return reduce(
            [](auto&& a, auto&& b) { return a || b; },
            std::forward<Tp>(tp));
}

// (a1, a2, ...) -> a1 && a2, && ...
template<tuple_like Tp>
FORCE_INLINE constexpr auto all(Tp&& tp) {
    return reduce(
            [](auto&& a, auto&& b) { return a && b; },
            std::forward<Tp>(tp));
}


// Permutation
template<size_t...Idx, tuple_like Tp>
FORCE_INLINE constexpr auto permute(Tp&& tp) {
    return std::forward_as_tuple(std::get<Idx>(std::forward<Tp>(tp))...);
}


// First N
namespace detail {
template<typename Tp, size_t...I>
FORCE_INLINE constexpr auto firstN_impl(Tp&& tp, std::index_sequence<I...>) {
    return permute<I...>(std::forward<Tp>(tp));
}
}
template<size_t N, tuple_like Tp>
FORCE_INLINE constexpr auto firstN(Tp&& tp) {
    return detail::firstN_impl(std::forward<Tp>(tp), std::make_index_sequence<N>{});
}


// More production

// dot product
template<tuple_like Tp1, tuple_like Tp2>
    requires(tuple_size_v<remove_cvref_t<Tp1>> == tuple_size_v<remove_cvref_t<Tp2>>)
FORCE_INLINE constexpr auto dot(Tp1&& tp1, Tp2&& tp2) {
    return sum(tp1 * tp2);
}

// cross2: (a1, a2) x (b1, b2) = a1 * b2 - a2 * b1
template<tuple_like Tp1, tuple_like Tp2>
    requires(tuple_size_v<remove_cvref_t<Tp1>> == 2 && tuple_size_v<remove_cvref_t<Tp2>> == 2)
FORCE_INLINE constexpr auto cross(Tp1&& tp1, Tp2&& tp2) {
    return std::get<0>(tp1) * std::get<1>(tp2) - 
           std::get<1>(tp1) * std::get<0>(tp2);
}
// cross3: (a1, a2, a3) x (b1, b2, b3) = (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)
template<tuple_like Tp1, tuple_like Tp2>
    requires(tuple_size_v<remove_cvref_t<Tp1>> == 3 && tuple_size_v<remove_cvref_t<Tp2>> == 3)
FORCE_INLINE constexpr auto cross(Tp1&& tp1, Tp2&& tp2) {
    return permute<1, 2, 0>(std::forward<Tp1>(tp1)) * permute<2, 0, 1>(std::forward<Tp2>(tp2)) -
           permute<2, 0, 1>(std::forward<Tp1>(tp1)) * permute<1, 2, 0>(std::forward<Tp2>(tp2));
}


// 3D indexing
template<tuple_like Tp>
    requires(tuple_size_v<remove_cvref_t<Tp>> >= 1)
FORCE_INLINE constexpr decltype(auto) get_x(Tp&& tp) {
    return std::get<0>(std::forward<Tp>(tp));
}
template<tuple_like Tp>
    requires(tuple_size_v<remove_cvref_t<Tp>> >= 2)
FORCE_INLINE constexpr decltype(auto) get_y(Tp&& tp) {
    return std::get<1>(std::forward<Tp>(tp));
}
template<tuple_like Tp>
    requires(tuple_size_v<remove_cvref_t<Tp>> >= 3)
FORCE_INLINE constexpr decltype(auto) get_z(Tp&& tp) {
    return std::get<2>(std::forward<Tp>(tp));
}

}


// AlgTuple: a type behaves same as std::tuple except
// for operator=, which is defined as std::assign (see above)
namespace alg_tuple {

template<typename...Tp>
struct alg_tuple : public std::tuple<Tp...> {
    public:
        using Base = std::tuple<Tp...>;

        using Base::Base;
        using Base::operator=;

        template<std::tuple_like Other>
        FORCE_INLINE void operator=(Other&& other) const {
            std::assign(*this, std::forward<Other>(other));
        }
};

namespace detail {
template<typename T, size_t N, typename...Ts>
struct repeated_tuple_impl {
    using type = typename repeated_tuple_impl<T, N-1, T, Ts...>::type;
};
template<typename T, typename...Ts>
struct repeated_tuple_impl<T, 0, Ts...> {
    using type = alg_tuple<Ts...>;
};
}
template<typename T, size_t N>
struct repeated_tuple {
    using type = typename detail::repeated_tuple_impl<T, N>::type;
};
template<typename T, size_t N>
using repeated_tuple_t = typename repeated_tuple<T, N>::type;

template<typename...Tp>
FORCE_INLINE auto tie(Tp&...val) {
    return alg_tuple<Tp&...>(val...);
}

template<typename...Val>
FORCE_INLINE auto forward_as_tuple(Val&&...val) {
    return alg_tuple<Val...>(std::forward<Val>(val)...);
}

template<typename...Val>
FORCE_INLINE auto make_tuple(Val...val) {
    return alg_tuple<Val...>(val...);
}

}

namespace std {
template<typename...Tp>
struct tuple_size<alg_tuple::alg_tuple<Tp...>> : integral_constant<size_t, sizeof...(Tp)> {};

template<typename...Tp, size_t N>
struct tuple_element<N, alg_tuple::alg_tuple<Tp...>> : tuple_element<N, tuple<Tp...>> {};
}
