#include <Kokkos_Core.hpp>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template<typename T>
struct remove_all_pointers { using type = T; };
template<typename T>
struct remove_all_pointers<T*> { using type = typename remove_all_pointers<T>::type; };
template<typename T>
using remove_all_pointers_t = typename remove_all_pointers<T>::type;

template<typename T>
using base_type_t = remove_all_pointers_t<std::remove_all_extents_t<T>>;

template<typename T, typename...I>
auto to_device_data(const std::string& label, base_type_t<T>* data, I...shape) {
    Kokkos::View<T, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(data, shape...);
    Kokkos::View<T> result(label, shape...);
    auto h_result = Kokkos::create_mirror_view(Kokkos::HostSpace{}, result);
    Kokkos::deep_copy(src, h_result);
    Kokkos::deep_copy(h_result, result);
    return result;
}

template<typename Float>
class Lic {

    public:
        Lic(Float *up, Float *vp, Float *tp,
               Float *kp, Float *rp, size_t nx, size_t ny, size_t nk) : nx(nx), ny(ny), nk(nk) {
            u = to_device_data<Float**>("u", up, nx, ny);
            v = to_device_data<Float**>("u", vp, nx, ny);
            u = to_device_data<Float**>("u", tp, nx, ny);
            result = to_device_data<Float**>("result", rp, nx, ny);
            kernel = to_device_data<Float*>("kernel", kp, nk);
        }

        template<int sign, bool acc_only>
        KOKKOS_INLINE_FUNCTION bool advance(int pos[2], Float idx[2], int kidx, Float& res) {
            Float wx[2] = {1-pos[0], pos[0]},
                  wy[2] = {1-pos[1], pos[1]};
            Float vx = u(idx[0]+0, idx[1]+0)*wx[0]*wy[0] +
                       u(idx[0]+1, idx[1]+0)*wx[1]*wy[0] +
                       u(idx[0]+0, idx[1]+1)*wx[0]*wy[1] +
                       u(idx[0]+1, idx[1]+1)*wx[1]*wy[1];
            Float vy = v(idx[0]+0, idx[1]+0)*wx[0]*wy[0] +
                       v(idx[0]+1, idx[1]+0)*wx[1]*wy[0] +
                       v(idx[0]+0, idx[1]+1)*wx[0]*wy[1] +
                       v(idx[0]+1, idx[1]+1)*wx[1]*wy[1];
            vx *= sign;
            vy *= sign;
            if constexpr (not acc_only) {
                Float res1 = texture(idx[0]+0, idx[1]+0)*wx[0]*wy[0] +
                             texture(idx[0]+1, idx[1]+0)*wx[1]*wy[0] +
                             texture(idx[0]+0, idx[1]+1)*wx[0]*wy[1] +
                             texture(idx[0]+1, idx[1]+1)*wx[1]*wy[1];
                res += res1 * kernel(kidx);
            }

            Float bx = vx > 0 ? 1 : 0,
                  by = vy > 0 ? 1 : 0;
            Float tx = Kokkos::abs(vx) > 1e-20 ? (bx - pos[0]) / vx : 1e21,
                  ty = Kokkos::abs(vy) > 1e-20 ? (by - pos[1]) / vy : 1e21;
            if (tx < ty) {
                idx[0] += vx > 0 ? 1 : -1;
                pos[0] = 1 - bx;
                pos[1] += ty * vy;
            }
            else {
                idx[1] += vy > 0 ? 1 : -1;
                pos[0] += tx * vx;
                pos[1] = 1 - by;
            }

            return (idx[0] >= 0 and idx[0] < nx) and
                   (idx[1] >= 0 and idx[1] < ny);
        }

        KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const {
            int kmid = nk / 2;

            int idx[2] = {i, j};
            Float pos[2] = {0, 0};
            Float res = 0;
            for (int kidx = kmid; kidx < nk; ++kidx)
                if (not advance<1, false>(pos, idx, kidx, res))
                    break;

            idx[0] = i; idx[1] = j;
            pos[0] = 0; pos[1] = 0;
            advance<-1, true>(pos, idx, kmid, res);
            for (int kidx = kmid-1; kidx >= 0; --kidx)
                if (not advance<-1, false>(pos, idx, kidx, res))
                    break;

            result(i, j) = res;
        }

    private:
        Kokkos::View<const Float**, Kokkos::MemoryTraits<Kokkos::RandomAccess|Kokkos::Restrict>> u, v, texture;
        Kokkos::View<const Float*, Kokkos::MemoryTraits<Kokkos::Restrict>> kernel;
        Kokkos::View<Float**, Kokkos::MemoryTraits<Kokkos::Restrict>> result;

        size_t nx, ny, nk;
};

template<typename Float>
py::array lic(
        py::array_t<Float> u,
        py::array_t<Float> v,
        py::array_t<Float> texture,
        py::array_t<Float> kernel) {
    if (u.ndim() != 2 || v.ndim() != 2 || texture.ndim() != 2 || kernel.ndim() != 1) {
        throw std::runtime_error("Wrong dimensions!");
    }
    std::array<size_t, 2> shape;
    for (auto i = 0; i < 2; ++i) shape[i] = u.shape(i);
    auto result = py::array_t<Float>(shape);

    Lic<Float> lic(
            u.mutable_data(),
            v.mutable_data(),
            texture.mutable_data(),
            kernel.mutable_data(),
            result.mutable_data(),
            shape[0], shape[1], kernel.shape(0));

    Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {shape[0], shape[1]}),
            lic);

    return result;
}

void add_to_module(py::module_ &m) {
    const char* doc_str = \
        "2D line integral convolution\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "u : 2D ndarray\n"
        "    Vector's x component\n"
        "v : 2D ndarray\n"
        "    Vector's y component\n"
        "texture : 2D ndarray\n"
        "    Texture array\n"
        "kernel : 1D ndarray\n"
        "    Kernal for convolution\n"
        "\n"
        "Returns\n"
        "-------\n"
        "2D ndarray with the same shape as u, v, and texture\n";

    m.def_submodule("lic", "Line integral convolution")
        .def("lic", &lic<float>, doc_str)
        .def("lic", &lic<double>, doc_str);
}
