#include "ndarray.hpp"
#include <pybind11/pytypes.h>
#include <vector>
#include <cmath>
#include <iostream>

#include <search_tree.hpp>

template<typename Real>
class MeshCore {

    public:
        MeshCore(
                const py::array_t<Real> x,
                const py::array_t<Real> y,
                const py::array_t<Real> z,
                const py::array_t<Real> _pqw_lims) {
            if (x.ndim() != 3 or y.ndim() != 3 or z.ndim() != 3)
                throw std::runtime_error("x, y, z must be 3 dimensional");
            if (not has_same_shape(x, y, z))
                throw std::runtime_error("x, y, z must have same shape");
            if (not has_same_stride(x, y, z))
                throw std::runtime_error("x, y, z must have same strides");
            std::array<const Real*, 3> coords {
                x.data(), y.data(), z.data() };
            size_t size = x.size();
            auto search_tree = SearchTree<Real>(coords, size);

            for (auto i = 0; i < 3; ++i) {
                for (auto j = 0; j < 2; ++j)
                    pqw_lims[i][j] = _pqw_lims.at(i, j);
                if (x.shape(i) > 1)
                    pqw_delta[i] = (pqw_lims[i][1] - pqw_lims[i][0]) / (x.shape(i) - 1);
                else
                    pqw_delta[i] = 1;
            }

            ax_order = std::array{0, 1, 2};
            for (auto i = 0; i < 3; ++i) {
                for (auto j = i+1; j < 3; ++j) {
                    if (x.strides(ax_order[i]) < x.strides(ax_order[j])) {
                        int tmp = ax_order[i];
                        ax_order[i] = ax_order[j];
                        ax_order[j] = tmp;
                    }
                }
            }

            for (auto i = 0; i < 3; ++i) {
                shape[i] = x.shape(i);
            }

            coords = std::array{ x, y, z };

        }

        INLINE std::array<int, 3> i2idx(int idx) {
            std::array<int, 3> ret;
            for (auto i = 0; i < 3; ++i) {
                int size_of_ax = shape[ax_order[3-i-1]];
                ret[3-i-1] = idx % size_of_ax;
                idx /= size_of_ax;
            }
            return ret;
        }

        INLINE std::array<Real, 3> from_cartesian_one(Real x, Real y, Real z) {
            auto indices = i2idx(search_tree(std::array{x, y, z}));
        }

        py::tuple to_cartesian(py::array_t<Real> p, py::array_t<Real> q, py::array_t<Real> w) const {
        }

    private:
        SearchTree<Real> search_tree;
        std::array<std::array<Real, 2>, 3> pqw_lims;
        std::array<Real, 3> pqw_delta;
        std::array<int, 3> ax_order, shape;
        std::array<py::array_t<Real>, 3> coords;

};
