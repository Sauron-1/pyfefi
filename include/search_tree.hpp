#include <array>
#include <vector>
#include <tuple>

#include <simd.hpp>

template<typename Real>
FORCE_INLINE auto min_max(Real* a, size_t len) {
    Real min = a[0], max = a[0];
    for (size_t i = 1; i < len; ++i) {
        if (a[i] < min) min = a[i];
        if (a[i] > max) max = a[i];
    }
    return std::make_pair(min, max);
}

template<typename Real>
struct Node {

    int left, right, ax, pivot;
    Real val;

    Node() : left(-1), right(-1), ax(0), pivot(-1), val(0.0) {}

    Node(int ax, int pivot, Real val) :
        left(-1), right(-1), ax(ax), pivot(pivot), val(val) {}

};

template<typename Real>
struct IterInfo {
    int start, end, last_idx, is_left;
    std::array<Real, 3> min, max;
};

template<typename T>
struct SmallStack {
    std::array<T, 64> stack;
    int top;

    SmallStack() : top(-1) {}

    FORCE_INLINE void push(T info) {
        stack[++top] = info;
    }

    FORCE_INLINE T pop() {
        return stack[top--];
    }

    FORCE_INLINE bool empty() {
        return top == -1;
    }

};

template<typename Real>
class SearchTree {

    public:
        using node_t = Node<Real>;

        SearchTree() = default;

        SearchTree(std::array<const Real*, 3> coords, size_t len) {
            node_list.resize(len);
            std::vector<int> idx_list(len);
            for (size_t i = 0; i < len; ++i) idx_list[i] = i;

            std::array<Real, 3> mins, maxs;
            for (int i = 0; i < 3; ++i) {
                auto [mi, ma] = min_max(coords[i], len);
                mins[i] = mi;
                maxs[i] = ma;
            }

            SmallStack<IterInfo<Real>> stack;
            stack.push({0, len, -1, 0, mins, maxs});

            int node_idx = 0;
            while(not stack.empty()) {
                auto info = stack.pop();
                if (info.end - info.start < 2) continue;
                auto [pivot, ax, val] = split_step(idx_list, coords, info.start, info.end, info.min, info.max);
                node_list[node_idx] = Node<Real>(ax, pivot, val);
                if (node_idx > 0) {
                    if (info.is_left) node_list[info.last_idx].left = node_idx;
                    else node_list[info.last_idx].right = node_idx;
                }
                std::array<Real, 3> min1 = info.min, max1 = info.max;
                min1[ax] = val;
                max1[ax] = val;
                IterInfo<Real> info1 = {info.start, pivot, node_idx, 1, mins, max1},
                               info2 = {pivot + 1, info.end, node_idx, 0, min1, maxs};
                if (info1.end - info1.start > info2.end - info2.start) {
                    stack.push(info1);
                    stack.push(info2);
                } else {
                    stack.push(info2);
                    stack.push(info1);
                }
            }
        }

        int operator()(std::array<Real, 3> p) {
            int idx = 0, next_idx = 0;
            while (next_idx != -1) {
                idx = next_idx;
                auto node = node_list[idx];
                if (p[node.ax] < node.val) next_idx = node.left;
                else next_idx = node.right;
            }
            return node_list[idx].pivot;
        }

    private:
        std::vector<node_t> node_list;

        auto split_step(std::vector<int>& idx_list, std::array<const Real*, 3> coords, int start, int end, std::array<Real, 3> mins, std::array<Real, 3> maxs) {
            Real delta = -1;
            int ax = -1;
            for (auto i = 0; i < 3; ++i) {
                auto d = maxs[i] - mins[i];
                if (d > delta) {
                    delta = d;
                    ax = i;
                }
            }
            Real mid = (maxs[ax] + mins[ax]) * 0.5;

            int pi = start, pj = end;
            while (pi < pj) {
                while (pi < pj and coords[ax][idx_list[pi]] < mid) ++pi;
                while (pi < pj and coords[ax][idx_list[pj - 1]] >= mid) --pj;
                if (pi < pj) std::swap(idx_list[pi], idx_list[pj - 1]);
            }
            return std::make_tuple(pi, ax, mid);
        }

};
