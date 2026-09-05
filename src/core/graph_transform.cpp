#include "community_detection/core/graph_transform.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cd {
namespace {

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

double uniform_from_edge(const WeightedEdge &edge, std::uint64_t seed) {
    std::uint64_t key = seed;
    key ^= splitmix64((static_cast<std::uint64_t>(edge.u) << 32U) | edge.v);
    const std::uint64_t random = splitmix64(key);
    constexpr long double denominator = static_cast<long double>(1ULL << 53U);
    return static_cast<double>(static_cast<long double>(random >> 11U) / denominator);
}

} // namespace

EdgeDropoutResult drop_edges(const Graph &graph, double rate, std::uint64_t seed) {
    if (!std::isfinite(rate) || rate < 0.0 || rate >= 1.0) {
        throw std::invalid_argument("edge dropout rate must satisfy 0 <= rate < 1");
    }

    std::vector<std::uint64_t> offsets(graph.num_nodes() + 1, 0);
    std::size_t removed = 0;
    graph.for_each_edge([&](const WeightedEdge &edge) {
        if (uniform_from_edge(edge, seed) < rate) {
            ++removed;
            return;
        }
        ++offsets[static_cast<std::size_t>(edge.u) + 1];
        if (edge.u != edge.v) {
            ++offsets[static_cast<std::size_t>(edge.v) + 1];
        }
    });
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        offsets[i] += offsets[i - 1];
    }

    std::vector<NodeId> neighbors(static_cast<std::size_t>(offsets.back()));
    std::vector<double> weights;
    if (graph.is_weighted()) {
        weights.resize(neighbors.size());
    }
    std::vector<std::uint64_t> cursor = offsets;
    graph.for_each_edge([&](const WeightedEdge &edge) {
        if (uniform_from_edge(edge, seed) < rate) {
            return;
        }
        const auto put = [&](NodeId from, NodeId to) {
            const std::size_t position = static_cast<std::size_t>(cursor[from]++);
            neighbors[position] = to;
            if (!weights.empty()) {
                weights[position] = edge.weight;
            }
        };
        put(edge.u, edge.v);
        if (edge.u != edge.v) {
            put(edge.v, edge.u);
        }
    });

    return {Graph::from_csr(graph.num_nodes(), std::move(offsets), std::move(neighbors),
                            std::move(weights)),
            removed};
}

} // namespace cd
