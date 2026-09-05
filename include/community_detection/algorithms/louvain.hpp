#pragma once

#include "community_detection/core/graph.hpp"
#include "community_detection/core/result.hpp"

#include <cstddef>
#include <cstdint>

namespace cd::louvain {

/** @brief Louvain parameters. */
struct Config {
    /** Resolution gamma in Q_gamma; gamma=1 gives standard modularity. */
    double resolution{1.0};

    /** Maximum number of aggregation levels. */
    std::size_t max_levels{100};

    /** Maximum node sweeps at one level. */
    std::size_t max_passes_per_level{100};

    /** Minimum modularity gain required to accept a new level. */
    double min_modularity_gain{1e-12};

    /** Whether to shuffle nodes before each sweep. */
    bool shuffle_nodes{true};

    /** Seed for randomized traversal order. */
    std::uint64_t seed{42};
};

/**
 * @brief Run Louvain using local-moving and aggregation phases.
 *
 * Input graphs may be unweighted, while the internal representation supports
 * weighted edges and self-loops to preserve aggregated graphs exactly. Every
 * partition level over original nodes is stored in hierarchy.
 */
[[nodiscard]] AlgorithmResult run(const Graph &graph, const Config &config = {});

} // namespace cd::louvain
