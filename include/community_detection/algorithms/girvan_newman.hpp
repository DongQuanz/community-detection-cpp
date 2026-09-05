#pragma once

#include "community_detection/core/graph.hpp"
#include "community_detection/core/result.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>

namespace cd::girvan_newman {

/** @brief Tie-breaking strategy for edges with maximum betweenness. */
enum class TieBreak {
    Lexicographic, ///< Choose the lexicographically smallest (u,v); deterministic.
    Random         ///< Choose among tied edges using the configured seed.
};

/** @brief Girvan--Newman parameters. */
struct Config {
    /**
     * Target number of communities. Zero selects the hierarchy level with
     * maximum modularity instead of stopping at a predefined K.
     */
    std::size_t target_communities{0};

    /** Maximum edge removals; effectively unlimited by default. */
    std::size_t max_edge_removals{std::numeric_limits<std::size_t>::max()};

    /** Tie-breaking rule for maximum-betweenness edges. */
    TieBreak tie_break{TieBreak::Lexicographic};

    /** Seed used when tie_break == Random. */
    std::uint64_t seed{42};

    /** Tolerance used to consider betweenness values tied. */
    double tie_tolerance{1e-12};
};

using Edge = std::pair<NodeId, NodeId>;
using EdgeBetweenness = std::map<Edge, double>;

/**
 * @brief Compute edge betweenness with Brandes' unweighted algorithm.
 *
 * Every undirected node pair is counted once by halving accumulated source
 * dependencies at the end.
 *
 * @throws std::invalid_argument If the graph is weighted or has self-loops.
 */
[[nodiscard]] EdgeBetweenness edge_betweenness_brandes(const Graph &graph);

/**
 * @brief Run Girvan--Newman, removing one maximum-betweenness edge per step.
 *
 * Betweenness is recomputed after every removal. A hierarchy level is recorded
 * whenever the number of connected components increases. If target_communities
 * is zero, the level with maximum modularity on the original graph is returned.
 *
 * @param graph A simple, undirected, unweighted graph.
 * @param config Algorithm parameters.
 * @return Partition result. The experiment wrapper adds runtime and final modularity.
 */
[[nodiscard]] AlgorithmResult run(const Graph &graph, const Config &config = {});

} // namespace cd::girvan_newman
