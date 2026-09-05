#pragma once

#include "community_detection/core/graph.hpp"
#include "community_detection/core/partition.hpp"

namespace cd::metrics {

/**
 * @brief Compute Newman--Girvan modularity for an undirected graph partition.
 *
 * The formula supports weights and self-loops, so it also applies to Louvain's
 * aggregated graphs. For unweighted graphs it reduces to
 * Q = sum_C [l_C / m - (d_C / 2m)^2].
 *
 * @param graph Graph to evaluate.
 * @param labels One community label per node.
 * @return Modularity, or zero for a graph with no edges.
 * @throws std::invalid_argument If the label count differs from the node count.
 */
[[nodiscard]] double modularity(const Graph &graph, const Labels &labels, double resolution = 1.0);

} // namespace cd::metrics
