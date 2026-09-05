#pragma once

#include "community_detection/algorithms/girvan_newman.hpp"
#include "community_detection/algorithms/label_propagation.hpp"
#include "community_detection/algorithms/louvain.hpp"
#include "community_detection/algorithms/spectral.hpp"
#include "community_detection/core/graph.hpp"
#include "community_detection/core/result.hpp"

namespace cd::experiment {

/**
 * @brief Run Girvan--Newman, measure runtime, and compute final modularity.
 */
[[nodiscard]] AlgorithmResult run(const Graph &graph, const girvan_newman::Config &config);

/**
 * @brief Run spectral clustering, measure runtime, and compute final modularity.
 */
[[nodiscard]] AlgorithmResult run(const Graph &graph, const spectral::Config &config);

/**
 * @brief Run Louvain, measure runtime, and compute final modularity.
 */
[[nodiscard]] AlgorithmResult run(const Graph &graph, const louvain::Config &config);

/**
 * @brief Run LPA, measure runtime, and compute final modularity.
 */
[[nodiscard]] AlgorithmResult run(const Graph &graph, const label_propagation::Config &config);

} // namespace cd::experiment
