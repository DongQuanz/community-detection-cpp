#pragma once

#include "community_detection/core/partition.hpp"

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace cd {

/**
 * @brief Common normalized result returned by every algorithm.
 */
struct AlgorithmResult {
    /** Algorithm identifier. */
    std::string algorithm;

    /** Final community label for each input node. */
    Labels labels;

    /**
     * Intermediate partitions for hierarchical algorithms. Girvan--Newman and
     * Louvain populate this field; other algorithms may leave it empty.
     */
    std::vector<Labels> hierarchy;

    /**
     * Algorithm-specific objective, when defined. Examples include
     * RatioCut/Ncut for spectral clustering and modularity for Louvain.
     */
    double objective{std::numeric_limits<double>::quiet_NaN()};

    /** Final partition modularity, always evaluated on the original graph. */
    double modularity{std::numeric_limits<double>::quiet_NaN()};

    /** Algorithm runtime in milliseconds. */
    double elapsed_ms{0.0};

    /** Additional diagnostics recorded by the experiment pipeline. */
    std::unordered_map<std::string, double> diagnostics;
};

} // namespace cd
