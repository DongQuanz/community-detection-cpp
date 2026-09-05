#pragma once

#include "community_detection/core/graph.hpp"
#include "community_detection/core/partition.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace cd::metrics {

struct CommunityMetrics {
    std::size_t community{};
    std::size_t size{};
    std::size_t internal_edges{};
    double internal_weight{0.0};
    double boundary_weight{0.0};
    double volume{0.0};
    double density{std::numeric_limits<double>::quiet_NaN()};
    double conductance{std::numeric_limits<double>::quiet_NaN()};
};

/** Distribution statistics; NaN values are excluded from the sample. */
struct DistributionSummary {
    std::size_t valid_count{};
    double minimum{std::numeric_limits<double>::quiet_NaN()};
    double q1{std::numeric_limits<double>::quiet_NaN()};
    double median{std::numeric_limits<double>::quiet_NaN()};
    double q3{std::numeric_limits<double>::quiet_NaN()};
    double maximum{std::numeric_limits<double>::quiet_NaN()};
    double community_mean{std::numeric_limits<double>::quiet_NaN()};
    double node_weighted_mean{std::numeric_limits<double>::quiet_NaN()};
};

struct PartitionMetrics {
    std::size_t num_communities{};
    std::size_t singleton_communities{};
    std::size_t isolated_nodes{};
    double resolution{1.0};
    double modularity{std::numeric_limits<double>::quiet_NaN()};
    double coverage{std::numeric_limits<double>::quiet_NaN()};
    double ratio_cut{std::numeric_limits<double>::quiet_NaN()};
    double normalized_cut{std::numeric_limits<double>::quiet_NaN()};
    std::vector<CommunityMetrics> communities;
    DistributionSummary size_distribution;
    DistributionSummary density_distribution;
    DistributionSummary conductance_distribution;
};

/**
 * Compute all intrinsic metrics used by the benchmark: density, conductance,
 * coverage, Q_gamma, RatioCut, Ncut, and per-community distributions.
 */
[[nodiscard]] PartitionMetrics evaluate_partition(const Graph &graph, const Labels &labels,
                                                  double resolution = 1.0);

} // namespace cd::metrics
