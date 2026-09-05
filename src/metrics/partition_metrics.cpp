#include "community_detection/metrics/partition_metrics.hpp"

#include "community_detection/metrics/modularity.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cd::metrics {
namespace {

double quantile(const std::vector<double> &sorted, double probability) {
    if (sorted.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double index = probability * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(index));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
    const double fraction = index - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

DistributionSummary summarize(const std::vector<double> &values,
                              const std::vector<std::size_t> &node_weights) {
    if (values.size() != node_weights.size()) {
        throw std::logic_error("Distribution values and weights have different sizes");
    }

    std::vector<double> valid;
    valid.reserve(values.size());
    long double total = 0.0L;
    long double weighted_total = 0.0L;
    std::uint64_t total_weight = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            continue;
        }
        valid.push_back(values[i]);
        total += values[i];
        weighted_total += static_cast<long double>(values[i]) * node_weights[i];
        total_weight += node_weights[i];
    }

    DistributionSummary summary;
    summary.valid_count = valid.size();
    if (valid.empty()) {
        return summary;
    }
    std::sort(valid.begin(), valid.end());
    summary.minimum = valid.front();
    summary.q1 = quantile(valid, 0.25);
    summary.median = quantile(valid, 0.50);
    summary.q3 = quantile(valid, 0.75);
    summary.maximum = valid.back();
    summary.community_mean = static_cast<double>(total / valid.size());
    if (total_weight > 0) {
        summary.node_weighted_mean = static_cast<double>(weighted_total / total_weight);
    }
    return summary;
}

} // namespace

PartitionMetrics evaluate_partition(const Graph &graph, const Labels &labels, double resolution) {
    if (labels.size() != graph.num_nodes()) {
        throw std::invalid_argument("labels size must match graph.num_nodes()");
    }

    const Labels normalized = normalize_labels(labels);
    PartitionMetrics result;
    result.num_communities = count_communities(normalized);
    result.resolution = resolution;
    result.modularity = modularity(graph, normalized, resolution);
    result.communities.resize(result.num_communities);

    std::vector<std::size_t> sizes(result.num_communities, 0);
    for (NodeId node = 0; node < graph.num_nodes(); ++node) {
        const std::size_t community = static_cast<std::size_t>(normalized[node]);
        ++sizes[community];
        const double degree = graph.weighted_degree(node);
        result.communities[community].volume += degree;
        result.isolated_nodes += degree == 0.0 ? 1 : 0;
    }

    graph.for_each_edge([&](const WeightedEdge &edge) {
        const std::size_t left = static_cast<std::size_t>(normalized[edge.u]);
        const std::size_t right = static_cast<std::size_t>(normalized[edge.v]);
        if (left == right) {
            result.communities[left].internal_weight += edge.weight;
            if (edge.u != edge.v) {
                ++result.communities[left].internal_edges;
            }
        } else {
            result.communities[left].boundary_weight += edge.weight;
            result.communities[right].boundary_weight += edge.weight;
        }
    });

    const double m = graph.total_edge_weight();
    const double total_volume = 2.0 * m;
    double internal_weight = 0.0;
    result.ratio_cut = 0.0;
    result.normalized_cut = 0.0;
    bool ncut_defined = true;

    std::vector<double> size_values;
    std::vector<double> density_values;
    std::vector<double> conductance_values;
    size_values.reserve(result.num_communities);
    density_values.reserve(result.num_communities);
    conductance_values.reserve(result.num_communities);

    for (std::size_t community = 0; community < result.num_communities; ++community) {
        CommunityMetrics &metrics = result.communities[community];
        metrics.community = community;
        metrics.size = sizes[community];
        result.singleton_communities += metrics.size == 1 ? 1 : 0;
        internal_weight += metrics.internal_weight;

        if (metrics.size >= 2) {
            metrics.density =
                2.0 * static_cast<double>(metrics.internal_edges) /
                (static_cast<double>(metrics.size) * static_cast<double>(metrics.size - 1));
        }
        const double smaller_volume = std::min(metrics.volume, total_volume - metrics.volume);
        if (smaller_volume > 0.0) {
            metrics.conductance = metrics.boundary_weight / smaller_volume;
        }

        result.ratio_cut += metrics.boundary_weight / static_cast<double>(metrics.size);
        if (metrics.volume > 0.0) {
            result.normalized_cut += metrics.boundary_weight / metrics.volume;
        } else {
            ncut_defined = false;
        }

        size_values.push_back(static_cast<double>(metrics.size));
        density_values.push_back(metrics.density);
        conductance_values.push_back(metrics.conductance);
    }

    if (m > 0.0) {
        result.coverage = internal_weight / m;
    }
    if (!ncut_defined) {
        result.normalized_cut = std::numeric_limits<double>::quiet_NaN();
    }

    result.size_distribution = summarize(size_values, sizes);
    result.density_distribution = summarize(density_values, sizes);
    result.conductance_distribution = summarize(conductance_values, sizes);
    return result;
}

} // namespace cd::metrics
