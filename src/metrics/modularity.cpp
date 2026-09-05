#include "community_detection/metrics/modularity.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace cd::metrics {

double modularity(const Graph &graph, const Labels &labels, double resolution) {
    if (labels.size() != graph.num_nodes()) {
        throw std::invalid_argument("labels size must match graph.num_nodes()");
    }

    if (!(resolution > 0.0) || !std::isfinite(resolution)) {
        throw std::invalid_argument("modularity resolution must be finite and positive");
    }

    const double m = graph.total_edge_weight();
    if (m <= 0.0) {
        return 0.0;
    }

    const Labels normalized = normalize_labels(labels);
    const std::size_t k = count_communities(normalized);
    std::vector<double> internal_weight(k, 0.0);
    std::vector<double> degree_sum(k, 0.0);

    for (NodeId u = 0; u < graph.num_nodes(); ++u) {
        degree_sum[static_cast<std::size_t>(normalized[u])] += graph.weighted_degree(u);
    }

    graph.for_each_edge([&](const WeightedEdge &edge) {
        if (normalized[edge.u] == normalized[edge.v]) {
            internal_weight[static_cast<std::size_t>(normalized[edge.u])] += edge.weight;
        }
    });

    double q = 0.0;
    for (std::size_t c = 0; c < k; ++c) {
        const double observed = internal_weight[c] / m;
        const double expected = degree_sum[c] / (2.0 * m);
        q += observed - resolution * expected * expected;
    }
    return q;
}

} // namespace cd::metrics
