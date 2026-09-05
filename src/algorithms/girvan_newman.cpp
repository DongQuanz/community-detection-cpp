#include "community_detection/algorithms/girvan_newman.hpp"

#include "community_detection/core/partition.hpp"
#include "community_detection/metrics/modularity.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace cd::girvan_newman {

EdgeBetweenness edge_betweenness_brandes(const Graph &graph) {
    if (!graph.is_simple_unweighted()) {
        throw std::invalid_argument(
            "Girvan-Newman Brandes implementation requires a simple unweighted graph");
    }

    const std::size_t n = graph.num_nodes();
    EdgeBetweenness betweenness;
    graph.for_each_edge([&](const WeightedEdge &edge) { betweenness[{edge.u, edge.v}] = 0.0; });

    // Reuse buffers across sources. Girvan--Newman invokes Brandes repeatedly;
    // avoiding n predecessor-vector allocations per BFS materially reduces
    // overhead even on graphs with only a few hundred nodes.
    std::vector<std::vector<NodeId>> predecessors(n);
    std::vector<double> sigma(n, 0.0);
    std::vector<double> dependency(n, 0.0);
    std::vector<int> distance(n, -1);
    std::vector<NodeId> stack;
    std::vector<NodeId> queue;
    stack.reserve(n);
    queue.reserve(n);

    for (NodeId source = 0; source < n; ++source) {
        for (auto &values : predecessors) {
            values.clear();
        }
        std::fill(sigma.begin(), sigma.end(), 0.0);
        std::fill(dependency.begin(), dependency.end(), 0.0);
        std::fill(distance.begin(), distance.end(), -1);
        stack.clear();
        queue.clear();

        sigma[source] = 1.0;
        distance[source] = 0;
        queue.push_back(source);

        std::size_t queue_head = 0;
        while (queue_head < queue.size()) {
            const NodeId v = queue[queue_head++];
            stack.push_back(v);

            for (const auto &[w, _] : graph.neighbors(v)) {
                if (distance[w] < 0) {
                    distance[w] = distance[v] + 1;
                    queue.push_back(w);
                }
                if (distance[w] == distance[v] + 1) {
                    sigma[w] += sigma[v];
                    predecessors[w].push_back(v);
                }
            }
        }

        while (!stack.empty()) {
            const NodeId w = stack.back();
            stack.pop_back();

            for (const NodeId v : predecessors[w]) {
                if (sigma[w] == 0.0) {
                    continue;
                }
                const double credit = (sigma[v] / sigma[w]) * (1.0 + dependency[w]);
                const Edge edge = std::minmax(v, w);
                betweenness[edge] += credit;
                dependency[v] += credit;
            }
        }
    }

    for (auto &[_, value] : betweenness) {
        value *= 0.5;
    }
    return betweenness;
}

AlgorithmResult run(const Graph &graph, const Config &config) {
    if (!(config.tie_tolerance >= 0.0) || !std::isfinite(config.tie_tolerance)) {
        throw std::invalid_argument("Girvan-Newman tie tolerance must be finite and nonnegative");
    }
    if (config.target_communities > graph.num_nodes()) {
        throw std::invalid_argument("target_communities cannot exceed graph.num_nodes()");
    }
    if (!graph.is_simple_unweighted()) {
        throw std::invalid_argument(
            "Girvan-Newman requires a simple undirected unweighted input graph");
    }

    AlgorithmResult result;
    result.algorithm = "girvan_newman";

    if (graph.num_nodes() == 0) {
        result.labels = {};
        result.objective = 0.0;
        return result;
    }

    Graph working = graph;
    Labels current = connected_components(working);
    std::size_t current_k = count_communities(current);

    if (config.target_communities > 0 && config.target_communities < current_k) {
        throw std::invalid_argument(
            "target_communities cannot be smaller than the initial number of connected components");
    }

    result.hierarchy.push_back(current);
    Labels best_labels = current;
    double best_modularity = metrics::modularity(graph, current);

    if (config.target_communities > 0 && current_k == config.target_communities) {
        result.labels = current;
        result.objective = best_modularity;
        result.diagnostics["edges_removed"] = 0.0;
        result.diagnostics["hierarchy_levels"] = 1.0;
        return result;
    }

    std::mt19937_64 rng(config.seed);
    std::size_t removed = 0;

    while (working.num_edges() > 0 && removed < config.max_edge_removals) {
        const EdgeBetweenness scores = edge_betweenness_brandes(working);
        if (scores.empty()) {
            break;
        }

        double max_score = -1.0;
        for (const auto &[_, score] : scores) {
            max_score = std::max(max_score, score);
        }

        std::vector<Edge> candidates;
        for (const auto &[edge, score] : scores) {
            if (std::abs(score - max_score) <= config.tie_tolerance) {
                candidates.push_back(edge);
            }
        }

        Edge chosen = candidates.front();
        if (config.tie_break == TieBreak::Random && candidates.size() > 1) {
            std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
            chosen = candidates[pick(rng)];
        }

        working.remove_edge(chosen.first, chosen.second);
        ++removed;

        Labels next = connected_components(working);
        const std::size_t next_k = count_communities(next);
        if (next_k > current_k) {
            current = std::move(next);
            current_k = next_k;
            result.hierarchy.push_back(current);

            const double q = metrics::modularity(graph, current);
            if (q > best_modularity) {
                best_modularity = q;
                best_labels = current;
            }

            if (config.target_communities > 0 && current_k >= config.target_communities) {
                best_labels = current;
                best_modularity = q;
                break;
            }
        }
    }

    result.labels = normalize_labels(best_labels);
    result.objective = best_modularity;
    result.diagnostics["edges_removed"] = static_cast<double>(removed);
    result.diagnostics["hierarchy_levels"] = static_cast<double>(result.hierarchy.size());
    return result;
}

} // namespace cd::girvan_newman
