#include "community_detection/algorithms/louvain.hpp"

#include "community_detection/core/partition.hpp"
#include "community_detection/metrics/modularity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace cd::louvain {
namespace {

struct LocalMoveResult {
    Labels labels;
    std::size_t passes{0};
    bool moved{false};
};

/**
 * @brief Accumulate edge weight from one node to each neighboring community.
 *
 * Self-loops are ignored because their contribution does not change when the
 * node moves and therefore cancels in the modularity delta.
 */
class CommunityWeightAccumulator {
  public:
    explicit CommunityWeightAccumulator(std::size_t community_capacity)
        : weights_(community_capacity, 0.0), stamps_(community_capacity, 0) {}

    void collect(const Graph &graph, NodeId node, const Labels &community) {
        advance_epoch();
        touched_.clear();
        for (const auto &[neighbor, weight] : graph.neighbors(node)) {
            if (neighbor == node) {
                continue;
            }
            const int label = community[neighbor];
            if (label < 0 || static_cast<std::size_t>(label) >= weights_.size()) {
                throw std::logic_error("Louvain community label is outside the accumulator range");
            }
            const std::size_t index = static_cast<std::size_t>(label);
            if (stamps_[index] != epoch_) {
                stamps_[index] = epoch_;
                weights_[index] = 0.0;
                touched_.push_back(label);
            }
            weights_[index] += weight;
        }
    }

    [[nodiscard]] double weight(int community) const {
        const std::size_t index = static_cast<std::size_t>(community);
        return stamps_[index] == epoch_ ? weights_[index] : 0.0;
    }

    [[nodiscard]] const std::vector<int> &touched() const noexcept { return touched_; }

  private:
    void advance_epoch() {
        if (epoch_ == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(stamps_.begin(), stamps_.end(), 0);
            epoch_ = 1;
        } else {
            ++epoch_;
        }
    }

    std::vector<double> weights_;
    std::vector<std::uint32_t> stamps_;
    std::vector<int> touched_;
    std::uint32_t epoch_{0};
};

/**
 * @brief Louvain local-moving phase on one graph level.
 *
 * After temporarily removing i's degree from its old community, insertion into
 * C is compared using a score proportional to delta Q:
 *   k_{i->C} - k_i * d_C / (2m).
 * The common 1/m factor is unnecessary when computing only the argmax.
 */
LocalMoveResult local_moving(const Graph &graph, const Config &config, std::mt19937_64 &rng) {
    const std::size_t n = graph.num_nodes();
    Labels community(n);
    std::iota(community.begin(), community.end(), 0);

    std::vector<double> node_degrees(n, 0.0);
    std::vector<double> community_degree(n, 0.0);
    for (NodeId node = 0; node < n; ++node) {
        node_degrees[node] = graph.weighted_degree(node);
        community_degree[node] = node_degrees[node];
    }

    std::vector<NodeId> order(n);
    std::iota(order.begin(), order.end(), NodeId{0});
    CommunityWeightAccumulator neighbor_weights(n);
    const double m = graph.total_edge_weight();

    LocalMoveResult result;
    if (m <= 0.0 || n == 0) {
        result.labels = std::move(community);
        return result;
    }

    double previous_q = metrics::modularity(graph, community, config.resolution);

    for (std::size_t pass = 0; pass < config.max_passes_per_level; ++pass) {
        if (config.shuffle_nodes) {
            std::shuffle(order.begin(), order.end(), rng);
        }

        bool moved_this_pass = false;
        for (const NodeId node : order) {
            const int old_community = community[node];
            const double node_degree = node_degrees[node];
            neighbor_weights.collect(graph, node, community);

            community_degree[static_cast<std::size_t>(old_community)] -= node_degree;

            const double weight_to_old = neighbor_weights.weight(old_community);
            int best_community = old_community;
            const double old_score =
                weight_to_old - config.resolution * node_degree *
                                    community_degree[static_cast<std::size_t>(old_community)] /
                                    (2.0 * m);
            double best_score = old_score;

            for (const int candidate : neighbor_weights.touched()) {
                const double weight_to_candidate = neighbor_weights.weight(candidate);
                const double score = weight_to_candidate -
                                     config.resolution * node_degree *
                                         community_degree[static_cast<std::size_t>(candidate)] /
                                         (2.0 * m);

                if (score > best_score + 1e-15 ||
                    (std::abs(score - best_score) <= 1e-15 && score > old_score + 1e-15 &&
                     candidate < best_community)) {
                    best_score = score;
                    best_community = candidate;
                }
            }

            community[node] = best_community;
            community_degree[static_cast<std::size_t>(best_community)] += node_degree;
            if (best_community != old_community) {
                moved_this_pass = true;
                result.moved = true;
            }
        }

        result.passes = pass + 1;
        const double current_q = metrics::modularity(graph, community, config.resolution);
        const double gain = current_q - previous_q;
        previous_q = current_q;

        if (!moved_this_pass || gain <= config.min_modularity_gain) {
            break;
        }
    }

    result.labels = normalize_labels(community);
    return result;
}

/**
 * @brief Contract each community into a supernode and accumulate edge weights.
 *
 * Internal edges become self-loops; inter-community edges become weighted
 * superedges whose weights equal the corresponding original edge-weight sums.
 */
Graph aggregate_graph(const Graph &graph, const Labels &partition) {
    const Labels normalized = normalize_labels(partition);
    const std::size_t k = count_communities(normalized);

    // Group nodes into a member CSR so each supergraph row is built in one pass.
    // This avoids materializing and sorting O(m) WeightedEdges, which otherwise
    // substantially increases peak memory on Reddit2.
    std::vector<std::uint64_t> member_offsets(k + 1, 0);
    for (const int label : normalized) {
        ++member_offsets[static_cast<std::size_t>(label) + 1];
    }
    for (std::size_t i = 1; i < member_offsets.size(); ++i) {
        member_offsets[i] += member_offsets[i - 1];
    }
    std::vector<NodeId> members(graph.num_nodes());
    std::vector<std::uint64_t> member_cursor = member_offsets;
    for (NodeId node = 0; node < graph.num_nodes(); ++node) {
        const std::size_t community = static_cast<std::size_t>(normalized[node]);
        members[static_cast<std::size_t>(member_cursor[community]++)] = node;
    }

    std::vector<std::uint64_t> offsets(k + 1, 0);
    std::vector<NodeId> neighbors;
    std::vector<double> edge_weights;
    const std::size_t maximum_size = std::numeric_limits<std::size_t>::max();
    const std::size_t edge_entry_bound =
        graph.num_edges() > maximum_size / 2 ? maximum_size : 2 * graph.num_edges();
    const std::size_t community_entry_bound = k > maximum_size / 16 ? maximum_size : 16 * k;
    const std::size_t reserve_guess = std::min(edge_entry_bound, community_entry_bound);
    neighbors.reserve(reserve_guess);
    edge_weights.reserve(reserve_guess);

    std::vector<double> row_weights(k, 0.0);
    std::vector<std::uint32_t> stamps(k, 0);
    std::vector<NodeId> touched;
    std::uint32_t epoch = 0;

    for (std::size_t community = 0; community < k; ++community) {
        if (epoch == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(stamps.begin(), stamps.end(), 0);
            epoch = 1;
        } else {
            ++epoch;
        }
        touched.clear();

        for (std::uint64_t position = member_offsets[community];
             position < member_offsets[community + 1]; ++position) {
            const NodeId node = members[static_cast<std::size_t>(position)];
            for (const Neighbor neighbor : graph.neighbors(node)) {
                const std::size_t target = static_cast<std::size_t>(normalized[neighbor.node]);
                if (stamps[target] != epoch) {
                    stamps[target] = epoch;
                    row_weights[target] = 0.0;
                    touched.push_back(static_cast<NodeId>(target));
                }

                // A regular internal edge is visited from both endpoints. A
                // self-loop occurs once in CSR and must not be halved.
                const bool internal_non_loop = target == community && neighbor.node != node;
                row_weights[target] += internal_non_loop ? 0.5 * neighbor.weight : neighbor.weight;
            }
        }

        std::sort(touched.begin(), touched.end());
        for (const NodeId target : touched) {
            neighbors.push_back(target);
            edge_weights.push_back(row_weights[target]);
        }
        offsets[community + 1] = neighbors.size();
    }

    return Graph::from_csr(k, std::move(offsets), std::move(neighbors), std::move(edge_weights));
}

} // namespace

AlgorithmResult run(const Graph &graph, const Config &config) {
    if (!(config.resolution > 0.0) || !std::isfinite(config.resolution)) {
        throw std::invalid_argument("Louvain resolution must be finite and positive");
    }
    if (config.max_levels == 0 || config.max_passes_per_level == 0) {
        throw std::invalid_argument("Louvain level/pass limits must be positive");
    }
    if (!(config.min_modularity_gain >= 0.0) || !std::isfinite(config.min_modularity_gain)) {
        throw std::invalid_argument("Louvain minimum gain must be finite and nonnegative");
    }
    AlgorithmResult result;
    result.algorithm = "louvain";

    const std::size_t n = graph.num_nodes();
    if (n == 0) {
        result.labels = {};
        result.objective = 0.0;
        return result;
    }

    Labels original_to_current(n);
    std::iota(original_to_current.begin(), original_to_current.end(), 0);
    Labels final_labels = original_to_current;

    // Read the original graph directly at level zero to avoid duplicating the
    // Reddit2 CSR. Own a separate (usually smaller) graph only after contraction.
    Graph aggregated_storage;
    const Graph *current_graph = &graph;
    std::mt19937_64 rng(config.seed);
    double previous_q = metrics::modularity(graph, final_labels, config.resolution);
    std::size_t total_passes = 0;

    for (std::size_t level = 0; level < config.max_levels; ++level) {
        LocalMoveResult local = local_moving(*current_graph, config, rng);
        total_passes += local.passes;

        for (NodeId original = 0; original < n; ++original) {
            const int current_node = original_to_current[original];
            original_to_current[original] = local.labels[static_cast<std::size_t>(current_node)];
        }
        final_labels = normalize_labels(original_to_current);

        if (result.hierarchy.empty() || result.hierarchy.back() != final_labels) {
            result.hierarchy.push_back(final_labels);
        }

        const double q = metrics::modularity(graph, final_labels, config.resolution);
        const std::size_t communities = count_communities(local.labels);
        const bool no_aggregation = communities == current_graph->num_nodes();
        const double level_gain = q - previous_q;

        previous_q = q;
        if (no_aggregation || !local.moved || level_gain <= config.min_modularity_gain) {
            break;
        }

        aggregated_storage = aggregate_graph(*current_graph, local.labels);
        current_graph = &aggregated_storage;
    }

    result.labels = normalize_labels(final_labels);
    result.objective = metrics::modularity(graph, result.labels, config.resolution);
    result.diagnostics["levels"] = static_cast<double>(result.hierarchy.size());
    result.diagnostics["local_passes"] = static_cast<double>(total_passes);
    result.diagnostics["resolution"] = config.resolution;
    return result;
}

} // namespace cd::louvain
