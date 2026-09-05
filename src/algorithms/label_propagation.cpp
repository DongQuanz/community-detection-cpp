#include "community_detection/algorithms/label_propagation.hpp"

#include "community_detection/core/partition.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace cd::label_propagation {
namespace {

/**
 * @brief Collect labels with maximum incident weight around a node.
 *
 * The stamp array reuses memory across nodes without clearing O(n) entries each
 * time. On unweighted graphs, accumulated weight is the label frequency.
 * Self-loops are ignored because this LPA operates on normalized input graphs.
 */
class LabelAccumulator {
  public:
    explicit LabelAccumulator(std::size_t label_capacity)
        : scores_(label_capacity, 0.0), stamps_(label_capacity, 0) {}

    void collect(const Graph &graph, NodeId node, const Labels &labels, std::vector<int> &modes) {
        advance_epoch();
        touched_.clear();
        modes.clear();
        double best = -1.0;

        for (const auto &[neighbor, weight] : graph.neighbors(node)) {
            if (neighbor == node) {
                continue;
            }
            const int label = labels[neighbor];
            if (label < 0 || static_cast<std::size_t>(label) >= scores_.size()) {
                throw std::logic_error("LPA label is outside the accumulator range");
            }
            const std::size_t index = static_cast<std::size_t>(label);
            if (stamps_[index] != epoch_) {
                stamps_[index] = epoch_;
                scores_[index] = 0.0;
                touched_.push_back(label);
            }
            scores_[index] += weight;
            best = std::max(best, scores_[index]);
        }

        if (touched_.empty()) {
            modes.push_back(labels[node]);
            return;
        }

        constexpr double eps = 1e-12;
        for (const int label : touched_) {
            if (std::abs(scores_[static_cast<std::size_t>(label)] - best) <= eps) {
                modes.push_back(label);
            }
        }
        std::sort(modes.begin(), modes.end());
    }

  private:
    void advance_epoch() {
        if (epoch_ == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(stamps_.begin(), stamps_.end(), 0);
            epoch_ = 1;
        } else {
            ++epoch_;
        }
    }

    std::vector<double> scores_;
    std::vector<std::uint32_t> stamps_;
    std::vector<int> touched_;
    std::uint32_t epoch_{0};
};

/**
 * @brief Choose a label from a tied set using the configured policy.
 */
int choose_label(const std::vector<int> &candidates, TieBreak tie_break, std::mt19937_64 &rng) {
    if (candidates.empty()) {
        throw std::logic_error("LPA tie candidates cannot be empty");
    }
    if (tie_break == TieBreak::SmallestLabel || candidates.size() == 1) {
        return candidates.front();
    }
    std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
    return candidates[pick(rng)];
}

/**
 * @brief Test whether every node currently has a locally maximal label.
 */
bool is_labeling_stable(const Graph &graph, const Labels &labels, LabelAccumulator &accumulator,
                        std::vector<int> &modes) {
    for (NodeId node = 0; node < graph.num_nodes(); ++node) {
        accumulator.collect(graph, node, labels, modes);
        if (std::find(modes.begin(), modes.end(), labels[node]) == modes.end()) {
            return false;
        }
    }
    return true;
}

} // namespace

AlgorithmResult run(const Graph &graph, const Config &config) {
    if (config.max_iterations == 0) {
        throw std::invalid_argument("LPA max_iterations must be positive");
    }
    AlgorithmResult result;
    result.algorithm = "label_propagation";

    const std::size_t n = graph.num_nodes();
    Labels labels(n);
    std::iota(labels.begin(), labels.end(), 0);

    if (n == 0) {
        result.labels = {};
        result.diagnostics["iterations"] = 0.0;
        return result;
    }

    std::vector<NodeId> order(n);
    std::iota(order.begin(), order.end(), NodeId{0});
    std::mt19937_64 rng(config.seed);
    LabelAccumulator accumulator(n);
    std::vector<int> modes;

    std::size_t iterations = 0;
    bool stable = false;

    for (; iterations < config.max_iterations; ++iterations) {
        if (config.shuffle_nodes) {
            std::shuffle(order.begin(), order.end(), rng);
        }

        if (config.asynchronous) {
            for (const NodeId node : order) {
                accumulator.collect(graph, node, labels, modes);
                labels[node] = choose_label(modes, config.tie_break, rng);
            }
        } else {
            const Labels previous = labels;
            Labels next = labels;
            for (const NodeId node : order) {
                accumulator.collect(graph, node, previous, modes);
                next[node] = choose_label(modes, config.tie_break, rng);
            }
            labels = std::move(next);
        }

        if (is_labeling_stable(graph, labels, accumulator, modes)) {
            stable = true;
            ++iterations;
            break;
        }
    }

    if (config.split_disconnected_labels) {
        labels = split_disconnected_communities(graph, labels);
    } else {
        labels = normalize_labels(labels);
    }

    result.labels = std::move(labels);
    result.diagnostics["iterations"] = static_cast<double>(iterations);
    result.diagnostics["converged"] = stable ? 1.0 : 0.0;
    return result;
}

} // namespace cd::label_propagation
