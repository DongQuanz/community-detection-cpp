#include "community_detection/core/partition.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace cd {

Labels normalize_labels(const Labels &labels) {
    Labels normalized;
    normalized.reserve(labels.size());

    std::unordered_map<int, int> remap;
    int next = 0;
    for (const int label : labels) {
        auto [it, inserted] = remap.emplace(label, next);
        if (inserted) {
            ++next;
        }
        normalized.push_back(it->second);
    }
    return normalized;
}

std::size_t count_communities(const Labels &labels) {
    if (labels.empty()) {
        return 0;
    }
    const Labels normalized = normalize_labels(labels);
    return static_cast<std::size_t>(*std::max_element(normalized.begin(), normalized.end()) + 1);
}

Labels connected_components(const Graph &graph) {
    const std::size_t n = graph.num_nodes();
    Labels labels(n, -1);
    int component = 0;

    for (NodeId start = 0; start < n; ++start) {
        if (labels[start] != -1) {
            continue;
        }

        std::queue<NodeId> queue;
        queue.push(start);
        labels[start] = component;

        while (!queue.empty()) {
            const NodeId u = queue.front();
            queue.pop();
            for (const auto &[v, _] : graph.neighbors(u)) {
                if (labels[v] == -1) {
                    labels[v] = component;
                    queue.push(v);
                }
            }
        }
        ++component;
    }
    return labels;
}

Labels split_disconnected_communities(const Graph &graph, const Labels &labels) {
    if (labels.size() != graph.num_nodes()) {
        throw std::invalid_argument("labels size must match graph.num_nodes()");
    }

    const std::size_t n = graph.num_nodes();
    Labels result(n, -1);
    int next_component = 0;

    for (NodeId start = 0; start < n; ++start) {
        if (result[start] != -1) {
            continue;
        }

        const int target_label = labels[start];
        std::queue<NodeId> queue;
        queue.push(start);
        result[start] = next_component;

        while (!queue.empty()) {
            const NodeId u = queue.front();
            queue.pop();
            for (const auto &[v, _] : graph.neighbors(u)) {
                if (result[v] == -1 && labels[v] == target_label) {
                    result[v] = next_component;
                    queue.push(v);
                }
            }
        }
        ++next_component;
    }
    return result;
}

} // namespace cd
