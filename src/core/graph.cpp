#include "community_detection/core/graph.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cd {
namespace {

bool same_endpoints(const WeightedEdge &lhs, const WeightedEdge &rhs) {
    return lhs.u == rhs.u && lhs.v == rhs.v;
}

} // namespace

Graph::Graph(std::size_t num_nodes) : num_nodes_(num_nodes) {
    if (num_nodes >= static_cast<std::size_t>(invalid_node())) {
        throw std::length_error("Graph has too many nodes for 32-bit NodeId");
    }
}

void Graph::validate_node(NodeId u) const {
    if (static_cast<std::size_t>(u) >= num_nodes_) {
        throw std::out_of_range("Graph node id is out of range");
    }
}

void Graph::reserve_edges(std::size_t num_edges) {
    thaw_for_mutation();
    pending_edges_.reserve(num_edges);
}

void Graph::add_edge(NodeId u, NodeId v, double weight) {
    validate_node(u);
    validate_node(v);
    if (!(weight > 0.0) || !std::isfinite(weight)) {
        throw std::invalid_argument("Edge weight must be finite and positive");
    }

    thaw_for_mutation();
    const auto endpoints = std::minmax(u, v);
    pending_edges_.push_back({endpoints.first, endpoints.second, weight});
}

void Graph::ensure_finalized() const {
    if (finalized_) {
        return;
    }

    std::sort(pending_edges_.begin(), pending_edges_.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
    });

    std::vector<WeightedEdge> merged;
    merged.reserve(pending_edges_.size());
    for (const WeightedEdge &edge : pending_edges_) {
        if (!merged.empty() && same_endpoints(merged.back(), edge)) {
            merged.back().weight += edge.weight;
        } else {
            merged.push_back(edge);
        }
    }

    offsets_.assign(num_nodes_ + 1, 0);
    bool weighted = false;
    total_edge_weight_ = 0.0;
    for (const WeightedEdge &edge : merged) {
        ++offsets_[static_cast<std::size_t>(edge.u) + 1];
        if (edge.u != edge.v) {
            ++offsets_[static_cast<std::size_t>(edge.v) + 1];
        }
        total_edge_weight_ += edge.weight;
        weighted = weighted || std::abs(edge.weight - 1.0) > 1e-12;
    }
    for (std::size_t i = 1; i < offsets_.size(); ++i) {
        offsets_[i] += offsets_[i - 1];
    }

    neighbors_.assign(static_cast<std::size_t>(offsets_.back()), invalid_node());
    if (weighted) {
        weights_.assign(neighbors_.size(), 0.0);
    } else {
        weights_.clear();
    }

    std::vector<std::uint64_t> cursor = offsets_;
    for (const WeightedEdge &edge : merged) {
        const auto put = [&](NodeId from, NodeId to) {
            const std::size_t position = static_cast<std::size_t>(cursor[from]++);
            neighbors_[position] = to;
            if (weighted) {
                weights_[position] = edge.weight;
            }
        };
        put(edge.u, edge.v);
        if (edge.u != edge.v) {
            put(edge.v, edge.u);
        }
    }

    edge_count_ = merged.size();
    std::vector<WeightedEdge>().swap(pending_edges_);
    finalized_ = true;
}

void Graph::thaw_for_mutation() {
    if (!finalized_) {
        return;
    }

    std::vector<WeightedEdge> active;
    active.reserve(edge_count_ + 1);
    for_each_edge([&](const WeightedEdge &edge) { active.push_back(edge); });
    pending_edges_ = std::move(active);
    offsets_.clear();
    neighbors_.clear();
    weights_.clear();
    edge_count_ = 0;
    total_edge_weight_ = 0.0;
    finalized_ = false;
}

double Graph::weight_at(std::size_t position) const noexcept {
    return weights_.empty() ? 1.0 : weights_[position];
}

bool Graph::remove_edge(NodeId u, NodeId v) {
    validate_node(u);
    validate_node(v);
    ensure_finalized();

    auto find_position = [&](NodeId from, NodeId to) -> std::size_t {
        const std::size_t begin = static_cast<std::size_t>(offsets_[from]);
        const std::size_t end =
            static_cast<std::size_t>(offsets_[static_cast<std::size_t>(from) + 1]);
        for (std::size_t position = begin; position < end; ++position) {
            if (neighbors_[position] == to) {
                return position;
            }
        }
        return neighbors_.size();
    };

    const std::size_t forward = find_position(u, v);
    if (forward == neighbors_.size()) {
        return false;
    }
    const double weight = weight_at(forward);
    neighbors_[forward] = invalid_node();
    if (!weights_.empty()) {
        weights_[forward] = 0.0;
    }

    if (u != v) {
        const std::size_t reverse = find_position(v, u);
        if (reverse == neighbors_.size()) {
            throw std::logic_error("Undirected CSR is missing the reverse edge");
        }
        neighbors_[reverse] = invalid_node();
        if (!weights_.empty()) {
            weights_[reverse] = 0.0;
        }
    }

    --edge_count_;
    total_edge_weight_ -= weight;
    if (std::abs(total_edge_weight_) < 1e-15) {
        total_edge_weight_ = 0.0;
    }
    return true;
}

bool Graph::has_edge(NodeId u, NodeId v) const { return edge_weight(u, v) > 0.0; }

double Graph::edge_weight(NodeId u, NodeId v) const {
    validate_node(u);
    validate_node(v);
    ensure_finalized();
    const std::size_t begin = static_cast<std::size_t>(offsets_[u]);
    const std::size_t end = static_cast<std::size_t>(offsets_[static_cast<std::size_t>(u) + 1]);
    for (std::size_t position = begin; position < end; ++position) {
        if (neighbors_[position] == v) {
            return weight_at(position);
        }
    }
    return 0.0;
}

Graph::NeighborRange Graph::neighbors(NodeId u) const {
    validate_node(u);
    ensure_finalized();
    return NeighborRange(this, offsets_[u], offsets_[static_cast<std::size_t>(u) + 1]);
}

std::size_t Graph::num_nodes() const noexcept { return num_nodes_; }

std::size_t Graph::num_edges() const {
    ensure_finalized();
    return edge_count_;
}

double Graph::weighted_degree(NodeId u) const {
    validate_node(u);
    double degree = 0.0;
    for (const Neighbor neighbor : neighbors(u)) {
        degree += neighbor.node == u ? 2.0 * neighbor.weight : neighbor.weight;
    }
    return degree;
}

double Graph::max_weighted_degree() const {
    double maximum = 0.0;
    for (std::size_t raw_u = 0; raw_u < num_nodes_; ++raw_u) {
        maximum = std::max(maximum, weighted_degree(static_cast<NodeId>(raw_u)));
    }
    return maximum;
}

double Graph::total_edge_weight() const {
    ensure_finalized();
    return total_edge_weight_;
}

bool Graph::is_weighted() const {
    ensure_finalized();
    return !weights_.empty();
}

std::vector<WeightedEdge> Graph::edges() const {
    std::vector<WeightedEdge> result;
    result.reserve(num_edges());
    for_each_edge([&](const WeightedEdge &edge) { result.push_back(edge); });
    return result;
}

bool Graph::is_simple_unweighted(double tolerance) const {
    ensure_finalized();
    if (!weights_.empty()) {
        bool all_unit = true;
        for_each_edge([&](const WeightedEdge &edge) {
            all_unit = all_unit && std::abs(edge.weight - 1.0) <= tolerance;
        });
        if (!all_unit) {
            return false;
        }
    }
    bool no_loops = true;
    for_each_edge([&](const WeightedEdge &edge) { no_loops = no_loops && edge.u != edge.v; });
    return no_loops;
}

Graph Graph::from_csr(std::size_t num_nodes, std::vector<std::uint64_t> offsets,
                      std::vector<NodeId> neighbors, std::vector<double> weights) {
    if (num_nodes >= static_cast<std::size_t>(invalid_node())) {
        throw std::length_error("Graph has too many nodes for 32-bit NodeId");
    }
    if (offsets.size() != num_nodes + 1 || offsets.empty() || offsets.front() != 0 ||
        offsets.back() != neighbors.size()) {
        throw std::invalid_argument("Invalid CSR offsets");
    }
    if (!weights.empty() && weights.size() != neighbors.size()) {
        throw std::invalid_argument("CSR weights size must match neighbors size");
    }
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1]) {
            throw std::invalid_argument("CSR offsets must be nondecreasing");
        }
    }
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        if (static_cast<std::size_t>(neighbors[i]) >= num_nodes) {
            throw std::invalid_argument("CSR neighbor is out of range");
        }
        if (!weights.empty() && (!(weights[i] > 0.0) || !std::isfinite(weights[i]))) {
            throw std::invalid_argument("CSR edge weights must be finite and positive");
        }
    }

    Graph graph(num_nodes);
    graph.offsets_ = std::move(offsets);
    graph.neighbors_ = std::move(neighbors);
    graph.weights_ = std::move(weights);
    graph.finalized_ = true;
    graph.edge_count_ = 0;
    graph.total_edge_weight_ = 0.0;
    graph.for_each_edge([&](const WeightedEdge &edge) {
        ++graph.edge_count_;
        graph.total_edge_weight_ += edge.weight;
    });
    return graph;
}

Graph::NeighborIterator::NeighborIterator(const Graph *graph, std::uint64_t position,
                                          std::uint64_t end)
    : graph_(graph), position_(position), end_(end) {
    skip_removed();
}

void Graph::NeighborIterator::skip_removed() {
    while (position_ < end_ &&
           graph_->neighbors_[static_cast<std::size_t>(position_)] == Graph::invalid_node()) {
        ++position_;
    }
}

Neighbor Graph::NeighborIterator::operator*() const {
    const std::size_t position = static_cast<std::size_t>(position_);
    return {graph_->neighbors_[position], graph_->weight_at(position)};
}

Graph::NeighborIterator &Graph::NeighborIterator::operator++() {
    ++position_;
    skip_removed();
    return *this;
}

bool Graph::NeighborIterator::operator==(const NeighborIterator &other) const noexcept {
    return graph_ == other.graph_ && position_ == other.position_ && end_ == other.end_;
}

bool Graph::NeighborIterator::operator!=(const NeighborIterator &other) const noexcept {
    return !(*this == other);
}

Graph::NeighborRange::NeighborRange(const Graph *graph, std::uint64_t begin, std::uint64_t end)
    : graph_(graph), begin_(begin), end_(end) {}

Graph::NeighborIterator Graph::NeighborRange::begin() const {
    return NeighborIterator(graph_, begin_, end_);
}

Graph::NeighborIterator Graph::NeighborRange::end() const {
    return NeighborIterator(graph_, end_, end_);
}

} // namespace cd
