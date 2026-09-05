#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

namespace cd {

using NodeId = std::uint32_t;

/** A weighted undirected edge, normalized so that u <= v. */
struct WeightedEdge {
    NodeId u{};
    NodeId v{};
    double weight{1.0};
};

/** An adjacency-list entry. */
struct Neighbor {
    NodeId node{};
    double weight{1.0};
};

/**
 * An undirected graph stored in compressed sparse row (CSR) form.
 *
 * Calls to add_edge() are buffered; edges are normalized and CSR is built on
 * first access. This keeps construction convenient for small examples while
 * avoiding millions of per-node allocations on Reddit2. Adjacency weights are
 * allocated only for weighted graphs, so a simple unweighted graph stores one
 * NodeId per directed adjacency entry.
 */
class Graph {
  public:
    class NeighborIterator;
    class NeighborRange;

    explicit Graph(std::size_t num_nodes = 0);

    /** Add an edge, accumulating weight when the edge already exists. */
    void add_edge(NodeId u, NodeId v, double weight = 1.0);

    /** Reserve storage for graph construction through add_edge(). */
    void reserve_edges(std::size_t num_edges);

    /** Remove edge {u,v}; primarily used by Girvan--Newman. */
    bool remove_edge(NodeId u, NodeId v);

    [[nodiscard]] bool has_edge(NodeId u, NodeId v) const;
    [[nodiscard]] double edge_weight(NodeId u, NodeId v) const;
    [[nodiscard]] NeighborRange neighbors(NodeId u) const;

    [[nodiscard]] std::size_t num_nodes() const noexcept;
    [[nodiscard]] std::size_t num_edges() const;
    [[nodiscard]] double weighted_degree(NodeId u) const;
    [[nodiscard]] double max_weighted_degree() const;
    [[nodiscard]] double total_edge_weight() const;
    [[nodiscard]] bool is_weighted() const;

    /** Materialize the edge list; avoid this operation in large hot loops. */
    [[nodiscard]] std::vector<WeightedEdge> edges() const;

    /** Visit every undirected edge once without an intermediate vector. */
    template <typename Callable> void for_each_edge(Callable &&callable) const {
        ensure_finalized();
        for (std::size_t raw_u = 0; raw_u < num_nodes_; ++raw_u) {
            const NodeId u = static_cast<NodeId>(raw_u);
            for (std::uint64_t pos = offsets_[raw_u]; pos < offsets_[raw_u + 1]; ++pos) {
                const NodeId v = neighbors_[static_cast<std::size_t>(pos)];
                if (v == invalid_node() || u > v) {
                    continue;
                }
                const double weight =
                    weights_.empty() ? 1.0 : weights_[static_cast<std::size_t>(pos)];
                callable(WeightedEdge{u, v, weight});
            }
        }
    }

    [[nodiscard]] bool is_simple_unweighted(double tolerance = 1e-12) const;

    /**
     * Build directly from symmetric CSR. offsets has n+1 entries; neighbors
     * stores both directions of each non-self edge. weights may be empty (all
     * weights equal one) or have the same size as neighbors.
     */
    [[nodiscard]] static Graph from_csr(std::size_t num_nodes, std::vector<std::uint64_t> offsets,
                                        std::vector<NodeId> neighbors,
                                        std::vector<double> weights = {});

    class NeighborIterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Neighbor;
        using difference_type = std::ptrdiff_t;

        NeighborIterator(const Graph *graph, std::uint64_t position, std::uint64_t end);
        [[nodiscard]] Neighbor operator*() const;
        NeighborIterator &operator++();
        [[nodiscard]] bool operator==(const NeighborIterator &other) const noexcept;
        [[nodiscard]] bool operator!=(const NeighborIterator &other) const noexcept;

      private:
        void skip_removed();

        const Graph *graph_{};
        std::uint64_t position_{};
        std::uint64_t end_{};
    };

    class NeighborRange {
      public:
        NeighborRange(const Graph *graph, std::uint64_t begin, std::uint64_t end);
        [[nodiscard]] NeighborIterator begin() const;
        [[nodiscard]] NeighborIterator end() const;

      private:
        const Graph *graph_{};
        std::uint64_t begin_{};
        std::uint64_t end_{};
    };

  private:
    [[nodiscard]] static constexpr NodeId invalid_node() noexcept {
        return std::numeric_limits<NodeId>::max();
    }

    void validate_node(NodeId u) const;
    void ensure_finalized() const;
    void thaw_for_mutation();
    [[nodiscard]] double weight_at(std::size_t position) const noexcept;

    std::size_t num_nodes_{0};
    mutable bool finalized_{false};
    mutable std::vector<WeightedEdge> pending_edges_;
    mutable std::vector<std::uint64_t> offsets_;
    mutable std::vector<NodeId> neighbors_;
    mutable std::vector<double> weights_;
    mutable std::size_t edge_count_{0};
    mutable double total_edge_weight_{0.0};
};

} // namespace cd
