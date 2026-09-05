#pragma once

#include "community_detection/core/graph.hpp"
#include "community_detection/core/result.hpp"

#include <cstddef>
#include <cstdint>

namespace cd::label_propagation {

/** @brief Tie-breaking strategy among maximum-frequency labels. */
enum class TieBreak {
    Random,       ///< Choose randomly among maximum-frequency labels.
    SmallestLabel ///< Choose the smallest label for deterministic execution.
};

/** @brief Label Propagation Algorithm (LPA) parameters. */
struct Config {
    /** Maximum number of sweeps. */
    std::size_t max_iterations{100};

    /** true for asynchronous updates; false for synchronous sweeps. */
    bool asynchronous{true};

    /** Shuffle node order before each sweep. */
    bool shuffle_nodes{true};

    /** Label tie-breaking rule. */
    TieBreak tie_break{TieBreak::Random};

    /** Seed for shuffling and randomized tie-breaking. */
    std::uint64_t seed{42};

    /**
     * Split disconnected components that finish with the same label. This is
     * an optional post-processing step for connected output communities.
     */
    bool split_disconnected_labels{true};
};

/**
 * @brief Run label propagation starting with one unique label per node.
 *
 * The algorithm stops when every node holds a locally maximal neighbor label,
 * or at max_iterations. Isolated nodes retain their own labels.
 */
[[nodiscard]] AlgorithmResult run(const Graph &graph, const Config &config = {});

} // namespace cd::label_propagation
