#pragma once

#include "community_detection/core/graph.hpp"

#include <cstddef>
#include <cstdint>

namespace cd {

struct EdgeDropoutResult {
    Graph graph;
    std::size_t removed_edges{};
};

/**
 * Drop each edge independently with probability rate using a deterministic
 * hash of (u,v,seed). Two passes build CSR directly without copying an m-edge list.
 */
[[nodiscard]] EdgeDropoutResult drop_edges(const Graph &graph, double rate, std::uint64_t seed);

} // namespace cd
