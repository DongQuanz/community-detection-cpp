#pragma once

#include "community_detection/core/graph.hpp"
#include "community_detection/core/partition.hpp"

#include <filesystem>

namespace cd::io {

/** Read/write CDGRPH1 binary graphs (little-endian, normalized edges u <= v). */
[[nodiscard]] Graph load_binary_graph(const std::filesystem::path &path);
void write_binary_graph(const Graph &graph, const std::filesystem::path &path);

/**
 * Read TSV labels. A row may be `label` or `node_id<TAB>label`; comment lines
 * beginning with # and a `node_id label` header are ignored.
 */
[[nodiscard]] Labels load_labels_tsv(const std::filesystem::path &path, std::size_t expected_nodes);
void write_labels_tsv(const Labels &labels, const std::filesystem::path &path);

} // namespace cd::io
