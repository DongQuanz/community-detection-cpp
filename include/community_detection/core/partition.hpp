#pragma once

#include "community_detection/core/graph.hpp"

#include <cstddef>
#include <vector>

namespace cd {

using Labels = std::vector<int>;

/**
 * @brief Normalize community labels to 0,1,...,K-1 in first-seen order.
 * @param labels Arbitrary, potentially non-contiguous labels.
 * @return Contiguous labels preserving the input partition.
 */
[[nodiscard]] Labels normalize_labels(const Labels &labels);

/**
 * @brief Count distinct communities in a label vector.
 */
[[nodiscard]] std::size_t count_communities(const Labels &labels);

/**
 * @brief Find graph connected components with breadth-first search.
 * @return Component labels normalized to 0..K-1.
 */
[[nodiscard]] Labels connected_components(const Graph &graph);

/**
 * @brief Split disconnected regions that happen to share a label.
 *
 * Label propagation can assign one label to disconnected regions because of
 * update history. This function preserves existing label boundaries and splits
 * each label further by connectivity.
 */
[[nodiscard]] Labels split_disconnected_communities(const Graph &graph, const Labels &labels);

} // namespace cd
