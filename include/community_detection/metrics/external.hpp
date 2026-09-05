#pragma once

#include "community_detection/core/partition.hpp"

#include <cstdint>
#include <limits>

namespace cd::metrics {

struct PairwiseAgreement {
    std::uint64_t same_both{};           // a
    std::uint64_t same_pred_only{};      // b
    std::uint64_t same_reference_only{}; // c
    std::uint64_t different_both{};      // d
    double precision{std::numeric_limits<double>::quiet_NaN()};
    double recall{std::numeric_limits<double>::quiet_NaN()};
    double f1{std::numeric_limits<double>::quiet_NaN()};
    double adjusted_rand{std::numeric_limits<double>::quiet_NaN()};
};

/** Pairwise precision/recall/F1 and ARI, invariant to label renaming. */
[[nodiscard]] PairwiseAgreement compare_partitions(const Labels &predicted,
                                                   const Labels &reference);

} // namespace cd::metrics
