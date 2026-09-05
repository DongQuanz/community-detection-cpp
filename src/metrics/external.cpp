#include "community_detection/metrics/external.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace cd::metrics {
namespace {

std::uint64_t choose_two(std::uint64_t value) { return value < 2 ? 0 : value * (value - 1) / 2; }

std::uint64_t joint_key(std::uint32_t left, std::uint32_t right) {
    return (static_cast<std::uint64_t>(left) << 32U) | right;
}

} // namespace

PairwiseAgreement compare_partitions(const Labels &predicted, const Labels &reference) {
    if (predicted.size() != reference.size()) {
        throw std::invalid_argument("partitions must contain the same number of nodes");
    }

    const Labels pred = normalize_labels(predicted);
    const Labels ref = normalize_labels(reference);
    std::unordered_map<int, std::uint64_t> pred_counts;
    std::unordered_map<int, std::uint64_t> ref_counts;
    std::unordered_map<std::uint64_t, std::uint64_t> joint_counts;
    pred_counts.reserve(count_communities(pred));
    ref_counts.reserve(count_communities(ref));

    for (std::size_t i = 0; i < pred.size(); ++i) {
        ++pred_counts[pred[i]];
        ++ref_counts[ref[i]];
        ++joint_counts[joint_key(static_cast<std::uint32_t>(pred[i]),
                                 static_cast<std::uint32_t>(ref[i]))];
    }

    std::uint64_t true_positive = 0;
    std::uint64_t predicted_positive = 0;
    std::uint64_t reference_positive = 0;
    for (const auto &[_, count] : joint_counts) {
        true_positive += choose_two(count);
    }
    for (const auto &[_, count] : pred_counts) {
        predicted_positive += choose_two(count);
    }
    for (const auto &[_, count] : ref_counts) {
        reference_positive += choose_two(count);
    }

    const std::uint64_t all_pairs = choose_two(pred.size());
    PairwiseAgreement result;
    result.same_both = true_positive;
    result.same_pred_only = predicted_positive - true_positive;
    result.same_reference_only = reference_positive - true_positive;
    result.different_both =
        all_pairs - result.same_both - result.same_pred_only - result.same_reference_only;

    if (predicted_positive > 0) {
        result.precision =
            static_cast<double>(true_positive) / static_cast<double>(predicted_positive);
    }
    if (reference_positive > 0) {
        result.recall =
            static_cast<double>(true_positive) / static_cast<double>(reference_positive);
    }
    const std::uint64_t f1_denominator =
        2 * true_positive + result.same_pred_only + result.same_reference_only;
    result.f1 = f1_denominator == 0 ? 1.0
                                    : 2.0 * static_cast<double>(true_positive) /
                                          static_cast<double>(f1_denominator);

    if (all_pairs == 0) {
        result.adjusted_rand = 1.0;
        return result;
    }

    const long double expected = static_cast<long double>(predicted_positive) *
                                 static_cast<long double>(reference_positive) /
                                 static_cast<long double>(all_pairs);
    const long double maximum =
        0.5L * static_cast<long double>(predicted_positive + reference_positive);
    const long double denominator = maximum - expected;
    if (std::abs(denominator) <= 1e-18L) {
        result.adjusted_rand =
            result.same_pred_only == 0 && result.same_reference_only == 0 ? 1.0 : 0.0;
    } else {
        result.adjusted_rand =
            static_cast<double>((static_cast<long double>(true_positive) - expected) / denominator);
    }
    return result;
}

} // namespace cd::metrics
