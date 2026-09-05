#include "community_detection/experiment/runner.hpp"

#include "community_detection/metrics/modularity.hpp"

#include <chrono>
#include <utility>

namespace cd::experiment {
namespace {

template <typename Callable> AlgorithmResult timed_run(const Graph &graph, Callable &&callable) {
    const auto start = std::chrono::steady_clock::now();
    AlgorithmResult result = callable();
    const auto stop = std::chrono::steady_clock::now();

    result.elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    result.labels = normalize_labels(result.labels);
    result.modularity = metrics::modularity(graph, result.labels);
    result.diagnostics["num_communities"] = static_cast<double>(count_communities(result.labels));
    return result;
}

} // namespace

AlgorithmResult run(const Graph &graph, const girvan_newman::Config &config) {
    return timed_run(graph, [&] { return girvan_newman::run(graph, config); });
}

AlgorithmResult run(const Graph &graph, const spectral::Config &config) {
    return timed_run(graph, [&] { return spectral::run(graph, config); });
}

AlgorithmResult run(const Graph &graph, const louvain::Config &config) {
    return timed_run(graph, [&] { return louvain::run(graph, config); });
}

AlgorithmResult run(const Graph &graph, const label_propagation::Config &config) {
    return timed_run(graph, [&] { return label_propagation::run(graph, config); });
}

} // namespace cd::experiment
