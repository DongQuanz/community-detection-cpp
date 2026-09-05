#include "community_detection/core/graph_transform.hpp"
#include "community_detection/experiment/runner.hpp"
#include "community_detection/io/graph_io.hpp"
#include "community_detection/metrics/external.hpp"
#include "community_detection/metrics/modularity.hpp"
#include "community_detection/metrics/partition_metrics.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error("test failure: " + message);
    }
}

void require_near(double actual, double expected, double tolerance, const std::string &message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error("test failure: " + message + " (actual=" + std::to_string(actual) +
                                 ", expected=" + std::to_string(expected) + ")");
    }
}

cd::Graph make_toy_graph() {
    cd::Graph graph(6);
    graph.reserve_edges(7);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.add_edge(2, 0);
    graph.add_edge(2, 3);
    graph.add_edge(3, 4);
    graph.add_edge(4, 5);
    graph.add_edge(5, 3);
    return graph;
}

bool same_bipartition(const cd::Labels &labels) {
    return labels.size() == 6 && labels[0] == labels[1] && labels[1] == labels[2] &&
           labels[3] == labels[4] && labels[4] == labels[5] && labels[0] != labels[3];
}

void test_graph_contract_and_binary_roundtrip() {
    cd::Graph graph(3);
    graph.add_edge(0, 1);
    graph.add_edge(1, 0, 2.0);
    graph.add_edge(1, 2);
    require(graph.num_edges() == 2, "duplicate undirected edges must merge");
    require_near(graph.edge_weight(0, 1), 3.0, 1e-12, "merged edge weight");
    require_near(graph.weighted_degree(1), 4.0, 1e-12, "weighted degree");

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "community_detection_roundtrip.cdgraph";
    cd::io::write_binary_graph(graph, path);
    const cd::Graph loaded = cd::io::load_binary_graph(path);
    std::filesystem::remove(path);
    require(loaded.num_nodes() == 3 && loaded.num_edges() == 2, "binary graph shape");
    require_near(loaded.edge_weight(0, 1), 3.0, 1e-12, "binary graph weight");

    cd::Graph mutable_copy = loaded;
    require(mutable_copy.remove_edge(1, 0), "remove existing edge");
    require(!mutable_copy.has_edge(0, 1), "removed edge must disappear both ways");
    require(mutable_copy.num_edges() == 1, "edge count after removal");
}

void test_internal_metrics() {
    const cd::Graph graph = make_toy_graph();
    const cd::Labels expected{0, 0, 0, 1, 1, 1};
    const auto metrics = cd::metrics::evaluate_partition(graph, expected);

    require(metrics.num_communities == 2, "community count");
    require(metrics.singleton_communities == 0, "singleton count");
    require(metrics.isolated_nodes == 0, "isolated node count");
    require_near(metrics.modularity, 5.0 / 14.0, 1e-12, "modularity");
    require_near(metrics.coverage, 6.0 / 7.0, 1e-12, "coverage");
    require_near(metrics.ratio_cut, 2.0 / 3.0, 1e-12, "RatioCut");
    require_near(metrics.normalized_cut, 2.0 / 7.0, 1e-12, "Ncut");
    for (const auto &community : metrics.communities) {
        require(community.size == 3, "community size");
        require(community.internal_edges == 3, "internal edges");
        require_near(community.density, 1.0, 1e-12, "density");
        require_near(community.conductance, 1.0 / 7.0, 1e-12, "conductance");
    }
    require_near(metrics.density_distribution.median, 1.0, 1e-12, "density median");
    require_near(cd::metrics::modularity(graph, expected, 2.0), -1.0 / 7.0, 1e-12,
                 "resolution modularity");

    cd::Graph aggregated(2);
    aggregated.add_edge(0, 0, 3.0);
    aggregated.add_edge(1, 1, 3.0);
    aggregated.add_edge(0, 1, 1.0);
    require_near(aggregated.weighted_degree(0), 7.0, 1e-12, "self-loop degree");
    require_near(cd::metrics::modularity(aggregated, {0, 1}), 5.0 / 14.0, 1e-12,
                 "modularity on a Louvain supergraph");
}

void test_external_metrics() {
    const cd::Labels reference{0, 0, 0, 1, 1, 1};
    const cd::Labels relabeled{7, 7, 7, 3, 3, 3};
    const auto exact = cd::metrics::compare_partitions(relabeled, reference);
    require_near(exact.precision, 1.0, 1e-12, "pairwise precision exact");
    require_near(exact.recall, 1.0, 1e-12, "pairwise recall exact");
    require_near(exact.f1, 1.0, 1e-12, "pairwise F1 exact");
    require_near(exact.adjusted_rand, 1.0, 1e-12, "ARI exact");

    const cd::Labels merged(6, 0);
    const auto one_cluster = cd::metrics::compare_partitions(merged, reference);
    require(one_cluster.same_both == 6 && one_cluster.same_pred_only == 9,
            "pair contingency for merged partition");
    require_near(one_cluster.precision, 0.4, 1e-12, "merged pairwise precision");
    require_near(one_cluster.recall, 1.0, 1e-12, "merged pairwise recall");
    require_near(one_cluster.f1, 4.0 / 7.0, 1e-12, "merged pairwise F1");
    require_near(one_cluster.adjusted_rand, 0.0, 1e-12, "merged ARI");
}

void test_brandes_and_girvan_newman() {
    const cd::Graph graph = make_toy_graph();
    const auto scores = cd::girvan_newman::edge_betweenness_brandes(graph);
    require_near(scores.at({2, 3}), 9.0, 1e-12, "bridge betweenness");

    cd::girvan_newman::Config config;
    config.target_communities = 2;
    const auto result = cd::experiment::run(graph, config);
    require(same_bipartition(result.labels), "Girvan-Newman toy partition");
    require(result.hierarchy.size() >= 2, "Girvan-Newman hierarchy");
}

void test_spectral_dense_and_lanczos() {
    const cd::Graph graph = make_toy_graph();
    for (const auto laplacian : {cd::spectral::LaplacianType::Unnormalized,
                                 cd::spectral::LaplacianType::SymmetricNormalized}) {
        cd::spectral::Config dense;
        dense.num_communities = 2;
        dense.laplacian = laplacian;
        dense.bisection = cd::spectral::BisectionMethod::BestSweep;
        dense.dense_solver_threshold = 100;
        const auto dense_result = cd::experiment::run(graph, dense);
        require(same_bipartition(dense_result.labels), "dense spectral toy partition");
        require(dense_result.diagnostics.at("eigensolver_max_residual") < 1e-7,
                "dense eigensolver residual");

        cd::spectral::Config lanczos = dense;
        lanczos.dense_solver_threshold = 0;
        lanczos.lanczos_basis_size = graph.num_nodes();
        const auto lanczos_result = cd::experiment::run(graph, lanczos);
        require(same_bipartition(lanczos_result.labels), "Lanczos spectral toy partition");
        require(lanczos_result.diagnostics.at("eigensolver_max_residual") < 1e-6,
                "Lanczos eigensolver residual");
    }

    // Exercise genuinely partial Lanczos (basis << n), not only the basis=n
    // case that is equivalent to a full decomposition.
    cd::Graph block_graph(300);
    for (cd::NodeId block = 0; block < 3; ++block) {
        const cd::NodeId offset = block * 100;
        for (cd::NodeId local = 0; local < 100; ++local) {
            for (cd::NodeId step = 1; step <= 3; ++step) {
                block_graph.add_edge(offset + local, offset + (local + step) % 100);
            }
        }
    }
    block_graph.add_edge(99, 100);
    block_graph.add_edge(199, 200);

    cd::spectral::Config partial;
    partial.num_communities = 3;
    partial.laplacian = cd::spectral::LaplacianType::SymmetricNormalized;
    partial.dense_solver_threshold = 0;
    partial.lanczos_basis_size = 0;
    partial.kmeans_restarts = 5;
    partial.seed = 20260824;
    const auto partial_result = cd::experiment::run(block_graph, partial);
    const double partial_residual = partial_result.diagnostics.at("eigensolver_max_residual");
    require_near(partial_result.diagnostics.at("eigensolver_basis_size"), 128.0, 0.0,
                 "automatic partial Lanczos basis");
    require(partial_residual < 1e-6,
            "partial Lanczos residual=" + std::to_string(partial_residual));
    for (std::size_t block = 0; block < 3; ++block) {
        const int label = partial_result.labels[block * 100];
        for (std::size_t local = 1; local < 100; ++local) {
            require(partial_result.labels[block * 100 + local] == label,
                    "partial Lanczos must recover each planted block");
        }
    }
    require(partial_result.labels[0] != partial_result.labels[100] &&
                partial_result.labels[0] != partial_result.labels[200] &&
                partial_result.labels[100] != partial_result.labels[200],
            "partial Lanczos must separate planted blocks");
}

void test_louvain_and_monotonicity_regression() {
    const cd::Graph toy = make_toy_graph();
    cd::louvain::Config config;
    config.shuffle_nodes = false;
    const auto result = cd::experiment::run(toy, config);
    require(same_bipartition(result.labels), "Louvain toy partition");
    require_near(result.modularity, 5.0 / 14.0, 1e-12, "Louvain toy Q");

    // Regression for the former bug: comparing candidates with zero could
    // accept a move worse than restoring the old community. Q must not decrease.
    std::mt19937_64 rng(20260824);
    for (const double resolution : {0.5, 1.0, 2.0}) {
        for (std::size_t sample = 0; sample < 100; ++sample) {
            cd::Graph graph(9);
            for (cd::NodeId u = 0; u < 9; ++u) {
                for (cd::NodeId v = u + 1; v < 9; ++v) {
                    if ((rng() % 100) < 28) {
                        graph.add_edge(u, v);
                    }
                }
            }
            cd::louvain::Config random_config;
            random_config.seed = sample;
            random_config.resolution = resolution;
            const auto random_result = cd::louvain::run(graph, random_config);
            cd::Labels singleton(9);
            std::iota(singleton.begin(), singleton.end(), 0);
            double previous = cd::metrics::modularity(graph, singleton, resolution);
            for (const auto &level : random_result.hierarchy) {
                const double current = cd::metrics::modularity(graph, level, resolution);
                require(current + 1e-10 >= previous, "Louvain hierarchy must not decrease Q_gamma");
                previous = current;
            }
        }
    }
}

void test_lpa_and_dropout() {
    const cd::Graph graph = make_toy_graph();
    cd::label_propagation::Config config;
    config.seed = 42;
    config.tie_break = cd::label_propagation::TieBreak::SmallestLabel;
    const auto result = cd::experiment::run(graph, config);
    require(result.labels.size() == graph.num_nodes(), "LPA labels size");
    require(result.diagnostics.at("iterations") <= static_cast<double>(config.max_iterations),
            "LPA iteration cap");

    const auto unchanged = cd::drop_edges(graph, 0.0, 9);
    require(unchanged.removed_edges == 0, "zero dropout removes nothing");
    require(unchanged.graph.num_edges() == graph.num_edges(), "zero dropout edge count");

    const auto first = cd::drop_edges(graph, 0.4, 123);
    const auto second = cd::drop_edges(graph, 0.4, 123);
    const auto first_edges = first.graph.edges();
    const auto second_edges = second.graph.edges();
    require(first.removed_edges == second.removed_edges, "dropout count reproducibility");
    require(first_edges.size() == second_edges.size(), "dropout size reproducibility");
    for (std::size_t i = 0; i < first_edges.size(); ++i) {
        require(first_edges[i].u == second_edges[i].u && first_edges[i].v == second_edges[i].v,
                "dropout edge reproducibility");
    }
}

} // namespace

int main() {
    try {
        test_graph_contract_and_binary_roundtrip();
        test_internal_metrics();
        test_external_metrics();
        test_brandes_and_girvan_newman();
        test_spectral_dense_and_lanczos();
        test_louvain_and_monotonicity_regression();
        test_lpa_and_dropout();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
