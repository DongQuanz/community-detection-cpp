#include "community_detection/experiment/runner.hpp"

#include <iomanip>
#include <iostream>
#include <string>

namespace {

cd::Graph make_toy_graph() {
    cd::Graph graph(6);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.add_edge(2, 0);
    graph.add_edge(2, 3);
    graph.add_edge(3, 4);
    graph.add_edge(4, 5);
    graph.add_edge(5, 3);
    return graph;
}

void print_result(const cd::AlgorithmResult &result) {
    std::cout << std::left << std::setw(20) << result.algorithm
              << " K=" << static_cast<int>(result.diagnostics.at("num_communities"))
              << "  Q=" << std::fixed << std::setprecision(4) << result.modularity
              << "  time=" << std::setprecision(3) << result.elapsed_ms << " ms\n  labels: ";
    for (const int label : result.labels) {
        std::cout << label << ' ';
    }
    std::cout << "\n";
}

} // namespace

int main() {
    const cd::Graph graph = make_toy_graph();

    cd::girvan_newman::Config gn;
    gn.target_communities = 2;

    cd::spectral::Config spectral;
    spectral.num_communities = 2;
    spectral.laplacian = cd::spectral::LaplacianType::Unnormalized;
    spectral.bisection = cd::spectral::BisectionMethod::BestSweep;

    cd::louvain::Config louvain;
    louvain.seed = 42;

    cd::label_propagation::Config lpa;
    lpa.seed = 42;
    lpa.tie_break = cd::label_propagation::TieBreak::SmallestLabel;

    print_result(cd::experiment::run(graph, gn));
    print_result(cd::experiment::run(graph, spectral));
    print_result(cd::experiment::run(graph, louvain));
    print_result(cd::experiment::run(graph, lpa));
    return 0;
}
