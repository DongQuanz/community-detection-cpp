#include "community_detection/algorithms/girvan_newman.hpp"
#include "community_detection/algorithms/label_propagation.hpp"
#include "community_detection/algorithms/louvain.hpp"
#include "community_detection/algorithms/spectral.hpp"
#include "community_detection/core/graph_transform.hpp"
#include "community_detection/experiment/runner.hpp"
#include "community_detection/io/graph_io.hpp"
#include "community_detection/metrics/external.hpp"
#include "community_detection/metrics/partition_metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

using Arguments = std::map<std::string, std::string>;

std::string usage() {
    return R"(community_detect --graph FILE --algorithm NAME --output-dir DIR [options]

Algorithms: girvan_newman, spectral, louvain, label_propagation
Common options:
  --dataset NAME             Dataset identifier recorded in output
  --reference FILE           Reference labels TSV (optional)
  --seed N                   Random seed (default: 42)
  --k N                      Required for spectral; optional GN stopping K
  --resolution X             Q_gamma/Louvain gamma (default: 1)
  --dropout-rate X           Drop edge probability for robustness run
  --dropout-seed N           Seed for deterministic edge dropout
  --force                    Bypass exact-GN/spectral work guards

Spectral options:
  --laplacian normalized|unnormalized  (default: normalized)
  --bisection sweep|sign               (default: sweep)
  --dense-threshold N                  (default: 256)
  --lanczos-basis N                    (0 = automatic)
  --kmeans-restarts N                  (default: 20)
  --max-eigensolver-residual X         Reject inaccurate eigenpairs (default: 1e-4)
  --spectral-max-edge-visits N         Work guard (default: 2000000000)
  --spectral-max-orthogonalization-ops N  Work guard (default: 5000000000)

GN safety options:
  --gn-max-n N               Default 1000
  --gn-max-m N               Default 10000
)";
}

Arguments parse_arguments(int argc, char **argv) {
    Arguments arguments;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            arguments["--help"] = "true";
            continue;
        }
        if (key.rfind("--", 0) != 0) {
            throw std::invalid_argument("Unexpected positional argument: " + key);
        }
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            arguments[key] = argv[++i];
        } else {
            arguments[key] = "true";
        }
    }

    const std::set<std::string> accepted{"--help",
                                         "--graph",
                                         "--algorithm",
                                         "--output-dir",
                                         "--dataset",
                                         "--reference",
                                         "--seed",
                                         "--k",
                                         "--resolution",
                                         "--dropout-rate",
                                         "--dropout-seed",
                                         "--force",
                                         "--laplacian",
                                         "--bisection",
                                         "--dense-threshold",
                                         "--lanczos-basis",
                                         "--kmeans-restarts",
                                         "--kmeans-max-iterations",
                                         "--eigensolver-tolerance",
                                         "--max-eigensolver-residual",
                                         "--gn-max-n",
                                         "--gn-max-m",
                                         "--gn-random-ties",
                                         "--spectral-max-edge-visits",
                                         "--spectral-max-orthogonalization-ops",
                                         "--lpa-synchronous",
                                         "--lpa-smallest-tie",
                                         "--max-iterations"};
    for (const auto &[key, _] : arguments) {
        if (accepted.find(key) == accepted.end()) {
            throw std::invalid_argument("Unknown option: " + key);
        }
    }
    return arguments;
}

std::string value_or(const Arguments &arguments, const std::string &key, std::string fallback) {
    const auto found = arguments.find(key);
    return found == arguments.end() ? std::move(fallback) : found->second;
}

std::string required(const Arguments &arguments, const std::string &key) {
    const auto found = arguments.find(key);
    if (found == arguments.end() || found->second == "true") {
        throw std::invalid_argument("Missing required option " + key);
    }
    return found->second;
}

bool flag(const Arguments &arguments, const std::string &key) {
    return arguments.find(key) != arguments.end();
}

std::uint64_t parse_u64(const std::string &text, const std::string &name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument("Invalid nonnegative integer for " + name + ": " + text);
    }
    std::size_t consumed = 0;
    const std::uint64_t value = std::stoull(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("Invalid integer for " + name + ": " + text);
    }
    return value;
}

std::size_t parse_size(const std::string &text, const std::string &name) {
    const std::uint64_t value = parse_u64(text, name);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::out_of_range(name + " is too large");
    }
    return static_cast<std::size_t>(value);
}

double parse_double(const std::string &text, const std::string &name) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("Invalid number for " + name + ": " + text);
    }
    return value;
}

std::string json_escape(const std::string &text) {
    std::ostringstream output;
    for (const unsigned char character : text) {
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20U) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(character) << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    return output.str();
}

void write_number(std::ostream &output, double value) {
    if (std::isfinite(value)) {
        output << std::setprecision(17) << value;
    } else {
        output << "null";
    }
}

void write_distribution(std::ostream &output, const cd::metrics::DistributionSummary &summary) {
    output << "{\"valid_count\":" << summary.valid_count << ",\"min\":";
    write_number(output, summary.minimum);
    output << ",\"q1\":";
    write_number(output, summary.q1);
    output << ",\"median\":";
    write_number(output, summary.median);
    output << ",\"q3\":";
    write_number(output, summary.q3);
    output << ",\"max\":";
    write_number(output, summary.maximum);
    output << ",\"community_mean\":";
    write_number(output, summary.community_mean);
    output << ",\"node_weighted_mean\":";
    write_number(output, summary.node_weighted_mean);
    output << '}';
}

double peak_rss_mb() {
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
#else
    return std::numeric_limits<double>::quiet_NaN();
#endif
}

void write_communities_csv(const cd::metrics::PartitionMetrics &metrics,
                           const std::filesystem::path &path) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot create " + path.string());
    }
    output << "community,size,internal_edges,internal_weight,boundary_weight,"
              "volume,density,conductance\n";
    output << std::setprecision(17);
    for (const auto &community : metrics.communities) {
        output << community.community << ',' << community.size << ',' << community.internal_edges
               << ',' << community.internal_weight << ',' << community.boundary_weight << ','
               << community.volume << ',';
        if (std::isfinite(community.density))
            output << community.density;
        output << ',';
        if (std::isfinite(community.conductance))
            output << community.conductance;
        output << '\n';
    }
}

cd::AlgorithmResult run_algorithm(const cd::Graph &graph, const Arguments &arguments,
                                  const std::string &algorithm, std::uint64_t seed, std::size_t k,
                                  double resolution) {
    if (algorithm == "girvan_newman") {
        const std::size_t max_n = parse_size(value_or(arguments, "--gn-max-n", "1000"), "gn-max-n");
        const std::size_t max_m =
            parse_size(value_or(arguments, "--gn-max-m", "10000"), "gn-max-m");
        if (!flag(arguments, "--force") &&
            (graph.num_nodes() > max_n || graph.num_edges() > max_m)) {
            throw std::runtime_error(
                "Exact Girvan-Newman blocked by safety guard; use the registered sample "
                "protocol or pass --force deliberately");
        }
        cd::girvan_newman::Config config;
        config.target_communities = k;
        config.seed = seed;
        config.tie_break = flag(arguments, "--gn-random-ties")
                               ? cd::girvan_newman::TieBreak::Random
                               : cd::girvan_newman::TieBreak::Lexicographic;
        return cd::experiment::run(graph, config);
    }
    if (algorithm == "spectral") {
        if (k == 0) {
            throw std::invalid_argument("--k is required for spectral clustering");
        }
        cd::spectral::Config config;
        config.num_communities = k;
        config.seed = seed;
        config.kmeans_restarts =
            parse_size(value_or(arguments, "--kmeans-restarts", "20"), "kmeans-restarts");
        config.kmeans_max_iterations = parse_size(
            value_or(arguments, "--kmeans-max-iterations", "300"), "kmeans-max-iterations");
        config.dense_solver_threshold =
            parse_size(value_or(arguments, "--dense-threshold", "256"), "dense-threshold");
        config.lanczos_basis_size =
            parse_size(value_or(arguments, "--lanczos-basis", "0"), "lanczos-basis");
        config.eigensolver_tolerance = parse_double(
            value_or(arguments, "--eigensolver-tolerance", "1e-9"), "eigensolver-tolerance");
        config.maximum_eigensolver_residual = parse_double(
            value_or(arguments, "--max-eigensolver-residual", "1e-4"), "max-eigensolver-residual");
        const std::string laplacian = value_or(arguments, "--laplacian", "normalized");
        if (laplacian == "normalized") {
            config.laplacian = cd::spectral::LaplacianType::SymmetricNormalized;
        } else if (laplacian == "unnormalized") {
            config.laplacian = cd::spectral::LaplacianType::Unnormalized;
        } else {
            throw std::invalid_argument("--laplacian must be normalized or unnormalized");
        }
        const std::string bisection = value_or(arguments, "--bisection", "sweep");
        if (bisection == "sweep") {
            config.bisection = cd::spectral::BisectionMethod::BestSweep;
        } else if (bisection == "sign") {
            config.bisection = cd::spectral::BisectionMethod::Sign;
        } else {
            throw std::invalid_argument("--bisection must be sweep or sign");
        }

        const std::size_t raw_basis = config.lanczos_basis_size == 0
                                          ? cd::spectral::automatic_lanczos_basis_size(k)
                                          : config.lanczos_basis_size;
        const std::size_t basis = std::min(raw_basis, graph.num_nodes());
        const long double estimated_edge_visits =
            2.0L * static_cast<long double>(graph.num_edges()) * static_cast<long double>(basis);
        const long double estimated_orthogonalization_ops =
            4.0L * static_cast<long double>(graph.num_nodes()) * static_cast<long double>(basis) *
            static_cast<long double>(basis);
        const long double maximum_edge_visits = static_cast<long double>(
            parse_size(value_or(arguments, "--spectral-max-edge-visits", "2000000000"),
                       "spectral-max-edge-visits"));
        const long double maximum_orthogonalization_ops = static_cast<long double>(
            parse_size(value_or(arguments, "--spectral-max-orthogonalization-ops", "5000000000"),
                       "spectral-max-orthogonalization-ops"));
        if (!flag(arguments, "--force") && graph.num_nodes() > config.dense_solver_threshold &&
            (estimated_edge_visits > maximum_edge_visits ||
             estimated_orthogonalization_ops > maximum_orthogonalization_ops)) {
            throw std::runtime_error(
                "Spectral run blocked by the registered edge/orthogonalization work guard; "
                "use a sample or pass --force");
        }
        return cd::experiment::run(graph, config);
    }
    if (algorithm == "louvain") {
        cd::louvain::Config config;
        config.seed = seed;
        config.resolution = resolution;
        return cd::experiment::run(graph, config);
    }
    if (algorithm == "label_propagation") {
        cd::label_propagation::Config config;
        config.seed = seed;
        config.asynchronous = !flag(arguments, "--lpa-synchronous");
        config.tie_break = flag(arguments, "--lpa-smallest-tie")
                               ? cd::label_propagation::TieBreak::SmallestLabel
                               : cd::label_propagation::TieBreak::Random;
        config.max_iterations =
            parse_size(value_or(arguments, "--max-iterations", "100"), "max-iterations");
        return cd::experiment::run(graph, config);
    }
    throw std::invalid_argument("Unknown algorithm: " + algorithm);
}

void write_summary_json(const std::filesystem::path &path, const std::string &dataset,
                        const std::string &algorithm, std::uint64_t seed, const cd::Graph &graph,
                        const cd::AlgorithmResult &result,
                        const cd::metrics::PartitionMetrics &metrics,
                        const cd::metrics::PairwiseAgreement *external, double metrics_elapsed_ms,
                        double dropout_rate, std::uint64_t dropout_seed, std::size_t removed_edges,
                        const std::string &laplacian, std::size_t requested_k) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot create " + path.string());
    }
    output << "{\n  \"schema_version\":1,\n"
           << "  \"status\":\"ok\",\n"
           << "  \"dataset\":\"" << json_escape(dataset) << "\",\n"
           << "  \"algorithm\":\"" << json_escape(algorithm) << "\",\n"
           << "  \"seed\":" << seed << ",\n"
           << "  \"n\":" << graph.num_nodes() << ",\n"
           << "  \"m\":" << graph.num_edges() << ",\n"
           << "  \"requested_k\":" << requested_k << ",\n"
           << "  \"num_communities\":" << metrics.num_communities << ",\n"
           << "  \"elapsed_ms\":";
    write_number(output, result.elapsed_ms);
    output << ",\n  \"metrics_elapsed_ms\":";
    write_number(output, metrics_elapsed_ms);
    output << ",\n  \"peak_rss_mb\":";
    write_number(output, peak_rss_mb());
    output << ",\n  \"objective\":";
    write_number(output, result.objective);
    output << ",\n  \"modularity_q1\":";
    write_number(output, result.modularity);
    output << ",\n  \"dropout\":{\"rate\":" << dropout_rate << ",\"seed\":" << dropout_seed
           << ",\"removed_edges\":" << removed_edges << "},\n"
           << "  \"parameters\":{\"resolution\":" << metrics.resolution << ",\"laplacian\":\""
           << json_escape(laplacian) << "\"},\n"
           << "  \"internal\":{\"coverage\":";
    write_number(output, metrics.coverage);
    output << ",\"ratio_cut\":";
    write_number(output, metrics.ratio_cut);
    output << ",\"normalized_cut\":";
    write_number(output, metrics.normalized_cut);
    output << ",\"modularity_q_gamma\":";
    write_number(output, metrics.modularity);
    output << ",\"singletons\":" << metrics.singleton_communities
           << ",\"isolated_nodes\":" << metrics.isolated_nodes << ",\"size_distribution\":";
    write_distribution(output, metrics.size_distribution);
    output << ",\"density_distribution\":";
    write_distribution(output, metrics.density_distribution);
    output << ",\"conductance_distribution\":";
    write_distribution(output, metrics.conductance_distribution);
    output << "},\n  \"external\":";
    if (external == nullptr) {
        output << "null";
    } else {
        output << "{\"a\":" << external->same_both << ",\"b\":" << external->same_pred_only
               << ",\"c\":" << external->same_reference_only
               << ",\"d\":" << external->different_both << ",\"pairwise_precision\":";
        write_number(output, external->precision);
        output << ",\"pairwise_recall\":";
        write_number(output, external->recall);
        output << ",\"pairwise_f1\":";
        write_number(output, external->f1);
        output << ",\"ari\":";
        write_number(output, external->adjusted_rand);
        output << '}';
    }
    output << ",\n  \"diagnostics\":{";
    std::vector<std::pair<std::string, double>> diagnostics(result.diagnostics.begin(),
                                                            result.diagnostics.end());
    std::sort(diagnostics.begin(), diagnostics.end());
    for (std::size_t i = 0; i < diagnostics.size(); ++i) {
        if (i > 0)
            output << ',';
        output << "\"" << json_escape(diagnostics[i].first) << "\":";
        write_number(output, diagnostics[i].second);
    }
    output << "}\n}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        if (flag(arguments, "--help")) {
            std::cout << usage();
            return 0;
        }

        const std::filesystem::path graph_path = required(arguments, "--graph");
        const std::filesystem::path output_directory = required(arguments, "--output-dir");
        const std::string algorithm = required(arguments, "--algorithm");
        const std::string dataset = value_or(arguments, "--dataset", graph_path.stem().string());
        const std::uint64_t seed = parse_u64(value_or(arguments, "--seed", "42"), "seed");
        const std::size_t k = parse_size(value_or(arguments, "--k", "0"), "k");
        const double resolution =
            parse_double(value_or(arguments, "--resolution", "1.0"), "resolution");
        const double dropout_rate =
            parse_double(value_or(arguments, "--dropout-rate", "0.0"), "dropout-rate");
        const std::uint64_t dropout_seed =
            parse_u64(value_or(arguments, "--dropout-seed", "0"), "dropout-seed");

        cd::Graph graph = cd::io::load_binary_graph(graph_path);
        std::size_t removed_edges = 0;
        if (dropout_rate > 0.0) {
            auto perturbed = cd::drop_edges(graph, dropout_rate, dropout_seed);
            graph = std::move(perturbed.graph);
            removed_edges = perturbed.removed_edges;
        }

        cd::Labels reference;
        const auto reference_path = arguments.find("--reference");
        if (reference_path != arguments.end()) {
            reference = cd::io::load_labels_tsv(reference_path->second, graph.num_nodes());
        }

        const cd::AlgorithmResult result =
            run_algorithm(graph, arguments, algorithm, seed, k, resolution);
        const auto metrics_start = std::chrono::steady_clock::now();
        const cd::metrics::PartitionMetrics metrics =
            cd::metrics::evaluate_partition(graph, result.labels, resolution);
        cd::metrics::PairwiseAgreement external;
        const cd::metrics::PairwiseAgreement *external_pointer = nullptr;
        if (!reference.empty()) {
            external = cd::metrics::compare_partitions(result.labels, reference);
            external_pointer = &external;
        }
        const auto metrics_stop = std::chrono::steady_clock::now();
        const double metrics_elapsed_ms =
            std::chrono::duration<double, std::milli>(metrics_stop - metrics_start).count();

        std::filesystem::create_directories(output_directory);
        cd::io::write_labels_tsv(result.labels, output_directory / "partition.tsv");
        write_communities_csv(metrics, output_directory / "communities.csv");
        write_summary_json(output_directory / "summary.json", dataset, algorithm, seed, graph,
                           result, metrics, external_pointer, metrics_elapsed_ms, dropout_rate,
                           dropout_seed, removed_edges,
                           value_or(arguments, "--laplacian", "not_applicable"), k);
        std::cout << (output_directory / "summary.json").string() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "community_detect: " << error.what() << '\n';
        return 2;
    }
}
