#include "community_detection/io/graph_io.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cd::io {
namespace {

constexpr std::array<char, 8> kMagic{'C', 'D', 'G', 'R', 'P', 'H', '1', '\0'};
constexpr std::streamoff kHeaderSize = 8 + 8 + 8 + 1 + 7;

void require_stream(const std::ios &stream, const std::string &message) {
    if (!stream) {
        throw std::runtime_error(message);
    }
}

std::uint32_t read_u32(std::istream &input) {
    std::array<unsigned char, 4> bytes{};
    input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    require_stream(input, "Unexpected end of CDGRPH1 file");
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint64_t read_u64(std::istream &input) {
    std::array<unsigned char, 8> bytes{};
    input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    require_stream(input, "Unexpected end of CDGRPH1 file");
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
    }
    return value;
}

double read_f64(std::istream &input) {
    const std::uint64_t bits = read_u64(input);
    double value = 0.0;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void write_u32(std::ostream &output, std::uint32_t value) {
    std::array<unsigned char, 4> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (8U * i)) & 0xffU);
    }
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

void write_u64(std::ostream &output, std::uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (8U * i)) & 0xffU);
    }
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

void write_f64(std::ostream &output, double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    write_u64(output, bits);
}

struct Header {
    std::size_t nodes{};
    std::uint64_t edges{};
    bool weighted{};
};

Header read_header(std::istream &input) {
    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    require_stream(input, "Cannot read CDGRPH1 header");
    if (magic != kMagic) {
        throw std::runtime_error("Not a CDGRPH1 graph file");
    }
    const std::uint64_t raw_nodes = read_u64(input);
    const std::uint64_t raw_edges = read_u64(input);
    char weighted = 0;
    input.read(&weighted, 1);
    std::array<char, 7> reserved{};
    input.read(reserved.data(), reserved.size());
    require_stream(input, "Truncated CDGRPH1 header");
    if (weighted != 0 && weighted != 1) {
        throw std::runtime_error("Invalid CDGRPH1 weighted flag");
    }
    if (raw_nodes >= std::numeric_limits<NodeId>::max() ||
        raw_nodes > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("CDGRPH1 node count exceeds this build's limits");
    }
    return {static_cast<std::size_t>(raw_nodes), raw_edges, weighted == 1};
}

WeightedEdge read_record(std::istream &input, bool weighted) {
    const NodeId u = read_u32(input);
    const NodeId v = read_u32(input);
    const double weight = weighted ? read_f64(input) : 1.0;
    return {u, v, weight};
}

} // namespace

Graph load_binary_graph(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    require_stream(input, "Cannot open graph file: " + path.string());
    const Header header = read_header(input);

    std::vector<std::uint64_t> offsets(header.nodes + 1, 0);
    NodeId previous_u = 0;
    NodeId previous_v = 0;
    bool has_previous = false;
    for (std::uint64_t i = 0; i < header.edges; ++i) {
        const WeightedEdge edge = read_record(input, header.weighted);
        if (edge.u > edge.v || static_cast<std::size_t>(edge.v) >= header.nodes) {
            throw std::runtime_error("CDGRPH1 edge is not canonical or is out of range");
        }
        if (!(edge.weight > 0.0) || !std::isfinite(edge.weight)) {
            throw std::runtime_error("CDGRPH1 contains an invalid edge weight");
        }
        if (has_previous &&
            (edge.u < previous_u || (edge.u == previous_u && edge.v <= previous_v))) {
            throw std::runtime_error("CDGRPH1 edges must be strictly sorted and unique");
        }
        has_previous = true;
        previous_u = edge.u;
        previous_v = edge.v;
        ++offsets[static_cast<std::size_t>(edge.u) + 1];
        if (edge.u != edge.v) {
            ++offsets[static_cast<std::size_t>(edge.v) + 1];
        }
    }
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        offsets[i] += offsets[i - 1];
    }

    std::vector<NodeId> neighbors(static_cast<std::size_t>(offsets.back()));
    std::vector<double> weights;
    if (header.weighted) {
        weights.resize(neighbors.size());
    }
    std::vector<std::uint64_t> cursor = offsets;
    input.clear();
    input.seekg(kHeaderSize, std::ios::beg);
    require_stream(input, "Cannot seek in CDGRPH1 file");

    for (std::uint64_t i = 0; i < header.edges; ++i) {
        const WeightedEdge edge = read_record(input, header.weighted);
        const auto put = [&](NodeId from, NodeId to) {
            const std::size_t position = static_cast<std::size_t>(cursor[from]++);
            neighbors[position] = to;
            if (header.weighted) {
                weights[position] = edge.weight;
            }
        };
        put(edge.u, edge.v);
        if (edge.u != edge.v) {
            put(edge.v, edge.u);
        }
    }
    return Graph::from_csr(header.nodes, std::move(offsets), std::move(neighbors),
                           std::move(weights));
}

void write_binary_graph(const Graph &graph, const std::filesystem::path &path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require_stream(output, "Cannot create graph file: " + path.string());
    output.write(kMagic.data(), kMagic.size());
    write_u64(output, graph.num_nodes());
    write_u64(output, graph.num_edges());
    const bool weighted = graph.is_weighted();
    output.put(weighted ? 1 : 0);
    const std::array<char, 7> reserved{};
    output.write(reserved.data(), reserved.size());

    graph.for_each_edge([&](const WeightedEdge &edge) {
        write_u32(output, edge.u);
        write_u32(output, edge.v);
        if (weighted) {
            write_f64(output, edge.weight);
        }
    });
    require_stream(output, "Failed while writing graph file: " + path.string());
}

Labels load_labels_tsv(const std::filesystem::path &path, std::size_t expected_nodes) {
    std::ifstream input(path);
    require_stream(input, "Cannot open labels file: " + path.string());
    Labels labels(expected_nodes, std::numeric_limits<int>::min());
    std::string line;
    std::size_t implicit_node = 0;
    std::size_t assigned = 0;

    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::istringstream parser(line);
        std::string first;
        std::string second;
        parser >> first;
        if (first == "node_id" || first == "node") {
            continue;
        }
        if (!(parser >> second)) {
            if (implicit_node >= expected_nodes) {
                throw std::runtime_error("Labels file has too many rows");
            }
            labels[implicit_node++] = std::stoi(first);
            ++assigned;
            continue;
        }

        const std::size_t node = std::stoull(first);
        if (node >= expected_nodes || labels[node] != std::numeric_limits<int>::min()) {
            throw std::runtime_error("Labels file contains an invalid or duplicate node id");
        }
        labels[node] = std::stoi(second);
        ++assigned;
    }

    if (assigned != expected_nodes) {
        throw std::runtime_error("Labels file does not assign every graph node");
    }
    return normalize_labels(labels);
}

void write_labels_tsv(const Labels &labels, const std::filesystem::path &path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::trunc);
    require_stream(output, "Cannot create labels file: " + path.string());
    output << "node_id\tlabel\n";
    for (std::size_t node = 0; node < labels.size(); ++node) {
        output << node << '\t' << labels[node] << '\n';
    }
    require_stream(output, "Failed while writing labels file: " + path.string());
}

} // namespace cd::io
