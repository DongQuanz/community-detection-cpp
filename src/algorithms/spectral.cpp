#include "community_detection/algorithms/spectral.hpp"

#include "community_detection/core/partition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cd::spectral {

std::size_t automatic_lanczos_basis_size(std::size_t num_communities) {
    constexpr std::size_t multiplier = 6;
    constexpr std::size_t padding = 32;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (num_communities > (maximum - padding) / multiplier) {
        return std::numeric_limits<std::size_t>::max();
    }
    return std::max<std::size_t>(128, multiplier * num_communities + padding);
}

namespace {

struct DenseMatrix {
    std::size_t rows{};
    std::size_t cols{};
    std::vector<double> values;

    DenseMatrix() = default;
    DenseMatrix(std::size_t row_count, std::size_t column_count, double value = 0.0)
        : rows(row_count), cols(column_count), values(row_count * column_count, value) {}

    double &operator()(std::size_t row, std::size_t column) { return values[row * cols + column]; }
    double operator()(std::size_t row, std::size_t column) const {
        return values[row * cols + column];
    }
};

struct EigenSystem {
    /** Laplacian eigenvalues in ascending order. */
    std::vector<double> values;
    /** Row-major eigenvectors: vectors[node, eigenvector]. */
    DenseMatrix vectors;
    std::size_t basis_size{};
    double max_residual{0.0};
    bool used_dense_solver{false};
};

void validate_graph(const Graph &graph) {
    bool has_loop = false;
    graph.for_each_edge([&](const WeightedEdge &edge) { has_loop = has_loop || edge.u == edge.v; });
    if (has_loop) {
        throw std::invalid_argument("Spectral clustering input must not contain self-loops");
    }
}

double dot(const double *left, const double *right, std::size_t size) {
    long double total = 0.0L;
    for (std::size_t i = 0; i < size; ++i) {
        total += static_cast<long double>(left[i]) * right[i];
    }
    return static_cast<double>(total);
}

double norm(const double *vector, std::size_t size) {
    return std::sqrt(std::max(0.0, dot(vector, vector, size)));
}

void axpy(double coefficient, const double *x, double *y, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        y[i] += coefficient * x[i];
    }
}

class ShiftedLaplacianOperator {
  public:
    ShiftedLaplacianOperator(const Graph &graph, LaplacianType type)
        : graph_(graph), type_(type), degree_(graph.num_nodes(), 0.0) {
        for (std::size_t raw_node = 0; raw_node < graph.num_nodes(); ++raw_node) {
            degree_[raw_node] = graph.weighted_degree(static_cast<NodeId>(raw_node));
        }
        shift_ = type == LaplacianType::Unnormalized
                     ? std::max(1.0, 2.0 * graph.max_weighted_degree())
                     : 1.0;
    }

    [[nodiscard]] std::size_t size() const noexcept { return graph_.num_nodes(); }
    [[nodiscard]] double shift() const noexcept { return shift_; }
    [[nodiscard]] const std::vector<double> &degree() const noexcept { return degree_; }

    /**
     * Multiplication by B preserves eigenvectors and reverses spectral order:
     * - normalized: B = D^{-1/2} A D^{-1/2} = I - L_sym;
     * - unnormalized: B = shift I - L.
     * The largest eigenvalues of B correspond to the smallest Laplacian values.
     */
    void perform(const double *input, double *output) const {
        const std::size_t n = size();
        std::fill(output, output + n, 0.0);
        if (type_ == LaplacianType::Unnormalized) {
            for (std::size_t raw_u = 0; raw_u < n; ++raw_u) {
                const NodeId u = static_cast<NodeId>(raw_u);
                output[raw_u] = (shift_ - degree_[raw_u]) * input[raw_u];
                for (const Neighbor neighbor : graph_.neighbors(u)) {
                    output[raw_u] += neighbor.weight * input[neighbor.node];
                }
            }
            return;
        }

        for (std::size_t raw_u = 0; raw_u < n; ++raw_u) {
            if (degree_[raw_u] <= 0.0) {
                continue;
            }
            const NodeId u = static_cast<NodeId>(raw_u);
            const double left_scale = 1.0 / std::sqrt(degree_[raw_u]);
            for (const Neighbor neighbor : graph_.neighbors(u)) {
                if (degree_[neighbor.node] <= 0.0) {
                    continue;
                }
                output[raw_u] += neighbor.weight * left_scale * input[neighbor.node] /
                                 std::sqrt(degree_[neighbor.node]);
            }
        }
    }

    [[nodiscard]] double laplacian_value(double shifted_value) const noexcept {
        return shift_ - shifted_value;
    }

  private:
    const Graph &graph_;
    LaplacianType type_;
    std::vector<double> degree_;
    double shift_{1.0};
};

/** Dense symmetric Jacobi solver for small graphs and Lanczos Ritz matrices. */
std::pair<std::vector<double>, DenseMatrix> jacobi_eigendecomposition(DenseMatrix matrix,
                                                                      double tolerance) {
    if (matrix.rows != matrix.cols) {
        throw std::invalid_argument("Jacobi eigensolver requires a square matrix");
    }
    const std::size_t n = matrix.rows;
    DenseMatrix eigenvectors(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvectors(i, i) = 1.0;
    }
    if (n < 2) {
        return {{n == 0 ? 0.0 : matrix(0, 0)}, std::move(eigenvectors)};
    }

    double scale = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        scale = std::max(scale, std::abs(matrix(i, i)));
    }
    const double threshold = tolerance * scale;
    const std::size_t max_sweeps = 100;

    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double maximum_off_diagonal = 0.0;
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = matrix(p, q);
                maximum_off_diagonal = std::max(maximum_off_diagonal, std::abs(apq));
                if (std::abs(apq) <= threshold) {
                    continue;
                }

                const double app = matrix(p, p);
                const double aqq = matrix(q, q);
                const double tau = (aqq - app) / (2.0 * apq);
                const double tangent = tau >= 0.0 ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                                  : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = tangent * cosine;

                for (std::size_t k = 0; k < n; ++k) {
                    if (k == p || k == q) {
                        continue;
                    }
                    const double akp = matrix(k, p);
                    const double akq = matrix(k, q);
                    matrix(k, p) = cosine * akp - sine * akq;
                    matrix(p, k) = matrix(k, p);
                    matrix(k, q) = sine * akp + cosine * akq;
                    matrix(q, k) = matrix(k, q);
                }
                matrix(p, p) = app - tangent * apq;
                matrix(q, q) = aqq + tangent * apq;
                matrix(p, q) = 0.0;
                matrix(q, p) = 0.0;

                for (std::size_t k = 0; k < n; ++k) {
                    const double vkp = eigenvectors(k, p);
                    const double vkq = eigenvectors(k, q);
                    eigenvectors(k, p) = cosine * vkp - sine * vkq;
                    eigenvectors(k, q) = sine * vkp + cosine * vkq;
                }
            }
        }
        if (maximum_off_diagonal <= threshold) {
            break;
        }
        if (sweep + 1 == max_sweeps) {
            throw std::runtime_error("Jacobi eigensolver did not converge");
        }
    }

    std::vector<double> eigenvalues(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = matrix(i, i);
    }
    return {std::move(eigenvalues), std::move(eigenvectors)};
}

std::vector<std::size_t> descending_order(const std::vector<double> &values) {
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        if (values[left] != values[right]) {
            return values[left] > values[right];
        }
        return left < right;
    });
    return order;
}

double eigen_residual(const ShiftedLaplacianOperator &op, const double *vector, double eigenvalue) {
    std::vector<double> product(op.size(), 0.0);
    op.perform(vector, product.data());
    for (std::size_t i = 0; i < op.size(); ++i) {
        product[i] -= eigenvalue * vector[i];
    }
    return norm(product.data(), product.size()) / std::max(1.0, std::abs(eigenvalue));
}

EigenSystem dense_eigenpairs(const ShiftedLaplacianOperator &op, std::size_t requested,
                             double tolerance) {
    const std::size_t n = op.size();
    DenseMatrix matrix(n, n, 0.0);
    std::vector<double> basis(n, 0.0);
    std::vector<double> product(n, 0.0);
    for (std::size_t column = 0; column < n; ++column) {
        std::fill(basis.begin(), basis.end(), 0.0);
        basis[column] = 1.0;
        op.perform(basis.data(), product.data());
        for (std::size_t row = 0; row < n; ++row) {
            matrix(row, column) = product[row];
        }
    }

    auto [shifted_values, shifted_vectors] =
        jacobi_eigendecomposition(std::move(matrix), std::max(1e-14, tolerance * 0.01));
    const auto order = descending_order(shifted_values);

    EigenSystem result;
    result.values.resize(requested);
    result.vectors = DenseMatrix(n, requested, 0.0);
    result.basis_size = n;
    result.used_dense_solver = true;
    for (std::size_t column = 0; column < requested; ++column) {
        const std::size_t source = order[column];
        result.values[column] = op.laplacian_value(shifted_values[source]);
        for (std::size_t row = 0; row < n; ++row) {
            result.vectors(row, column) = shifted_vectors(row, source);
        }
    }

    // eigen_residual requires a contiguous vector; results are row-major.
    std::vector<double> vector(n, 0.0);
    for (std::size_t column = 0; column < requested; ++column) {
        const std::size_t source = order[column];
        for (std::size_t row = 0; row < n; ++row) {
            vector[row] = result.vectors(row, column);
        }
        result.max_residual = std::max(result.max_residual,
                                       eigen_residual(op, vector.data(), shifted_values[source]));
    }
    return result;
}

void random_orthogonal_vector(std::vector<double> &vector, const std::vector<double> &basis,
                              std::size_t n, std::size_t columns, std::mt19937_64 &rng,
                              double tolerance) {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (std::size_t attempt = 0; attempt < 16; ++attempt) {
        for (double &value : vector) {
            value = distribution(rng);
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t column = 0; column < columns; ++column) {
                const double *q = basis.data() + column * n;
                axpy(-dot(q, vector.data(), n), q, vector.data(), n);
            }
        }
        const double length = norm(vector.data(), n);
        if (length > tolerance) {
            for (double &value : vector) {
                value /= length;
            }
            return;
        }
    }
    throw std::runtime_error("Cannot construct a new orthogonal Lanczos vector");
}

EigenSystem lanczos_eigenpairs(const ShiftedLaplacianOperator &op, std::size_t requested,
                               const Config &config) {
    const std::size_t n = op.size();
    std::size_t basis_size = config.lanczos_basis_size;
    if (basis_size == 0) {
        basis_size = automatic_lanczos_basis_size(requested);
    }
    basis_size = std::min(basis_size, n);
    if (basis_size < requested || (basis_size == requested && basis_size < n)) {
        throw std::invalid_argument("Lanczos basis size must be greater than K (or equal to n)");
    }

    // Store Q by column for contiguous orthogonalization and Ritz combinations.
    std::vector<double> q(n * basis_size, 0.0);
    std::vector<double> work(n, 0.0);
    std::vector<double> random_vector(n, 0.0);
    std::vector<double> alpha(basis_size, 0.0);
    std::vector<double> beta(basis_size > 0 ? basis_size - 1 : 0, 0.0);
    std::mt19937_64 rng(config.seed ^ 0x9e3779b97f4a7c15ULL);
    random_orthogonal_vector(random_vector, q, n, 0, rng, config.eigensolver_tolerance);
    std::copy(random_vector.begin(), random_vector.end(), q.begin());

    for (std::size_t column = 0; column < basis_size; ++column) {
        double *current = q.data() + column * n;
        op.perform(current, work.data());
        if (column > 0) {
            axpy(-beta[column - 1], q.data() + (column - 1) * n, work.data(), n);
        }
        alpha[column] = dot(current, work.data(), n);
        axpy(-alpha[column], current, work.data(), n);

        // Two MGS passes control loss of orthogonality in classical Lanczos.
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t previous = 0; previous <= column; ++previous) {
                const double *vector = q.data() + previous * n;
                axpy(-dot(vector, work.data(), n), vector, work.data(), n);
            }
        }

        if (column + 1 == basis_size) {
            break;
        }
        beta[column] = norm(work.data(), n);
        double *next = q.data() + (column + 1) * n;
        if (beta[column] <= config.eigensolver_tolerance) {
            // Degenerate-spectrum breakdown: start an independent Krylov block.
            beta[column] = 0.0;
            random_orthogonal_vector(random_vector, q, n, column + 1, rng,
                                     config.eigensolver_tolerance);
            std::copy(random_vector.begin(), random_vector.end(), next);
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                next[i] = work[i] / beta[column];
            }
        }
    }

    DenseMatrix tridiagonal(basis_size, basis_size, 0.0);
    for (std::size_t i = 0; i < basis_size; ++i) {
        tridiagonal(i, i) = alpha[i];
        if (i + 1 < basis_size) {
            tridiagonal(i, i + 1) = beta[i];
            tridiagonal(i + 1, i) = beta[i];
        }
    }
    auto [ritz_values, ritz_vectors] = jacobi_eigendecomposition(
        std::move(tridiagonal), std::max(1e-14, config.eigensolver_tolerance * 0.01));
    const auto order = descending_order(ritz_values);

    EigenSystem result;
    result.values.resize(requested);
    result.vectors = DenseMatrix(n, requested, 0.0);
    result.basis_size = basis_size;
    result.used_dense_solver = false;
    std::vector<double> vector(n, 0.0);

    for (std::size_t output_column = 0; output_column < requested; ++output_column) {
        const std::size_t ritz_column = order[output_column];
        std::fill(vector.begin(), vector.end(), 0.0);
        for (std::size_t basis_column = 0; basis_column < basis_size; ++basis_column) {
            axpy(ritz_vectors(basis_column, ritz_column), q.data() + basis_column * n,
                 vector.data(), n);
        }
        const double length = norm(vector.data(), n);
        if (length <= config.eigensolver_tolerance) {
            throw std::runtime_error("Lanczos produced a zero Ritz vector");
        }
        for (std::size_t row = 0; row < n; ++row) {
            vector[row] /= length;
            result.vectors(row, output_column) = vector[row];
        }
        result.values[output_column] = op.laplacian_value(ritz_values[ritz_column]);
        result.max_residual = std::max(result.max_residual,
                                       eigen_residual(op, vector.data(), ritz_values[ritz_column]));
    }
    return result;
}

EigenSystem smallest_laplacian_eigenpairs(const Graph &graph, const Config &config) {
    ShiftedLaplacianOperator op(graph, config.laplacian);
    if (graph.num_nodes() <= config.dense_solver_threshold) {
        return dense_eigenpairs(op, config.num_communities, config.eigensolver_tolerance);
    }
    return lanczos_eigenpairs(op, config.num_communities, config);
}

double bisection_objective(const Graph &graph, const Labels &labels, LaplacianType type) {
    std::size_t size0 = 0;
    double volume0 = 0.0;
    double cut = 0.0;
    for (std::size_t raw_u = 0; raw_u < graph.num_nodes(); ++raw_u) {
        const NodeId u = static_cast<NodeId>(raw_u);
        if (labels[raw_u] == 0) {
            ++size0;
            volume0 += graph.weighted_degree(u);
        }
    }
    graph.for_each_edge([&](const WeightedEdge &edge) {
        if (edge.u != edge.v && labels[edge.u] != labels[edge.v]) {
            cut += edge.weight;
        }
    });

    const std::size_t n = graph.num_nodes();
    if (size0 == 0 || size0 == n) {
        return std::numeric_limits<double>::infinity();
    }
    if (type == LaplacianType::Unnormalized) {
        return cut * (1.0 / static_cast<double>(size0) + 1.0 / static_cast<double>(n - size0));
    }
    const double volume1 = 2.0 * graph.total_edge_weight() - volume0;
    if (volume0 <= 0.0 || volume1 <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return cut * (1.0 / volume0 + 1.0 / volume1);
}

std::pair<Labels, double> best_sweep_cut(const Graph &graph, const std::vector<double> &coordinate,
                                         LaplacianType type) {
    const std::size_t n = graph.num_nodes();
    std::vector<NodeId> order(n);
    std::iota(order.begin(), order.end(), NodeId{0});
    std::sort(order.begin(), order.end(), [&](NodeId left, NodeId right) {
        if (coordinate[left] != coordinate[right]) {
            return coordinate[left] < coordinate[right];
        }
        return left < right;
    });

    std::vector<bool> in_left(n, false);
    double cut = 0.0;
    double left_volume = 0.0;
    const double total_volume = 2.0 * graph.total_edge_weight();
    double best_score = std::numeric_limits<double>::infinity();
    std::size_t best_size = 0;

    for (std::size_t position = 0; position + 1 < n; ++position) {
        const NodeId node = order[position];
        for (const Neighbor neighbor : graph.neighbors(node)) {
            if (neighbor.node != node) {
                cut += in_left[neighbor.node] ? -neighbor.weight : neighbor.weight;
            }
        }
        in_left[node] = true;
        left_volume += graph.weighted_degree(node);

        if (std::abs(coordinate[node] - coordinate[order[position + 1]]) <= 1e-14) {
            continue;
        }
        const std::size_t left_size = position + 1;
        double score = std::numeric_limits<double>::infinity();
        if (type == LaplacianType::Unnormalized) {
            score = cut * (1.0 / static_cast<double>(left_size) +
                           1.0 / static_cast<double>(n - left_size));
        } else {
            const double right_volume = total_volume - left_volume;
            if (left_volume > 0.0 && right_volume > 0.0) {
                score = cut * (1.0 / left_volume + 1.0 / right_volume);
            }
        }
        if (score < best_score) {
            best_score = score;
            best_size = left_size;
        }
    }

    if (best_size == 0) {
        best_size = n / 2;
    }
    Labels labels(n, 1);
    for (std::size_t i = 0; i < best_size; ++i) {
        labels[order[i]] = 0;
    }
    labels = normalize_labels(labels);
    return {labels, bisection_objective(graph, labels, type)};
}

double squared_distance(const DenseMatrix &points, std::size_t row, const DenseMatrix &centers,
                        std::size_t center) {
    double distance = 0.0;
    for (std::size_t column = 0; column < points.cols; ++column) {
        const double difference = points(row, column) - centers(center, column);
        distance += difference * difference;
    }
    return distance;
}

DenseMatrix initialize_kmeans_pp(const DenseMatrix &points, std::size_t k, std::mt19937_64 &rng) {
    DenseMatrix centers(k, points.cols, 0.0);
    std::uniform_int_distribution<std::size_t> first_pick(0, points.rows - 1);
    const std::size_t first = first_pick(rng);
    for (std::size_t column = 0; column < points.cols; ++column) {
        centers(0, column) = points(first, column);
    }

    std::vector<double> minimum_distance(points.rows, std::numeric_limits<double>::infinity());
    for (std::size_t center = 1; center < k; ++center) {
        double total = 0.0;
        for (std::size_t row = 0; row < points.rows; ++row) {
            minimum_distance[row] =
                std::min(minimum_distance[row], squared_distance(points, row, centers, center - 1));
            total += minimum_distance[row];
        }

        std::size_t chosen = 0;
        if (total <= 0.0 || !std::isfinite(total)) {
            chosen = first_pick(rng);
        } else {
            std::uniform_real_distribution<double> pick(0.0, total);
            const double target = pick(rng);
            double cumulative = 0.0;
            for (std::size_t row = 0; row < points.rows; ++row) {
                cumulative += minimum_distance[row];
                if (cumulative >= target) {
                    chosen = row;
                    break;
                }
            }
        }
        for (std::size_t column = 0; column < points.cols; ++column) {
            centers(center, column) = points(chosen, column);
        }
    }
    return centers;
}

struct KMeansResult {
    Labels labels;
    double sse{std::numeric_limits<double>::infinity()};
    std::size_t iterations{};
};

KMeansResult kmeans(const DenseMatrix &points, std::size_t k, const Config &config) {
    const std::size_t restarts = std::max<std::size_t>(1, config.kmeans_restarts);
    std::mt19937_64 rng(config.seed);
    KMeansResult global_best;

    for (std::size_t restart = 0; restart < restarts; ++restart) {
        DenseMatrix centers = initialize_kmeans_pp(points, k, rng);
        Labels labels(points.rows, -1);
        std::vector<double> assigned_distance(points.rows, 0.0);
        std::size_t iteration = 0;

        for (; iteration < config.kmeans_max_iterations; ++iteration) {
            bool changed = false;
            for (std::size_t row = 0; row < points.rows; ++row) {
                int best_cluster = 0;
                double best_distance = squared_distance(points, row, centers, 0);
                for (std::size_t center = 1; center < k; ++center) {
                    const double distance = squared_distance(points, row, centers, center);
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_cluster = static_cast<int>(center);
                    }
                }
                changed = changed || labels[row] != best_cluster;
                labels[row] = best_cluster;
                assigned_distance[row] = best_distance;
            }

            DenseMatrix next_centers(k, points.cols, 0.0);
            std::vector<std::size_t> counts(k, 0);
            for (std::size_t row = 0; row < points.rows; ++row) {
                const std::size_t cluster = static_cast<std::size_t>(labels[row]);
                ++counts[cluster];
                for (std::size_t column = 0; column < points.cols; ++column) {
                    next_centers(cluster, column) += points(row, column);
                }
            }
            std::vector<bool> used_farthest(points.rows, false);
            for (std::size_t cluster = 0; cluster < k; ++cluster) {
                if (counts[cluster] > 0) {
                    for (std::size_t column = 0; column < points.cols; ++column) {
                        next_centers(cluster, column) /= static_cast<double>(counts[cluster]);
                    }
                    continue;
                }
                std::size_t farthest = 0;
                for (std::size_t row = 1; row < points.rows; ++row) {
                    if (!used_farthest[row] &&
                        (used_farthest[farthest] ||
                         assigned_distance[row] > assigned_distance[farthest])) {
                        farthest = row;
                    }
                }
                used_farthest[farthest] = true;
                for (std::size_t column = 0; column < points.cols; ++column) {
                    next_centers(cluster, column) = points(farthest, column);
                }
            }

            double shift_squared = 0.0;
            for (std::size_t i = 0; i < centers.values.size(); ++i) {
                const double difference = next_centers.values[i] - centers.values[i];
                shift_squared += difference * difference;
            }
            centers = std::move(next_centers);
            if (!changed || std::sqrt(shift_squared) <= config.kmeans_tolerance) {
                ++iteration;
                break;
            }
        }

        double sse = 0.0;
        for (std::size_t row = 0; row < points.rows; ++row) {
            sse += squared_distance(points, row, centers, static_cast<std::size_t>(labels[row]));
        }
        if (sse < global_best.sse) {
            global_best.labels = normalize_labels(labels);
            global_best.sse = sse;
            global_best.iterations = iteration;
        }
    }
    return global_best;
}

} // namespace

AlgorithmResult run(const Graph &graph, const Config &config) {
    validate_graph(graph);
    AlgorithmResult result;
    result.algorithm = "spectral";

    const std::size_t n = graph.num_nodes();
    if (n == 0) {
        result.labels = {};
        return result;
    }
    if (config.num_communities < 2 || config.num_communities > n) {
        throw std::invalid_argument("spectral num_communities must satisfy 2 <= K <= n");
    }
    if (!(config.eigensolver_tolerance > 0.0) || !std::isfinite(config.eigensolver_tolerance) ||
        !(config.maximum_eigensolver_residual > 0.0) ||
        !std::isfinite(config.maximum_eigensolver_residual) || !(config.kmeans_tolerance > 0.0) ||
        !std::isfinite(config.kmeans_tolerance) || config.kmeans_restarts == 0 ||
        config.kmeans_max_iterations == 0) {
        throw std::invalid_argument("spectral tolerances/iteration limits must be positive");
    }

    const EigenSystem eigen = smallest_laplacian_eigenpairs(graph, config);
    if (!std::isfinite(eigen.max_residual) ||
        eigen.max_residual > config.maximum_eigensolver_residual) {
        throw std::runtime_error("Lanczos/Jacobi eigen residual exceeds the configured maximum; "
                                 "increase --lanczos-basis or inspect the graph spectrum");
    }
    result.diagnostics["lambda_1"] = eigen.values[0];
    if (eigen.values.size() > 1) {
        result.diagnostics["lambda_2"] = eigen.values[1];
    }
    result.diagnostics["eigensolver_backend"] = eigen.used_dense_solver ? 0.0 : 1.0;
    result.diagnostics["eigensolver_basis_size"] = static_cast<double>(eigen.basis_size);
    result.diagnostics["eigensolver_max_residual"] = eigen.max_residual;

    if (config.num_communities == 2) {
        std::vector<double> coordinate(n, 0.0);
        for (std::size_t node = 0; node < n; ++node) {
            coordinate[node] = eigen.vectors(node, 1);
            if (config.laplacian == LaplacianType::SymmetricNormalized) {
                const double degree = graph.weighted_degree(static_cast<NodeId>(node));
                coordinate[node] = degree > 0.0 ? coordinate[node] / std::sqrt(degree) : 0.0;
            }
        }

        Labels labels(n, 0);
        double objective = std::numeric_limits<double>::infinity();
        if (config.bisection == BisectionMethod::Sign) {
            std::size_t negatives = 0;
            for (std::size_t node = 0; node < n; ++node) {
                labels[node] = coordinate[node] < 0.0 ? 0 : 1;
                negatives += labels[node] == 0 ? 1 : 0;
            }
            if (negatives == 0 || negatives == n) {
                auto swept = best_sweep_cut(graph, coordinate, config.laplacian);
                labels = std::move(swept.first);
                objective = swept.second;
            } else {
                labels = normalize_labels(labels);
                objective = bisection_objective(graph, labels, config.laplacian);
            }
        } else {
            auto swept = best_sweep_cut(graph, coordinate, config.laplacian);
            labels = std::move(swept.first);
            objective = swept.second;
        }
        result.labels = std::move(labels);
        result.objective = objective;
        return result;
    }

    DenseMatrix embedding = eigen.vectors;
    if (config.row_normalize_embedding) {
        for (std::size_t row = 0; row < embedding.rows; ++row) {
            double squared_length = 0.0;
            for (std::size_t column = 0; column < embedding.cols; ++column) {
                squared_length += embedding(row, column) * embedding(row, column);
            }
            const double length = std::sqrt(squared_length);
            if (length > 0.0) {
                for (std::size_t column = 0; column < embedding.cols; ++column) {
                    embedding(row, column) /= length;
                }
            }
        }
    }

    const KMeansResult clustered = kmeans(embedding, config.num_communities, config);
    result.labels = clustered.labels;
    result.objective = clustered.sse;
    result.diagnostics["kmeans_sse"] = clustered.sse;
    result.diagnostics["kmeans_iterations_best_restart"] =
        static_cast<double>(clustered.iterations);
    return result;
}

} // namespace cd::spectral
