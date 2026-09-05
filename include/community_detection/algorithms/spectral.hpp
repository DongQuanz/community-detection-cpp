#pragma once

#include "community_detection/core/graph.hpp"
#include "community_detection/core/result.hpp"

#include <cstddef>
#include <cstdint>

namespace cd::spectral {

/** @brief Laplacian used to construct the spectral embedding. */
enum class LaplacianType {
    Unnormalized,       ///< L = D - A, associated with RatioCut.
    SymmetricNormalized ///< L_sym = I - D^{-1/2} A D^{-1/2}, associated with Ncut.
};

/** @brief Fiedler-vector discretization used when K = 2. */
enum class BisectionMethod {
    Sign,     ///< Threshold at zero.
    BestSweep ///< Test all valid thresholds and retain the best RatioCut/Ncut.
};

/** Default Lanczos basis size shared by the solver and resource guard. */
[[nodiscard]] std::size_t automatic_lanczos_basis_size(std::size_t num_communities);

/** @brief Spectral clustering parameters. */
struct Config {
    /** Number of communities K; must be in [2, n]. */
    std::size_t num_communities{2};

    /** Laplacian variant. */
    LaplacianType laplacian{LaplacianType::SymmetricNormalized};

    /** Bisection method used when K = 2. */
    BisectionMethod bisection{BisectionMethod::BestSweep};

    /** Normalize embedding rows before k-means when K > 2. */
    bool row_normalize_embedding{true};

    /** Number of k-means restarts; the lowest-SSE solution is retained. */
    std::size_t kmeans_restarts{10};

    /** Maximum iterations per k-means restart. */
    std::size_t kmeans_max_iterations{300};

    /** K-means convergence tolerance for centroid displacement. */
    double kmeans_tolerance{1e-6};

    /** Seed for k-means. */
    std::uint64_t seed{42};

    /** Use dense Jacobi up to this n; zero always selects Lanczos. */
    std::size_t dense_solver_threshold{256};

    /**
     * Lanczos Krylov-space size. Zero selects `max(128, 6*K+32)`, capped by n.
     * The value must exceed K unless it equals n.
     */
    std::size_t lanczos_basis_size{0};

    /** Lanczos stopping and orthogonality tolerance. */
    double eigensolver_tolerance{1e-9};

    /** Maximum relative eigensolver residual accepted for a partition. */
    double maximum_eigensolver_residual{1e-4};
};

/**
 * @brief Run spectral clustering on an undirected graph.
 *
 * - K = 2: use the smallest non-trivial eigenvector (Fiedler vector), followed
 *   by sign thresholding or a sweep cut.
 * - K > 2: use the K smallest eigenvectors as node coordinates and cluster the
 *   rows with k-means.
 *
 * Small graphs use dense symmetric Jacobi as a validation reference. Large
 * graphs use matrix-free Lanczos with full reorthogonalization and never build
 * an n-by-n matrix.
 */
[[nodiscard]] AlgorithmResult run(const Graph &graph, const Config &config = {});

} // namespace cd::spectral
