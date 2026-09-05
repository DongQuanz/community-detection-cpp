#pragma once

/**
 * @file community_detection.hpp
 * @brief Convenience header for the complete public library API.
 *
 * Include this file to use Graph, all four algorithms, modularity metrics, and
 * the experiment runner without including each component separately.
 */

#include "community_detection/algorithms/girvan_newman.hpp"
#include "community_detection/algorithms/label_propagation.hpp"
#include "community_detection/algorithms/louvain.hpp"
#include "community_detection/algorithms/spectral.hpp"
#include "community_detection/core/graph.hpp"
#include "community_detection/core/graph_transform.hpp"
#include "community_detection/core/partition.hpp"
#include "community_detection/core/result.hpp"
#include "community_detection/experiment/runner.hpp"
#include "community_detection/io/graph_io.hpp"
#include "community_detection/metrics/external.hpp"
#include "community_detection/metrics/modularity.hpp"
#include "community_detection/metrics/partition_metrics.hpp"
