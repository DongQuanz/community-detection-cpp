# Community Detection Benchmark

A reproducible C++17 and Python research pipeline for evaluating disjoint
community detection on simple, undirected, unweighted graphs.

The repository provides verified implementations of four classical methods,
a common command-line interface, deterministic dataset preparation, resource
guards for infeasible jobs, and publication-ready aggregate tables.

| Method | Implementation | Intended scale |
|---|---|---|
| Girvan–Newman | Exact unweighted Brandes; one edge removed and betweenness recomputed per step | Small graphs and registered samples |
| Spectral clustering | RatioCut or Ncut; dense Jacobi and matrix-free Lanczos backends | Small to medium graphs, subject to work guards |
| Louvain | Multi-level modularity optimization with a consistent resolution parameter | Large graphs |
| Label propagation (LPA) | Seeded asynchronous or synchronous updates; isolates preserved | Large graphs |

## Highlights

- Sparse CSR graph storage throughout the C++ core.
- Reproducible randomness controlled exclusively by configured seeds.
- One canonical benchmark suite in [`configs/experiments.json`](configs/experiments.json).
- Dataset checksums, binary/config provenance, resumable jobs, and explicit
  failure or resource-guard statuses.
- Intrinsic quality, external agreement, stability, robustness, runtime, and
  peak-memory measurements.
- A reusable C++ library, a single-run CLI, and Python orchestration tools.
- No required third-party C++ dependencies.

## Requirements

- A C++17 compiler
- CMake 3.18 or newer
- Python 3.10 or newer
- Python packages listed in [`requirements.txt`](requirements.txt)

The Python dependencies are used for dataset conversion and report validation;
the C++ library itself uses only the standard library.

## Quick start

### Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On multi-configuration generators such as Visual Studio, add `--config Release`
to the build and test commands.

Install the Python tooling when running the full pipeline:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

### Prepare datasets

Download datasets locally following [`data/README.md`](data/README.md), then run:

```bash
python3 tools/prepare_datasets.py --config configs/datasets.example.json
```

Preparation converts every source to the compact `CDGRPH1` format, removes
self-loops and duplicate edges, maps node IDs to `0..n-1`, creates registered
induced samples, and records source and output SHA-256 hashes in
`data/processed/manifest.json`.

Raw data, processed data, local path overrides, and results are excluded from
Git. Verify the generated manifest before starting a benchmark.

### Run the canonical benchmark

Preview the complete job matrix first:

```bash
python3 tools/run_experiments.py \
  --config configs/experiments.json \
  --binary build/community_detect \
  --dry-run
```

Run the full registered suite and safely resume interrupted work:

```bash
python3 tools/run_experiments.py \
  --config configs/experiments.json \
  --binary build/community_detect \
  --resume
```

`configs/experiments.json` is the only published experiment definition. Its
protocol entries are phases of one benchmark suite: full-graph evaluation,
shared large-graph samples, spectral scaling, registered sensitivity analyses,
and edge-dropout robustness. Keeping these phases in one configuration ensures
that seeds, guards, defaults, and provenance remain consistent.

Jobs are sequential by default so runtime and peak-memory measurements are not
distorted by resource contention. Infeasible jobs are retained as
`skipped_resource_guard` records rather than silently omitted or reported as
zero.

### Summarize results

```bash
python3 tools/summarize_results.py \
  --results results \
  --manifest data/processed/manifest.json
```

The report tables include per-run and per-community metrics, grouped
distributions, stability across seeds, robustness under 1% and 5% edge dropout,
dataset statistics, and all skipped jobs. Missing metrics are represented as
`null` in JSON and blank/`NA` in CSV.

## Run one algorithm

```bash
build/community_detect \
  --graph data/processed/karate/full.cdgraph \
  --reference data/processed/karate/reference.tsv \
  --dataset karate \
  --algorithm louvain \
  --seed 42 \
  --resolution 1.0 \
  --output-dir results/manual/karate_louvain_seed42
```

Run `community_detect --help` for all algorithm and output options. Each run
writes a normalized partition, per-community metrics, a machine-readable
summary, convergence diagnostics, and timing information.

## Metrics and interpretation

The pipeline reports coverage, resolution-aware modularity, RatioCut, Ncut,
community density and conductance distributions, pairwise precision/recall/F1,
Adjusted Rand Index, and Adjusted Mutual Information. ARI is independently
recomputed with scikit-learn during summarization.

Compare methods across multiple evidence layers. Louvain directly optimizes
modularity, while spectral clustering directly targets a cut objective; neither
objective alone establishes an overall winner. Always report the number of
communities, scope, stability, robustness, and resource use alongside quality.

Full graphs, 400-node samples, and 5,000-node samples are distinct scopes and
must not be pooled into a single ranking. Reddit2 class labels are external
metadata rather than structural ground truth. ca-CondMat has no external labels,
so its external metrics remain undefined.

## Use as a C++ library

The umbrella header exposes the graph, algorithms, metrics, I/O, and experiment
wrappers:

```cpp
#include <community_detection/community_detection.hpp>

cd::Graph graph(3);
graph.add_edge(0, 1);
graph.add_edge(1, 2);

cd::louvain::Config config;
config.seed = 42;
const auto result = cd::experiment::run(graph, config);
```

Install and consume the exported CMake package:

```bash
cmake --install build --prefix /your/install/prefix
```

```cmake
find_package(community_detection CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE community_detection::community_detection)
```

## Repository layout

```text
apps/                         single-run command-line application
cmake/                        installed-package configuration
configs/                      canonical benchmark and dataset templates
data/README.md                local dataset acquisition and layout
examples/                     minimal C++ library example
include/community_detection/  public C++ API
src/                          C++ implementations
tests/                        unit and end-to-end smoke tests
tools/                        dataset, orchestration, and reporting tools
```

## Extending the project

The project is designed to grow without changing existing experiment semantics.

To add an algorithm, create matching public and implementation files under
`include/community_detection/algorithms/` and `src/algorithms/`, expose a typed
configuration and `run()` function, add an experiment wrapper and CLI dispatch,
then register job generation in `tools/run_experiments.py`. Add hand-computed toy
tests before including the method in the canonical benchmark.

To add a dataset, add a reader or reuse an existing input format in
`tools/prepare_datasets.py`, create a dataset entry in a copied local config,
verify the manifest, and only then update the published dataset template and
benchmark configuration. New ablations or sensitivity studies should be new
protocol phases; never change a registered default after inspecting results.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the required checks and compatibility
rules.

## Reproducibility and data policy

- The canonical input is a simple, undirected, unweighted graph after preprocessing.
- Every stochastic decision is derived from an explicit configuration seed.
- Provenance binds results to the binary, configuration, manifest, scripts, and platform.
- Existing output roots cannot be resumed with incompatible provenance.
- Datasets and generated results are not redistributed in this repository.
- Private maintainer notes (`AGENT.md`, `AGENTS.md`, and `docs/`) are ignored by Git.

If result artifacts need to be published, release them separately with the
configuration, manifest, provenance, run index, and validated aggregate tables.

## Contributing

Contributions are welcome. Please keep public code, comments, error messages,
and documentation in English, preserve algorithm identities and output schema
semantics, and run the complete quality gate before opening a pull request.

This repository does not currently declare a software license. Add an explicit
license before accepting third-party contributions or redistributing the code.
