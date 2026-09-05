# Contributing

Thank you for improving the Community Detection Benchmark. Public code,
comments, diagnostics, and documentation must be written in English.

## Development workflow

1. Keep each change focused and preserve backward-compatible defaults.
2. Add or update tests with every algorithm, metric, schema, or parser change.
3. Use a new protocol phase for sensitivity analyses and ablations. Do not alter
   a registered benchmark default after inspecting its results.
4. Never commit datasets, generated results, local configurations, credentials,
   binaries, caches, logs, or private maintainer documentation.
5. Run the complete quality gate before submitting a change.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 -m py_compile tools/*.py tests/test_pipeline.py
```

Run AddressSanitizer and UndefinedBehaviorSanitizer after graph-core or algorithm
changes on a supported compiler:

```bash
cmake -S . -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCOMMUNITY_DETECTION_ENABLE_SANITIZERS=ON
cmake --build build-sanitize -j
ctest --test-dir build-sanitize --output-on-failure
```

## Algorithm changes

- Girvan–Newman must remove exactly one maximum-betweenness edge and recompute
  Brandes after every removal.
- Louvain must use the same resolution in local moves, aggregation, objectives,
  and metrics. Modularity on original nodes must not decrease across levels
  beyond numerical tolerance.
- Normalized spectral bisection must use `g = D^{-1/2}z`. Large-graph execution
  must remain matrix-free, report eigensolver residuals, and reject inaccurate runs.
- LPA randomness must depend only on its configured seed, and isolated nodes must
  remain represented.
- All outputs are disjoint partitions normalized to labels `0..K-1`.

New algorithms should follow the existing typed `Config` plus `run()` API, use
the common `AlgorithmResult`, expose CLI options, participate in provenance and
resource-guard handling, and include a hand-computed toy test.

## Metrics and output schemas

Undefined JSON metrics use `null`; CSV uses an empty field or `NA`. Never encode
missing, failed, or skipped measurements as zero. Full graphs and different
sample sizes remain separate scopes.

When changing JSON or CSV schemas, update the runner, summarizer, smoke test, and
README together. Make a schema-version change explicit when compatibility is broken.

## Data and artifacts

Only `data/README.md` belongs in version control. Generated reports must remain
traceable to `manifest.json`, `provenance.json`, `run_index.jsonl`, the exact
configuration, and the binary hash. Do not edit experiment output manually.
