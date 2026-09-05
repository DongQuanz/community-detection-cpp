#!/usr/bin/env python3
"""Aggregate raw runs into report, stability, and robustness tables."""

from __future__ import annotations

import argparse
import functools
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


@functools.lru_cache(maxsize=128)
def load_labels(raw_path: str) -> np.ndarray:
    values = np.loadtxt(resolve(raw_path), delimiter="\t", skiprows=1, usecols=1, dtype=np.int64)
    return np.atleast_1d(values)


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
            "q1": np.nan, "median": np.nan, "q3": np.nan, "max": np.nan,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q3": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    frame = pd.json_normalize(summary, sep=".")
    return frame.iloc[0].to_dict()


def read_runs(index_records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    community_frames: list[pd.DataFrame] = []
    for record in index_records:
        if record["status"] not in {"ok", "existing"}:
            continue
        summary_path = resolve(record["summary"])
        if not summary_path.is_file():
            raise FileNotFoundError(f"Indexed summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            **flatten_summary(summary),
            "job_id": record["job_id"],
            "protocol": record["protocol"],
            "scope_id": record["scope_id"],
            "sample_profile": record.get("sample_profile"),
            "sample_seed": record.get("sample_seed"),
            "graph": record["graph"],
            "graph_sha256": record.get("graph_sha256"),
            "reference": record.get("reference"),
            "reference_sha256": record.get("reference_sha256"),
            "partition": relative(summary_path.parent / "partition.tsv"),
            "summary": record["summary"],
            "command": record.get("command"),
            "wall_seconds": record.get("wall_seconds"),
            "algorithm_seed": int(record["algorithm_seed"]),
            "requested_k": int(record["requested_k"]),
            "resolution": float(record["resolution"]),
            "dropout_rate": float(record["dropout_rate"]),
            "dropout_seed": int(record["dropout_seed"]),
        }

        predicted = load_labels(row["partition"])
        if predicted.size != int(row["n"]):
            raise ValueError(f"Partition size mismatch in job {record['job_id']}")
        if record.get("reference"):
            reference = load_labels(record["reference"])
            if reference.size != predicted.size:
                raise ValueError(f"Reference size mismatch in job {record['job_id']}")
            sklearn_ari = adjusted_rand_score(reference, predicted)
            native_ari = row.get("external.ari")
            if native_ari is None or abs(float(native_ari) - sklearn_ari) > 1e-9:
                raise ValueError(
                    f"C++/scikit-learn ARI mismatch in {record['job_id']}: "
                    f"{native_ari} vs {sklearn_ari}"
                )
            row["external.ami_arithmetic"] = adjusted_mutual_info_score(
                reference, predicted, average_method="arithmetic"
            )
        else:
            row["external.ami_arithmetic"] = np.nan
        run_rows.append(row)

        communities_path = summary_path.parent / "communities.csv"
        communities = pd.read_csv(communities_path)
        communities.insert(0, "job_id", record["job_id"])
        communities.insert(1, "protocol", record["protocol"])
        communities.insert(2, "dataset", record["dataset"])
        communities.insert(3, "scope_id", record["scope_id"])
        communities.insert(4, "sample_profile", record.get("sample_profile"))
        communities.insert(5, "sample_seed", record.get("sample_seed"))
        communities.insert(6, "algorithm", record["algorithm"])
        communities.insert(7, "algorithm_seed", int(record["algorithm_seed"]))
        communities.insert(8, "requested_k", int(record["requested_k"]))
        communities.insert(9, "resolution", float(record["resolution"]))
        communities.insert(10, "dropout_rate", float(record["dropout_rate"]))
        communities.insert(11, "dropout_seed", int(record["dropout_seed"]))
        community_frames.append(communities)

    runs = pd.DataFrame(run_rows)
    communities = (
        pd.concat(community_frames, ignore_index=True)
        if community_frames else pd.DataFrame()
    )
    return runs, communities


GROUP_KEYS = [
    "protocol", "dataset", "scope_id", "algorithm", "requested_k", "resolution",
    "parameters.laplacian",
]


def stability_tables(runs: pd.DataFrame, compute_ami: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs: list[dict[str, Any]] = []
    baseline = runs[runs["dropout_rate"] == 0.0]
    for keys, group in baseline.groupby(GROUP_KEYS, dropna=False):
        if len(group) < 2:
            continue
        key_values = dict(zip(GROUP_KEYS, keys, strict=True))
        records = group.to_dict("records")
        for left, right in itertools.combinations(records, 2):
            left_labels = load_labels(left["partition"])
            right_labels = load_labels(right["partition"])
            row = {
                **key_values,
                "sample_profile": left.get("sample_profile"),
                "sample_seed": left.get("sample_seed"),
                "left_job_id": left["job_id"],
                "right_job_id": right["job_id"],
                "left_seed": int(left["algorithm_seed"]),
                "right_seed": int(right["algorithm_seed"]),
                "ari": adjusted_rand_score(left_labels, right_labels),
            }
            row["ami_arithmetic"] = (
                adjusted_mutual_info_score(
                    left_labels, right_labels, average_method="arithmetic"
                )
                if compute_ami else np.nan
            )
            pairs.append(row)

    pair_frame = pd.DataFrame(pairs)
    summaries: list[dict[str, Any]] = []
    if not pair_frame.empty:
        for keys, group in pair_frame.groupby(GROUP_KEYS, dropna=False):
            row = dict(zip(GROUP_KEYS, keys, strict=True))
            for metric in ["ari", "ami_arithmetic"]:
                for statistic, value in distribution(group[metric]).items():
                    row[f"{metric}_{statistic}"] = value
            summaries.append(row)
    return pair_frame, pd.DataFrame(summaries)


def robustness_tables(runs: pd.DataFrame, compute_ami: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_keys = [
        "protocol", "dataset", "scope_id", "algorithm", "requested_k", "resolution",
        "parameters.laplacian", "algorithm_seed",
    ]
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in runs[runs["dropout_rate"] == 0.0].to_dict("records"):
        key = tuple(row.get(column) for column in baseline_keys)
        baselines[key] = row

    comparisons: list[dict[str, Any]] = []
    for perturbed in runs[runs["dropout_rate"] > 0.0].to_dict("records"):
        key = tuple(perturbed.get(column) for column in baseline_keys)
        baseline = baselines.get(key)
        if baseline is None:
            continue
        base_labels = load_labels(baseline["partition"])
        changed_labels = load_labels(perturbed["partition"])
        row = {
            **{column: perturbed.get(column) for column in baseline_keys},
            "sample_profile": perturbed.get("sample_profile"),
            "sample_seed": perturbed.get("sample_seed"),
            "baseline_job_id": baseline["job_id"],
            "perturbed_job_id": perturbed["job_id"],
            "dropout_rate": float(perturbed["dropout_rate"]),
            "dropout_seed": int(perturbed["dropout_seed"]),
            "removed_edges": int(perturbed["dropout.removed_edges"]),
            "ari": adjusted_rand_score(base_labels, changed_labels),
        }
        row["ami_arithmetic"] = (
            adjusted_mutual_info_score(
                base_labels, changed_labels, average_method="arithmetic"
            )
            if compute_ami else np.nan
        )
        comparisons.append(row)

    comparison_frame = pd.DataFrame(comparisons)
    summaries: list[dict[str, Any]] = []
    summary_keys = GROUP_KEYS + ["dropout_rate"]
    if not comparison_frame.empty:
        for keys, group in comparison_frame.groupby(summary_keys, dropna=False):
            row = dict(zip(summary_keys, keys, strict=True))
            for metric in ["ari", "ami_arithmetic"]:
                for statistic, value in distribution(group[metric]).items():
                    row[f"{metric}_{statistic}"] = value
            summaries.append(row)
    return comparison_frame, pd.DataFrame(summaries)


RUN_METRICS = [
    "num_communities", "elapsed_ms", "metrics_elapsed_ms", "wall_seconds",
    "peak_rss_mb", "dropout.removed_edges",
    "objective", "modularity_q1", "internal.coverage", "internal.ratio_cut",
    "internal.normalized_cut", "internal.modularity_q_gamma",
    "internal.singletons", "internal.isolated_nodes",
    "internal.size_distribution.valid_count", "internal.size_distribution.min",
    "internal.size_distribution.q1", "internal.size_distribution.median",
    "internal.size_distribution.q3", "internal.size_distribution.max",
    "internal.size_distribution.community_mean",
    "internal.size_distribution.node_weighted_mean",
    "internal.density_distribution.valid_count", "internal.density_distribution.min",
    "internal.density_distribution.q1", "internal.density_distribution.median",
    "internal.density_distribution.q3", "internal.density_distribution.max",
    "internal.density_distribution.community_mean",
    "internal.density_distribution.node_weighted_mean",
    "internal.conductance_distribution.valid_count",
    "internal.conductance_distribution.min", "internal.conductance_distribution.q1",
    "internal.conductance_distribution.median", "internal.conductance_distribution.q3",
    "internal.conductance_distribution.max",
    "internal.conductance_distribution.community_mean",
    "internal.conductance_distribution.node_weighted_mean",
    "external.pairwise_precision",
    "external.pairwise_recall", "external.pairwise_f1", "external.ari",
    "external.ami_arithmetic",
    "diagnostics.edges_removed", "diagnostics.hierarchy_levels",
    "diagnostics.eigensolver_basis_size", "diagnostics.eigensolver_max_residual",
    "diagnostics.kmeans_sse", "diagnostics.kmeans_iterations_best_restart",
    "diagnostics.levels", "diagnostics.local_passes", "diagnostics.iterations",
    "diagnostics.converged",
]


def aggregate_by(runs: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    available = [metric for metric in RUN_METRICS if metric in runs.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in runs.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, keys, strict=True))
        row["runs"] = len(group)
        for metric in available:
            values = pd.to_numeric(group[metric], errors="coerce")
            for statistic, value in distribution(values).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by(runs, GROUP_KEYS + ["dropout_rate"])


def aggregate_samples(runs: pd.DataFrame) -> pd.DataFrame:
    sampled = runs[runs["sample_profile"].notna()]
    if sampled.empty:
        return pd.DataFrame()
    group_keys = [
        "protocol", "dataset", "sample_profile", "algorithm", "requested_k",
        "resolution", "parameters.laplacian", "dropout_rate",
    ]
    return aggregate_by(sampled, group_keys)


def aggregate_sample_stability(stability_pairs: pd.DataFrame) -> pd.DataFrame:
    if stability_pairs.empty or "sample_profile" not in stability_pairs:
        return pd.DataFrame()
    sampled = stability_pairs[stability_pairs["sample_profile"].notna()]
    if sampled.empty:
        return pd.DataFrame()
    group_keys = [
        "protocol", "dataset", "sample_profile", "algorithm", "requested_k",
        "resolution", "parameters.laplacian",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in sampled.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, keys, strict=True))
        row["sample_scopes"] = int(group["sample_seed"].nunique())
        row["partition_pairs"] = len(group)
        for metric in ["ari", "ami_arithmetic"]:
            for statistic, value in distribution(group[metric]).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_table(manifest: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for dataset in manifest["datasets"]:
        source_files = dataset.get("source_files", [])
        row = {
            key: value for key, value in dataset.items()
            if key not in {"samples", "source_files"}
        }
        row["source_file_count"] = len(source_files)
        row["source_paths"] = " | ".join(record["path"] for record in source_files)
        row["source_sha256"] = " | ".join(record["sha256"] for record in source_files)
        rows.append(row)
    return pd.json_normalize(rows, sep=".")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="NA", lineterminator="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--manifest", default="data/processed/manifest.json")
    parser.add_argument(
        "--skip-stability-ami", action="store_true",
        help="Use only ARI for stability/robustness; external AMI is still computed",
    )
    arguments = parser.parse_args()

    results = resolve(arguments.results)
    index_path = results / "run_index.jsonl"
    manifest_path = resolve(arguments.manifest)
    index_records = load_jsonl(index_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    runs, communities = read_runs(index_records)
    if runs.empty:
        raise RuntimeError("No successful runs to summarize")
    stability_pairs, stability_summary = stability_tables(
        runs, compute_ami=not arguments.skip_stability_ami
    )
    robustness_pairs, robustness_summary = robustness_tables(
        runs, compute_ami=not arguments.skip_stability_ami
    )
    aggregate = aggregate_runs(runs)
    sample_aggregate = aggregate_samples(runs)
    sample_stability_summary = aggregate_sample_stability(stability_pairs)
    skipped = pd.DataFrame(
        [record for record in index_records if record["status"] not in {"ok", "existing"}]
    )

    outputs = {
        "runs": tables / "runs.csv",
        "communities": tables / "communities.csv",
        "aggregate": tables / "aggregate.csv",
        "sample_aggregate": tables / "sample_aggregate.csv",
        "stability_pairs": tables / "stability_pairs.csv",
        "stability_summary": tables / "stability_summary.csv",
        "sample_stability_summary": tables / "sample_stability_summary.csv",
        "robustness_pairs": tables / "robustness_pairs.csv",
        "robustness_summary": tables / "robustness_summary.csv",
        "datasets": tables / "datasets.csv",
        "skipped": tables / "skipped.csv",
    }
    write_csv(runs, outputs["runs"])
    write_csv(communities, outputs["communities"])
    write_csv(aggregate, outputs["aggregate"])
    write_csv(sample_aggregate, outputs["sample_aggregate"])
    write_csv(stability_pairs, outputs["stability_pairs"])
    write_csv(stability_summary, outputs["stability_summary"])
    write_csv(sample_stability_summary, outputs["sample_stability_summary"])
    write_csv(robustness_pairs, outputs["robustness_pairs"])
    write_csv(robustness_summary, outputs["robustness_summary"])
    write_csv(dataset_table(manifest), outputs["datasets"])
    write_csv(skipped, outputs["skipped"])

    ready = {
        "schema_version": 1,
        "scikit_learn_version": sklearn.__version__,
        "ami_convention": "arithmetic mean normalization; natural logarithm",
        "quantile_convention": "linear interpolation (NumPy default)",
        "files": {name: relative(path) for name, path in outputs.items()},
    }
    (results / "report_ready.json").write_text(
        json.dumps(ready, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(relative(results / "report_ready.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
