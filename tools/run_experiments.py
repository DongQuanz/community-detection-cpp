#!/usr/bin/env python3
"""Build a reproducible job matrix and run community_detect from JSON."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class Target:
    dataset: str
    scope_id: str
    graph: str
    graph_sha256: str
    reference: str | None
    reference_sha256: str | None
    n: int
    m: int
    sample_profile: str | None
    sample_seed: int | None


@dataclass(frozen=True)
class Job:
    job_id: str
    protocol: str
    dataset: str
    scope_id: str
    graph: str
    graph_sha256: str
    reference: str | None
    reference_sha256: str | None
    n: int
    m: int
    sample_profile: str | None
    sample_seed: int | None
    algorithm: str
    algorithm_seed: int
    requested_k: int
    resolution: float
    dropout_rate: float
    dropout_seed: int
    output_dir: str
    options: tuple[str, ...]


def targets_for_protocol(manifest: dict[str, Any], protocol: dict[str, Any]) -> list[Target]:
    allowed = set(protocol.get("datasets", []))
    targets: list[Target] = []
    for dataset in manifest["datasets"]:
        if allowed and dataset["name"] not in allowed:
            continue
        if protocol["scope"] == "full":
            targets.append(
                Target(
                    dataset=dataset["name"],
                    scope_id="full",
                    graph=dataset["graph"],
                    graph_sha256=dataset["sha256"],
                    reference=dataset.get("reference"),
                    reference_sha256=dataset.get("reference_sha256"),
                    n=int(dataset["n"]),
                    m=int(dataset["m"]),
                    sample_profile=None,
                    sample_seed=None,
                )
            )
            continue
        profile = protocol["sample_profile"]
        for sample in dataset.get("samples", []):
            if sample["profile"] != profile:
                continue
            targets.append(
                Target(
                    dataset=dataset["name"],
                    scope_id=f"{profile}_seed_{sample['seed']}",
                    graph=sample["graph"],
                    graph_sha256=sample["sha256"],
                    reference=sample.get("reference"),
                    reference_sha256=sample.get("reference_sha256"),
                    n=int(sample["n"]),
                    m=int(sample["m"]),
                    sample_profile=profile,
                    sample_seed=int(sample["seed"]),
                )
            )
    return targets


def k_values(
    config: dict[str, Any], protocol: dict[str, Any], dataset: str, algorithm: str
) -> list[int]:
    settings = {
        **config["dataset_settings"].get(dataset, {}),
        **protocol.get("dataset_settings", {}).get(dataset, {}),
    }
    if algorithm == "spectral":
        values = settings.get("spectral_k", [])
        if not values:
            raise ValueError(f"Missing spectral_k for {dataset}")
        return [int(value) for value in values]
    if algorithm == "girvan_newman":
        return [int(value) for value in settings.get("girvan_newman_k", [0])]
    return [0]


def dropout_pairs(protocol: dict[str, Any]) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    for specification in protocol.get("dropout", [{"rate": 0.0, "seeds": [0]}]):
        rate = float(specification["rate"])
        if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
            raise ValueError(f"Invalid dropout rate in {protocol['name']}: {rate}")
        for seed in specification.get("seeds", [0]):
            parsed_seed = int(seed)
            if parsed_seed < 0:
                raise ValueError(f"Dropout seed must be non-negative: {parsed_seed}")
            pairs.append((rate, parsed_seed))
    return pairs


def job_identifier(parts: list[str]) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def build_jobs(config: dict[str, Any], manifest: dict[str, Any]) -> list[Job]:
    output_root = resolve(config.get("output_dir", "results"))
    jobs: list[Job] = []
    algorithm_defaults = config.get("algorithm_options", {})
    resource_limits = config.get("resource_guards", {})

    for protocol in config["protocols"]:
        targets = targets_for_protocol(manifest, protocol)
        resolutions = [float(value) for value in protocol.get("resolutions", [1.0])]
        if any(not math.isfinite(value) or value <= 0.0 for value in resolutions):
            raise ValueError(f"Invalid resolution in protocol {protocol['name']}")
        for target in targets:
            for algorithm in protocol["algorithms"]:
                seeds = [int(value) for value in protocol["algorithm_seeds"][algorithm]]
                if any(seed < 0 for seed in seeds):
                    raise ValueError(f"Algorithm seed must be non-negative for {algorithm}")
                options = {
                    **algorithm_defaults.get(algorithm, {}),
                    **protocol.get("algorithm_options", {}).get(algorithm, {}),
                }
                if algorithm == "girvan_newman":
                    options.setdefault(
                        "gn_max_n", resource_limits.get("girvan_newman_max_n", 1000)
                    )
                    options.setdefault(
                        "gn_max_m", resource_limits.get("girvan_newman_max_m", 10000)
                    )
                elif algorithm == "spectral":
                    options.setdefault(
                        "spectral_max_edge_visits",
                        resource_limits.get("spectral_max_edge_visits", 2_000_000_000),
                    )
                    options.setdefault(
                        "spectral_max_orthogonalization_ops",
                        resource_limits.get(
                            "spectral_max_orthogonalization_ops", 5_000_000_000
                        ),
                    )
                option_arguments: list[str] = []
                for key, value in sorted(options.items()):
                    if value is False or value is None:
                        continue
                    option_arguments.append(f"--{key.replace('_', '-')}")
                    if value is not True:
                        option_arguments.append(str(value))

                algorithm_resolutions = resolutions if algorithm == "louvain" else [1.0]
                for requested_k in k_values(config, protocol, target.dataset, algorithm):
                    if algorithm == "spectral" and not 2 <= requested_k <= target.n:
                        raise ValueError(
                            f"spectral K={requested_k} is outside [2,{target.n}] for "
                            f"{target.dataset}/{target.scope_id}"
                        )
                    if algorithm == "girvan_newman" and not 0 <= requested_k <= target.n:
                        raise ValueError(
                            f"Girvan-Newman K={requested_k} is outside [0,{target.n}]"
                        )
                    for resolution in algorithm_resolutions:
                        for algorithm_seed in seeds:
                            for dropout_rate, dropout_seed in dropout_pairs(protocol):
                                labels = [
                                    protocol["name"], target.dataset, target.scope_id,
                                    target.graph_sha256, target.reference_sha256 or "no_reference",
                                    algorithm,
                                    str(requested_k), str(resolution), str(algorithm_seed),
                                    str(dropout_rate), str(dropout_seed),
                                    *option_arguments,
                                ]
                                identifier = job_identifier(labels)
                                output_dir = (
                                    output_root / "raw" / protocol["name"] / target.dataset
                                    / target.scope_id / algorithm / f"job_{identifier}"
                                )
                                jobs.append(
                                    Job(
                                        job_id=identifier,
                                        protocol=protocol["name"],
                                        dataset=target.dataset,
                                        scope_id=target.scope_id,
                                        graph=target.graph,
                                        graph_sha256=target.graph_sha256,
                                        reference=target.reference,
                                        reference_sha256=target.reference_sha256,
                                        n=target.n,
                                        m=target.m,
                                        sample_profile=target.sample_profile,
                                        sample_seed=target.sample_seed,
                                        algorithm=algorithm,
                                        algorithm_seed=algorithm_seed,
                                        requested_k=requested_k,
                                        resolution=resolution,
                                        dropout_rate=dropout_rate,
                                        dropout_seed=dropout_seed,
                                        output_dir=relative(output_dir),
                                        options=tuple(option_arguments),
                                    )
                                )
    return jobs


def numeric_job_option(job: Job, name: str, fallback: int) -> int:
    flag = f"--{name}"
    try:
        position = job.options.index(flag)
    except ValueError:
        return fallback
    if position + 1 >= len(job.options) or job.options[position + 1].startswith("--"):
        raise ValueError(f"Numeric option {flag} has no value in job {job.job_id}")
    return int(job.options[position + 1])


def skip_reason(job: Job, config: dict[str, Any]) -> str | None:
    limits = config.get("resource_guards", {})
    if job.algorithm == "girvan_newman":
        maximum_n = numeric_job_option(
            job, "gn-max-n", int(limits.get("girvan_newman_max_n", 1000))
        )
        maximum_m = numeric_job_option(
            job, "gn-max-m", int(limits.get("girvan_newman_max_m", 10000))
        )
        if job.n > maximum_n or job.m > maximum_m:
            return f"exact_gn_guard:n={job.n}>{maximum_n} or m={job.m}>{maximum_m}"
    if job.algorithm == "spectral":
        basis = numeric_job_option(
            job, "lanczos-basis", int(limits.get("spectral_basis_override", 0))
        )
        if basis == 0:
            basis = max(128, 6 * job.requested_k + 32)
        basis = min(basis, job.n)
        estimated_edges = 2 * job.m * basis
        maximum_edges = numeric_job_option(
            job,
            "spectral-max-edge-visits",
            int(limits.get("spectral_max_edge_visits", 2_000_000_000)),
        )
        if estimated_edges > maximum_edges:
            return f"spectral_edge_guard:{estimated_edges}>{maximum_edges}"
        estimated_orthogonalization = 4 * job.n * basis * basis
        maximum_orthogonalization = numeric_job_option(
            job,
            "spectral-max-orthogonalization-ops",
            int(limits.get("spectral_max_orthogonalization_ops", 5_000_000_000)),
        )
        if estimated_orthogonalization > maximum_orthogonalization:
            return (
                "spectral_orthogonalization_guard:"
                f"{estimated_orthogonalization}>{maximum_orthogonalization}"
            )
    return None


def command_for_job(binary: Path, job: Job) -> list[str]:
    command = [
        str(binary),
        "--graph", str(resolve(job.graph)),
        "--algorithm", job.algorithm,
        "--dataset", job.dataset,
        "--output-dir", str(resolve(job.output_dir)),
        "--seed", str(job.algorithm_seed),
        "--resolution", str(job.resolution),
        "--dropout-rate", str(job.dropout_rate),
        "--dropout-seed", str(job.dropout_seed),
    ]
    if job.reference:
        command.extend(["--reference", str(resolve(job.reference))])
    if job.requested_k > 0:
        command.extend(["--k", str(job.requested_k)])
    command.extend(job.options)
    return command


def valid_existing_summary(path: Path, job: Job) -> bool:
    partition = path.parent / "partition.tsv"
    communities = path.parent / "communities.csv"
    if not path.is_file() or not partition.is_file() or not communities.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        return (
            summary.get("status") == "ok"
            and summary.get("dataset") == job.dataset
            and summary.get("algorithm") == job.algorithm
            and int(summary.get("seed", -1)) == job.algorithm_seed
            and int(summary.get("n", -1)) == job.n
            and int(summary.get("requested_k", -1)) == job.requested_k
            and float(summary["parameters"]["resolution"]) == job.resolution
            and float(summary["dropout"]["rate"]) == job.dropout_rate
            and int(summary["dropout"]["seed"]) == job.dropout_seed
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False


def git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance_payload(binary: Path, config_path: Path, manifest_path: Path) -> dict[str, Any]:
    source_paths = [
        ROOT / "tools/run_experiments.py",
        ROOT / "tools/prepare_datasets.py",
        ROOT / "tools/summarize_results.py",
    ]
    git_commit = git_value("rev-parse", "HEAD")
    git_status = git_value("status", "--porcelain")
    return {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_packages": {
            package: installed_version(package)
            for package in ["numpy", "scipy", "pandas", "scikit-learn"]
        },
        "binary": relative(binary),
        "binary_sha256": file_sha256(binary),
        "config": relative(config_path),
        "config_sha256": file_sha256(config_path),
        "manifest": relative(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "pipeline_sha256": {
            relative(path): file_sha256(path) for path in source_paths
        },
        "git_commit": git_commit,
        "git_dirty": None if git_status is None else bool(git_status),
        "environment": {
            key: os.environ.get(key)
            for key in ["CC", "CXX", "CMAKE_BUILD_TYPE", "OMP_NUM_THREADS"]
            if os.environ.get(key) is not None
        },
    }


def write_provenance(
    output_root: Path, binary: Path, config_path: Path, manifest_path: Path
) -> None:
    provenance = provenance_payload(binary, config_path, manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    provenance_path = output_root / "provenance.json"
    index_path = output_root / "run_index.jsonl"
    if index_path.is_file() and index_path.stat().st_size > 0:
        if not provenance_path.is_file():
            raise RuntimeError(
                "results contains run_index but no provenance; use a new output_dir"
            )
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        compatibility_keys = [
            "binary_sha256", "config_sha256", "manifest_sha256", "pipeline_sha256"
        ]
        mismatches = [
            key for key in compatibility_keys if previous.get(key) != provenance.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "Refusing to mix existing runs with new provenance (changed "
                + ", ".join(mismatches)
                + "). Use a new output_dir or archive the existing results first."
            )
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.json")
    parser.add_argument("--binary", default="build/community_detect")
    parser.add_argument("--protocol", action="append", help="Run only this named protocol")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    arguments = parser.parse_args()

    config_path = resolve(arguments.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = resolve(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binary = resolve(arguments.binary)
    if not arguments.dry_run and not binary.is_file():
        raise FileNotFoundError(f"Executable not found: {binary}")

    selected_protocols = set(arguments.protocol or [])
    if selected_protocols:
        known_protocols = {protocol["name"] for protocol in config["protocols"]}
        unknown_protocols = selected_protocols - known_protocols
        if unknown_protocols:
            raise ValueError(
                "Unknown protocol: " + ", ".join(sorted(unknown_protocols))
            )
        config = {
            **config,
            "protocols": [
                protocol for protocol in config["protocols"]
                if protocol["name"] in selected_protocols
            ],
        }
    jobs = build_jobs(config, manifest)
    identifiers = [job.job_id for job in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Experiment matrix produced duplicate job_ids")
    if not jobs:
        raise RuntimeError("Configuration produced no jobs")
    if arguments.max_jobs is not None:
        if arguments.max_jobs < 0:
            raise ValueError("--max-jobs must be non-negative")
        jobs = jobs[: arguments.max_jobs]

    output_root = resolve(config.get("output_dir", "results"))
    if not arguments.dry_run:
        write_provenance(output_root, binary, config_path, manifest_path)

    index_path = output_root / "run_index.jsonl"
    existing_records: dict[str, dict[str, Any]] = {}
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    prior = json.loads(line)
                    existing_records[prior["job_id"]] = prior

    records: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")

    for index, job in enumerate(jobs, start=1):
        command = command_for_job(binary, job)
        reason = skip_reason(job, config)
        summary_path = resolve(job.output_dir) / "summary.json"
        record: dict[str, Any] = {
            **asdict(job),
            "options": list(job.options),
            "summary": relative(summary_path),
            "command": shlex.join(command),
        }
        if reason is not None:
            record.update(status="skipped_resource_guard", reason=reason)
            records.append(record)
            print(f"[{index}/{len(jobs)}] SKIP {job.job_id}: {reason}")
            continue
        prior = existing_records.get(job.job_id)
        if (arguments.resume
            and prior is not None
            and prior.get("status") in {"ok", "existing"}
            and valid_existing_summary(summary_path, job)):
            record.update(status="existing", wall_seconds=prior.get("wall_seconds"))
            records.append(record)
            print(f"[{index}/{len(jobs)}] KEEP {job.job_id}")
            continue
        if arguments.dry_run:
            record.update(status="dry_run")
            records.append(record)
            print(shlex.join(command))
            continue

        print(f"[{index}/{len(jobs)}] RUN  {job.job_id} {job.dataset}/{job.algorithm}", flush=True)
        started = time.perf_counter()
        process = subprocess.run(
            command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=environment, check=False,
        )
        wall_seconds = time.perf_counter() - started
        record["wall_seconds"] = wall_seconds
        if process.returncode == 0 and summary_path.is_file():
            record["status"] = "ok"
        else:
            record.update(
                status="failed",
                return_code=process.returncode,
                stderr=process.stderr[-8000:],
                stdout=process.stdout[-2000:],
            )
            failure_path = resolve(job.output_dir) / "failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        records.append(record)

    if arguments.dry_run:
        print(f"{len(records)} jobs")
        return 0

    merged_records = dict(existing_records)
    for record in records:
        merged_records[record["job_id"]] = record
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    with temporary_index.open("w", encoding="utf-8", newline="\n") as output:
        for record in sorted(
            merged_records.values(),
            key=lambda item: (item["protocol"], item["dataset"], item["scope_id"], item["job_id"]),
        ):
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_index.replace(index_path)
    print(relative(index_path))
    failed = sum(record["status"] == "failed" for record in records)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
