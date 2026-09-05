#!/usr/bin/env python3
"""End-to-end smoke test for prepare -> run -> summarize."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def require_failure(*arguments: str) -> None:
    process = subprocess.run(
        arguments, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode != 0
    assert "provenance" in process.stderr.lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="community_detection_pipeline_") as temporary:
        work = Path(temporary)
        dataset_config = work / "datasets.json"
        processed = work / "processed"
        dataset_config.write_text(
            json.dumps(
                {
                    "output_dir": str(processed),
                    "samples": [
                        {"name": "tiny", "size": 4, "seeds": [17, 29]}
                    ],
                    "datasets": [
                        {
                            "name": "toy",
                            "format": "gml",
                            "path": str(ROOT / "tests/fixtures/toy.gml"),
                            "reference_attributes": ["value"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        run(sys.executable, "tools/prepare_datasets.py", "--config", str(dataset_config))
        manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["datasets"][0]["n"] == 6
        assert manifest["datasets"][0]["m"] == 7
        assert len(manifest["datasets"][0]["sha256"]) == 64
        assert len(manifest["datasets"][0]["reference_sha256"]) == 64
        assert len(manifest["datasets"][0]["source_files"][0]["sha256"]) == 64
        assert len(manifest["datasets"][0]["samples"]) == 2

        results = work / "results"
        experiment_config = work / "experiments.json"
        experiment_config.write_text(
            json.dumps(
                {
                    "manifest": str(processed / "manifest.json"),
                    "output_dir": str(results),
                    "resource_guards": {
                        "girvan_newman_max_n": 100,
                        "girvan_newman_max_m": 100,
                        "spectral_max_edge_visits": 1_000_000,
                    },
                    "algorithm_options": {
                        "spectral": {
                            "laplacian": "normalized",
                            "dense_threshold": 32,
                            "kmeans_restarts": 2,
                        }
                    },
                    "dataset_settings": {
                        "toy": {"spectral_k": [2], "girvan_newman_k": [2]}
                    },
                    "protocols": [
                        {
                            "name": "smoke",
                            "scope": "full",
                            "algorithms": [
                                "girvan_newman", "spectral", "louvain",
                                "label_propagation",
                            ],
                            "algorithm_seeds": {
                                "girvan_newman": [42],
                                "spectral": [11, 37],
                                "louvain": [11, 37],
                                "label_propagation": [11, 37],
                            },
                            "dropout": [{"rate": 0.0, "seeds": [0]}],
                        },
                        {
                            "name": "sample_smoke",
                            "scope": "sample",
                            "sample_profile": "tiny",
                            "algorithms": ["spectral", "louvain"],
                            "algorithm_seeds": {
                                "spectral": [11, 37],
                                "louvain": [11, 37],
                            },
                            "dropout": [{"rate": 0.0, "seeds": [0]}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        run(
            sys.executable, "tools/run_experiments.py", "--config", str(experiment_config),
            "--binary", str(Path(arguments.binary).resolve()),
        )
        run(
            sys.executable, "tools/run_experiments.py", "--config", str(experiment_config),
            "--binary", str(Path(arguments.binary).resolve()), "--resume",
        )

        changed_config = work / "experiments_changed.json"
        changed = json.loads(experiment_config.read_text(encoding="utf-8"))
        changed["resource_guards"]["spectral_max_edge_visits"] = 999_999
        changed_config.write_text(json.dumps(changed), encoding="utf-8")
        require_failure(
            sys.executable, "tools/run_experiments.py", "--config", str(changed_config),
            "--binary", str(Path(arguments.binary).resolve()), "--resume",
        )
        run(
            sys.executable, "tools/summarize_results.py", "--results", str(results),
            "--manifest", str(processed / "manifest.json"),
        )

        runs = pd.read_csv(results / "tables/runs.csv")
        assert set(runs["algorithm"]) == {
            "girvan_newman", "spectral", "louvain", "label_propagation"
        }
        assert "external.ami_arithmetic" in runs.columns
        assert runs["external.ari"].notna().all()
        assert (runs["wall_seconds"] > 0.0).all()
        stability = pd.read_csv(results / "tables/stability_summary.csv")
        assert {"spectral", "louvain", "label_propagation"}.issubset(
            set(stability["algorithm"])
        )
        sample_aggregate = pd.read_csv(results / "tables/sample_aggregate.csv")
        assert set(sample_aggregate["sample_profile"]) == {"tiny"}
        sample_stability = pd.read_csv(
            results / "tables/sample_stability_summary.csv"
        )
        assert set(sample_stability["algorithm"]) == {"spectral", "louvain"}
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
