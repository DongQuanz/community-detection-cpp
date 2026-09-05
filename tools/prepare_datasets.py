#!/usr/bin/env python3
"""Normalize Karate, Football, Cond-Mat, and GraphSAINT/Reddit2 to CDGRPH1.

This script is intentionally separate from the C++ core: SciPy reads Reddit2's
CSR `.npz` directly, while the executable only needs a compact, stable binary
format. All generated data is written below data/processed and ignored by Git.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


MAGIC = b"CDGRPH1\0"
HEADER = struct.Struct("<8sQQB7x")
EDGE_DTYPE = np.dtype([("u", "<u4"), ("v", "<u4")])

# Observed groups after the split in the NetworkX/Zachary Karate graph.
KARATE_REFERENCE = np.asarray(
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ],
    dtype=np.int64,
)


@dataclass
class LoadedDataset:
    adjacency: sp.csr_matrix
    node_ids: list[str]
    reference: np.ndarray | None
    source_format: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def normalize_reference(values: Iterable[Any]) -> np.ndarray:
    raw = [str(value) for value in values]
    mapping = {value: index for index, value in enumerate(sorted(set(raw)))}
    return np.asarray([mapping[value] for value in raw], dtype=np.int64)


def parse_gml_attributes(block: str) -> dict[str, str]:
    pattern = re.compile(r"([A-Za-z_][\w]*)\s+(?:\"((?:\\.|[^\"])*)\"|([^\s\]]+))")
    attributes: dict[str, str] = {}
    for match in pattern.finditer(block):
        quoted, bare = match.group(2), match.group(3)
        attributes[match.group(1)] = quoted if quoted is not None else bare
    return attributes


def load_gml(specification: dict[str, Any]) -> LoadedDataset:
    path = resolve_path(specification["path"])
    text = path.read_text(encoding="utf-8", errors="replace")
    node_blocks = re.findall(r"\bnode\s*\[(.*?)\]", text, flags=re.DOTALL)
    edge_blocks = re.findall(r"\bedge\s*\[(.*?)\]", text, flags=re.DOTALL)
    if not node_blocks or not edge_blocks:
        raise ValueError(f"Could not read GML nodes or edges from {path}")

    nodes: dict[str, dict[str, str]] = {}
    for block in node_blocks:
        attributes = parse_gml_attributes(block)
        if "id" not in attributes:
            raise ValueError(f"GML node has no id in {path}")
        if attributes["id"] in nodes:
            raise ValueError(f"Duplicate GML node id={attributes['id']}")
        nodes[attributes["id"]] = attributes

    def sort_key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    ordered_ids = sorted(nodes, key=sort_key)
    index = {node_id: position for position, node_id in enumerate(ordered_ids)}
    rows: list[int] = []
    columns: list[int] = []
    for block in edge_blocks:
        attributes = parse_gml_attributes(block)
        source, target = attributes.get("source"), attributes.get("target")
        if source not in index or target not in index:
            raise ValueError("GML edge references a missing node")
        rows.append(index[source])
        columns.append(index[target])

    adjacency = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(ordered_ids), len(ordered_ids)),
    ).tocsr()

    reference: np.ndarray | None = None
    reference_mode = specification.get("reference")
    if reference_mode == "builtin_karate":
        if len(ordered_ids) != KARATE_REFERENCE.size:
            raise ValueError("builtin_karate requires exactly 34 nodes")
        try:
            numeric_ids = [int(node_id) for node_id in ordered_ids]
        except ValueError as error:
            raise ValueError("builtin_karate requires numeric node ids") from error
        if numeric_ids not in [list(range(34)), list(range(1, 35))]:
            raise ValueError(
                "builtin_karate requires contiguous GML node ids 0..33 or 1..34"
            )
        reference = KARATE_REFERENCE.copy()
    else:
        attributes_to_try = specification.get("reference_attributes", ["value", "club"])
        for attribute in attributes_to_try:
            if all(attribute in nodes[node_id] for node_id in ordered_ids):
                reference = normalize_reference(
                    nodes[node_id][attribute] for node_id in ordered_ids
                )
                break

    return LoadedDataset(adjacency, ordered_ids, reference, "gml")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_edge_list(specification: dict[str, Any]) -> LoadedDataset:
    path = resolve_path(specification["path"])
    comments = tuple(specification.get("comment_prefixes", ["#", "%"]))
    raw_edges: list[tuple[str, str]] = []
    node_ids: set[str] = set()
    with open_text(path) as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(comments):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(f"Line {line_number} of {path} has fewer than two columns")
            left, right = fields[0], fields[1]
            if left == right:
                continue
            raw_edges.append((left, right))
            node_ids.add(left)
            node_ids.add(right)

    def sort_key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    ordered_ids = sorted(node_ids, key=sort_key)
    index = {node_id: position for position, node_id in enumerate(ordered_ids)}
    rows = np.fromiter((index[left] for left, _ in raw_edges), dtype=np.int64)
    columns = np.fromiter((index[right] for _, right in raw_edges), dtype=np.int64)
    adjacency = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, columns)),
        shape=(len(ordered_ids), len(ordered_ids)),
    ).tocsr()
    return LoadedDataset(adjacency, ordered_ids, None, "edge_list")


def read_graphsaint_adjacency(path: Path) -> sp.csr_matrix:
    try:
        return sp.load_npz(path).tocsr()
    except (ValueError, KeyError):
        archive = np.load(path, allow_pickle=False)
        required = {"data", "indices", "indptr", "shape"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path} is not a GraphSAINT/Reddit2 CSR npz file")
        shape = tuple(int(value) for value in archive["shape"])
        return sp.csr_matrix(
            (archive["data"], archive["indices"], archive["indptr"]), shape=shape
        )


def load_graphsaint(specification: dict[str, Any]) -> LoadedDataset:
    root = resolve_path(specification.get("root", specification.get("path")))
    adjacency_path = root if root.suffix == ".npz" else root / "adj_full.npz"
    metadata_root = adjacency_path.parent
    adjacency = read_graphsaint_adjacency(adjacency_path)
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("GraphSAINT adjacency must be square")

    class_map_path = metadata_root / specification.get("class_map", "class_map.json")
    reference: np.ndarray | None = None
    if class_map_path.exists():
        class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
        labels = np.full(adjacency.shape[0], -1, dtype=np.int64)
        for raw_node, value in class_map.items():
            node = int(raw_node)
            if isinstance(value, list):
                positives = np.flatnonzero(np.asarray(value) > 0)
                if positives.size != 1:
                    raise ValueError(
                        "Multi-label references are incompatible with disjoint partitions"
                    )
                labels[node] = int(positives[0])
            else:
                labels[node] = int(value)
        if np.any(labels < 0):
            raise ValueError("class_map.json does not label every node")
        reference = normalize_reference(labels)

    node_ids = [str(index) for index in range(adjacency.shape[0])]
    return LoadedDataset(adjacency, node_ids, reference, "graphsaint_reddit2")


def sanitize(dataset: LoadedDataset, keep_largest_component: bool) -> LoadedDataset:
    adjacency = dataset.adjacency.tocsr(copy=True)
    adjacency.sum_duplicates()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    adjacency.data = np.ones(adjacency.nnz, dtype=np.uint8)
    adjacency.sum_duplicates()
    adjacency.sort_indices()

    node_ids = dataset.node_ids
    reference = dataset.reference
    if keep_largest_component and adjacency.shape[0] > 0:
        component_count, labels = connected_components(
            adjacency, directed=False, return_labels=True
        )
        if component_count > 1:
            sizes = np.bincount(labels)
            largest = int(np.flatnonzero(sizes == sizes.max())[0])
            selected = np.flatnonzero(labels == largest)
            adjacency = adjacency[selected][:, selected].tocsr()
            node_ids = [node_ids[index] for index in selected]
            reference = None if reference is None else reference[selected]

    if adjacency.shape[0] >= np.iinfo(np.uint32).max:
        raise ValueError("Node count exceeds the CDGRPH1 uint32 limit")
    return LoadedDataset(adjacency, node_ids, reference, dataset.source_format)


def write_cdgraph(adjacency: sp.csr_matrix, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    adjacency = adjacency.tocsr()
    adjacency.sort_indices()
    edge_count = 0
    buffered_u: list[np.ndarray] = []
    buffered_v: list[np.ndarray] = []
    buffered_count = 0

    def flush(output) -> None:
        nonlocal buffered_count
        if buffered_count == 0:
            return
        left = np.concatenate(buffered_u)
        right = np.concatenate(buffered_v)
        records = np.empty(buffered_count, dtype=EDGE_DTYPE)
        records["u"] = left
        records["v"] = right
        records.tofile(output)
        buffered_u.clear()
        buffered_v.clear()
        buffered_count = 0

    with temporary.open("wb") as output:
        output.write(HEADER.pack(MAGIC, adjacency.shape[0], 0, 0))
        for raw_u in range(adjacency.shape[0]):
            start, end = adjacency.indptr[raw_u : raw_u + 2]
            selected = adjacency.indices[start:end]
            selected = selected[selected > raw_u]
            if selected.size == 0:
                continue
            buffered_u.append(np.full(selected.size, raw_u, dtype=np.uint32))
            buffered_v.append(selected.astype(np.uint32, copy=False))
            buffered_count += int(selected.size)
            edge_count += int(selected.size)
            if buffered_count >= 1_000_000:
                flush(output)
        flush(output)
        output.seek(0)
        output.write(HEADER.pack(MAGIC, adjacency.shape[0], edge_count, 0))
    temporary.replace(path)
    return edge_count


def write_labels(labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write("node_id\tlabel\n")
        for node, label in enumerate(labels):
            output.write(f"{node}\t{int(label)}\n")
    temporary.replace(path)


def write_node_ids(node_ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write("node_id\toriginal_id\n")
        for node, original in enumerate(node_ids):
            escaped = original.replace("\t", " ").replace("\n", " ")
            output.write(f"{node}\t{escaped}\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_records(specification: dict[str, Any]) -> list[dict[str, Any]]:
    data_format = specification["format"]
    if data_format in {"gml", "edge_list"}:
        paths = [resolve_path(specification["path"])]
    elif data_format in {"graphsaint", "reddit2"}:
        root = resolve_path(specification.get("root", specification.get("path")))
        adjacency = root if root.suffix == ".npz" else root / "adj_full.npz"
        class_map = adjacency.parent / specification.get("class_map", "class_map.json")
        paths = [adjacency] + ([class_map] if class_map.is_file() else [])
    else:
        paths = []
    return [
        {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def snowball_nodes(adjacency: sp.csr_matrix, size: int, seed: int) -> np.ndarray:
    n = adjacency.shape[0]
    if size >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    degree = np.diff(adjacency.indptr)
    eligible = np.flatnonzero(degree > 0)
    if eligible.size == 0:
        return np.sort(rng.choice(n, size=size, replace=False))

    visited = np.zeros(n, dtype=bool)
    selected: list[int] = []
    queue: collections.deque[int] = collections.deque()
    while len(selected) < size:
        remaining_roots = eligible[~visited[eligible]]
        if remaining_roots.size == 0:
            remaining_roots = np.flatnonzero(~visited)
        root = int(rng.choice(remaining_roots))
        visited[root] = True
        selected.append(root)
        queue.append(root)
        while queue and len(selected) < size:
            node = queue.popleft()
            start, end = adjacency.indptr[node : node + 2]
            neighbors = adjacency.indices[start:end].copy()
            rng.shuffle(neighbors)
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                selected.append(neighbor)
                queue.append(neighbor)
                if len(selected) == size:
                    break
    return np.sort(np.asarray(selected, dtype=np.int64))


def graph_metadata(adjacency: sp.csr_matrix, edge_count: int) -> dict[str, Any]:
    degree = np.diff(adjacency.indptr)
    components, _ = connected_components(adjacency, directed=False, return_labels=True)
    return {
        "n": int(adjacency.shape[0]),
        "m": int(edge_count),
        "components": int(components),
        "isolated_nodes": int(np.count_nonzero(degree == 0)),
        "min_degree": int(degree.min()) if degree.size else 0,
        "max_degree": int(degree.max()) if degree.size else 0,
        "mean_degree": float(degree.mean()) if degree.size else 0.0,
    }


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def process_dataset(
    specification: dict[str, Any], output_root: Path, global_samples: list[dict[str, Any]]
) -> dict[str, Any]:
    name = specification["name"]
    data_format = specification["format"]
    if data_format == "gml":
        loaded = load_gml(specification)
    elif data_format == "edge_list":
        loaded = load_edge_list(specification)
    elif data_format in {"graphsaint", "reddit2"}:
        loaded = load_graphsaint(specification)
    else:
        raise ValueError(f"Unsupported format: {data_format}")

    loaded = sanitize(loaded, bool(specification.get("largest_component", False)))
    dataset_directory = output_root / name
    graph_path = dataset_directory / "full.cdgraph"
    labels_path = dataset_directory / "reference.tsv"
    ids_path = dataset_directory / "node_ids.tsv"
    edge_count = write_cdgraph(loaded.adjacency, graph_path)
    write_node_ids(loaded.node_ids, ids_path)
    if loaded.reference is not None:
        write_labels(loaded.reference, labels_path)

    entry: dict[str, Any] = {
        "name": name,
        "source_format": loaded.source_format,
        "source_files": source_file_records(specification),
        "graph": relative(graph_path),
        "reference": relative(labels_path) if loaded.reference is not None else None,
        "node_ids": relative(ids_path),
        "sha256": sha256(graph_path),
        "reference_sha256": sha256(labels_path) if loaded.reference is not None else None,
        "node_ids_sha256": sha256(ids_path),
        **graph_metadata(loaded.adjacency, edge_count),
        "samples": [],
    }

    sample_specs = specification.get("samples", global_samples)
    seen_samples: set[tuple[str, int]] = set()
    for sample_spec in sample_specs:
        sample_size = int(sample_spec["size"])
        sample_name = str(sample_spec["name"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_name):
            raise ValueError(f"Unsafe sample profile name: {sample_name}")
        if sample_size <= 0:
            raise ValueError(f"Sample size must be positive: {sample_size}")
        if sample_size >= loaded.adjacency.shape[0]:
            continue
        for seed in sample_spec.get("seeds", [0]):
            parsed_seed = int(seed)
            if parsed_seed < 0:
                raise ValueError(f"Sample seed must be non-negative: {parsed_seed}")
            sample_key = (sample_name, parsed_seed)
            if sample_key in seen_samples:
                raise ValueError(f"Duplicate sample profile/seed: {sample_key}")
            seen_samples.add(sample_key)
            nodes = snowball_nodes(loaded.adjacency, sample_size, parsed_seed)
            sample_adjacency = loaded.adjacency[nodes][:, nodes].tocsr()
            sample_directory = (
                dataset_directory / "samples" / sample_name / f"seed_{parsed_seed}"
            )
            sample_graph_path = sample_directory / "graph.cdgraph"
            sample_reference_path = sample_directory / "reference.tsv"
            sample_ids_path = sample_directory / "node_ids.tsv"
            sample_edges = write_cdgraph(sample_adjacency, sample_graph_path)
            write_node_ids([loaded.node_ids[index] for index in nodes], sample_ids_path)
            if loaded.reference is not None:
                write_labels(loaded.reference[nodes], sample_reference_path)
            entry["samples"].append(
                {
                    "profile": sample_name,
                    "seed": parsed_seed,
                    "sampling": "randomized_bfs_induced",
                    "graph": relative(sample_graph_path),
                    "reference": (
                        relative(sample_reference_path) if loaded.reference is not None else None
                    ),
                    "node_ids": relative(sample_ids_path),
                    "sha256": sha256(sample_graph_path),
                    "reference_sha256": (
                        sha256(sample_reference_path) if loaded.reference is not None else None
                    ),
                    "node_ids_sha256": sha256(sample_ids_path),
                    **graph_metadata(sample_adjacency, sample_edges),
                }
            )

    metadata_path = dataset_directory / "metadata.json"
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets.example.json")
    parser.add_argument("--only", action="append", help="Process only this dataset (repeatable)")
    arguments = parser.parse_args()

    config_path = resolve_path(arguments.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve_path(config.get("output_dir", "data/processed"))
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(arguments.only or [])
    configured_names = [str(specification["name"]) for specification in config["datasets"]]
    if len(set(configured_names)) != len(configured_names):
        raise ValueError("Dataset names in the configuration must be unique")
    unknown = selected - set(configured_names)
    if unknown:
        raise ValueError("Unknown dataset: " + ", ".join(sorted(unknown)))
    entries = []
    for specification in config["datasets"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(specification["name"])):
            raise ValueError(f"Unsafe dataset name: {specification['name']}")
        if selected and specification["name"] not in selected:
            continue
        print(f"[prepare] {specification['name']}", flush=True)
        entries.append(process_dataset(specification, output_root, config.get("samples", [])))

    manifest = {
        "schema_version": 1,
        "generator": "tools/prepare_datasets.py",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "datasets": entries,
    }
    manifest_path = output_root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(relative(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
