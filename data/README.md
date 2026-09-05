# Local datasets

Datasets are not redistributed with this repository. Download them from their
official sources and place them under `data/raw/` using the default layout below,
or copy `configs/datasets.example.json` to a `.local.json` file and adjust paths.

| Dataset key | Suggested source | Expected reference labels |
|---|---|---|
| `karate` | [Zachary's Karate Club, Mark Newman's GML archive](https://websites.umich.edu/~mejn/netdata/karate.zip) | Two factions embedded by the preparation tool |
| `football` | [American college football, Mark Newman's GML archive](https://websites.umich.edu/~mejn/netdata/football.zip) | GML node attribute `value` |
| `cond_mat` | [SNAP ca-CondMat](https://snap.stanford.edu/data/ca-CondMat.html) | None |
| `reddit2` | [GraphSAINT data instructions](https://github.com/GraphSAINT/GraphSAINT/blob/master/README.md) | `class_map.json` |

Confirm the licensing and redistribution terms at each source. Published node
and edge counts are useful sanity checks, but report statistics and checksums
from the manifest generated from the exact files used in an experiment.

```text
data/raw/
├── karate/karate.gml
├── football/football.gml
├── cond-mat/ca-CondMat.txt.gz
└── reddit2/
    ├── adj_full.npz
    ├── class_map.json
    ├── feats.npy       # retained with Reddit2; not read by this pipeline
    └── role.json       # retained with Reddit2; not read by this pipeline
```

Prepare all configured datasets with:

```bash
python3 tools/prepare_datasets.py --config configs/datasets.example.json
```

The command writes normalized `*.cdgraph` files, reference labels, registered
samples, ID mappings, and `data/processed/manifest.json`. Inspect the manifest
before running experiments: it records preprocessing decisions, graph statistics,
and SHA-256 hashes for every consumed source and generated graph.

Everything below `data/` except this README is ignored by Git. Do not commit raw
or processed datasets, derived samples, local path configurations, or labels
whose redistribution terms have not been verified.
