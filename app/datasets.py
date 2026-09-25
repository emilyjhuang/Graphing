"""Graph datasets: classic real-world networks, random generators, and user edge lists.

Every graph is returned with string node ids (JSON-friendly) and, where one
exists, a ground-truth partition stored on each node as the ``truth`` attribute.
"""

from __future__ import annotations

import io
import csv
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import networkx as nx

MAX_NODES = 2000
MAX_EDGES = 20000


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    default: float
    min: float
    max: float
    step: float = 1


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    description: str
    has_truth: bool
    params: tuple[Param, ...] = field(default_factory=tuple)


DATASETS: dict[str, Dataset] = {
    d.key: d
    for d in [
        Dataset(
            "karate",
            "Zachary's Karate Club",
            "34 members of a university karate club that split in two after a dispute "
            "(Zachary, 1977). Ground truth = which faction each member joined.",
            has_truth=True,
        ),
        Dataset(
            "lesmis",
            "Les Misérables",
            "Character co-appearance network from Victor Hugo's novel. Edge weight = "
            "number of chapters two characters share.",
            has_truth=False,
        ),
        Dataset(
            "florentine",
            "Florentine Families",
            "Marriage alliances between Renaissance Florentine families. The Medici's "
            "rise is a textbook betweenness-centrality story.",
            has_truth=False,
        ),
        Dataset(
            "sbm",
            "Stochastic Block Model",
            "Planted-partition random graph. Tune intra/inter-block edge probability "
            "to make communities easy or impossible to recover.",
            has_truth=True,
            params=(
                Param("blocks", "Blocks", 4, 2, 8),
                Param("block_size", "Nodes per block", 30, 5, 150),
                Param("p_in", "p(in)", 0.25, 0.01, 1, 0.01),
                Param("p_out", "p(out)", 0.02, 0.0, 1, 0.005),
                Param("seed", "Seed", 7, 0, 9999),
            ),
        ),
        Dataset(
            "ba",
            "Barabási–Albert",
            "Preferential attachment: new nodes link to already-popular nodes, "
            "producing hubs and a power-law degree tail.",
            has_truth=False,
            params=(
                Param("n", "Nodes", 150, 10, MAX_NODES),
                Param("m", "Edges per new node", 2, 1, 10),
                Param("seed", "Seed", 7, 0, 9999),
            ),
        ),
        Dataset(
            "ws",
            "Watts–Strogatz",
            "Small-world ring lattice with random rewiring: high clustering and short "
            "paths at the same time.",
            has_truth=False,
            params=(
                Param("n", "Nodes", 120, 10, MAX_NODES),
                Param("k", "Neighbours", 6, 2, 20, 2),
                Param("p", "Rewire prob.", 0.1, 0, 1, 0.01),
                Param("seed", "Seed", 7, 0, 9999),
            ),
        ),
        Dataset(
            "custom",
            "Your Edge List",
            "Paste an edge list (source,target[,weight] per line). Optionally add a "
            "`# labels` section with node,label lines to enable ground-truth scoring.",
            has_truth=False,
        ),
    ]
}


def _param_values(ds: Dataset, params: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for p in ds.params:
        raw = params.get(p.name, p.default)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise DatasetError(f"Parameter {p.name!r} must be a number")
        v = min(max(v, p.min), p.max)
        values[p.name] = int(v) if float(p.step).is_integer() and float(p.default).is_integer() else v
    return values


def _relabel(g: nx.Graph) -> nx.Graph:
    return nx.relabel_nodes(g, {n: str(n) for n in g.nodes}, copy=True)


def _check_size(g: nx.Graph) -> None:
    if g.number_of_nodes() > MAX_NODES:
        raise DatasetError(f"Graph has {g.number_of_nodes()} nodes; limit is {MAX_NODES}")
    if g.number_of_edges() > MAX_EDGES:
        raise DatasetError(f"Graph has {g.number_of_edges()} edges; limit is {MAX_EDGES}")


def parse_edge_list(text: str) -> nx.Graph:
    """Parse ``a,b[,w]`` lines (comma, tab or whitespace separated).

    A line ``# labels`` switches to ``node,label`` mode for ground truth.
    """
    g = nx.Graph()
    labels: dict[str, str] = {}
    mode = "edges"
    for lineno, line in enumerate(io.StringIO(text), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.lstrip("#").strip().lower() == "labels":
                mode = "labels"
            continue
        sep = "," if "," in line else ("\t" if "\t" in line else None)
        parts = next(csv.reader([line], delimiter=sep)) if sep else line.split()
        parts = [p.strip() for p in parts if p.strip()]
        if mode == "labels":
            if len(parts) < 2:
                raise DatasetError(f"Line {lineno}: expected node,label")
            labels[parts[0]] = parts[1]
            continue
        if len(parts) < 2:
            raise DatasetError(f"Line {lineno}: expected source,target[,weight]")
        if parts[0].lower() in {"source", "src", "from"} and lineno == 1:
            continue  # header row
        u, v = parts[0], parts[1]
        if u == v:
            continue
        w = 1.0
        if len(parts) >= 3:
            try:
                w = float(parts[2])
            except ValueError:
                raise DatasetError(f"Line {lineno}: weight {parts[2]!r} is not a number")
            if w <= 0:
                raise DatasetError(f"Line {lineno}: weights must be positive")
        g.add_edge(u, v, weight=w)
        if g.number_of_edges() > MAX_EDGES:
            raise DatasetError(f"Edge list exceeds {MAX_EDGES} edges")
    if g.number_of_edges() == 0:
        raise DatasetError("Edge list is empty")
    _check_size(g)
    if labels:
        missing = [n for n in g.nodes if n not in labels]
        if missing:
            raise DatasetError(
                f"Labels section is missing {len(missing)} node(s), e.g. {missing[0]!r}"
            )
        for n in g.nodes:
            g.nodes[n]["truth"] = labels[n]
    return g


@lru_cache(maxsize=64)
def _build_cached(key: str, frozen_params: tuple[tuple[str, Any], ...]) -> nx.Graph:
    p = dict(frozen_params)
    if key == "karate":
        g = _relabel(nx.karate_club_graph())
        for n, d in g.nodes(data=True):
            d["truth"] = d.pop("club")
        for _, _, d in g.edges(data=True):
            d["weight"] = float(d.get("weight", 1))
        return g
    if key == "lesmis":
        return _relabel(nx.les_miserables_graph())
    if key == "florentine":
        return _relabel(nx.florentine_families_graph())
    if key == "sbm":
        sizes = [p["block_size"]] * p["blocks"]
        probs = [
            [p["p_in"] if i == j else p["p_out"] for j in range(p["blocks"])]
            for i in range(p["blocks"])
        ]
        g = nx.stochastic_block_model(sizes, probs, seed=p["seed"])
        for n, d in g.nodes(data=True):
            d["truth"] = f"Block {d.pop('block') + 1}"
        g.graph.pop("partition", None)
        return _relabel(g)
    if key == "ba":
        if p["m"] >= p["n"]:
            raise DatasetError("Edges per new node must be smaller than the node count")
        return _relabel(nx.barabasi_albert_graph(p["n"], p["m"], seed=p["seed"]))
    if key == "ws":
        if p["k"] >= p["n"]:
            raise DatasetError("Neighbours must be smaller than the node count")
        return _relabel(nx.watts_strogatz_graph(p["n"], p["k"], p["p"], seed=p["seed"]))
    raise DatasetError(f"Unknown dataset {key!r}")


def build_graph(spec: dict[str, Any]) -> nx.Graph:
    """Build a fresh (mutable) graph from a dataset spec.

    ``spec`` = ``{"key": str, "params": {...}, "edges": str}``.
    """
    key = spec.get("key", "karate")
    if key not in DATASETS:
        raise DatasetError(f"Unknown dataset {key!r}")
    if key == "custom":
        return parse_edge_list(spec.get("edges") or "")
    values = _param_values(DATASETS[key], spec.get("params") or {})
    g = _build_cached(key, tuple(sorted(values.items())))
    _check_size(g)
    return g.copy()


def has_truth(g: nx.Graph) -> bool:
    return g.number_of_nodes() > 0 and all("truth" in d for _, d in g.nodes(data=True))


def catalog() -> list[dict[str, Any]]:
    return [
        {
            "key": d.key,
            "name": d.name,
            "description": d.description,
            "has_truth": d.has_truth,
            "params": [p.__dict__ for p in d.params],
        }
        for d in DATASETS.values()
    ]
