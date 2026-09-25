"""Descriptive network analysis: summary statistics, centrality, communities,
shortest paths and robustness under node removal."""

from __future__ import annotations

import random
import time
from typing import Any, Callable

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .datasets import has_truth

EXACT_LIMIT = 600  # above this many nodes, sample for O(n·m) metrics


# ---------------------------------------------------------------- statistics


def _largest_cc(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        return g
    return g.subgraph(max(nx.connected_components(g), key=len))


def _path_stats(lcc: nx.Graph, seed: int = 0) -> tuple[float, int, bool]:
    """Average shortest-path length and diameter of a connected graph.

    Exact for small graphs; for large ones, BFS from a random sample of sources
    (the diameter is then a lower bound, flagged by ``approx``).
    """
    nodes = list(lcc.nodes)
    approx = len(nodes) > EXACT_LIMIT
    sources = random.Random(seed).sample(nodes, 150) if approx else nodes
    total = count = diameter = 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(lcc, s)
        total += sum(lengths.values())
        count += len(lengths) - 1
        diameter = max(diameter, max(lengths.values()))
    return (total / count if count else 0.0), diameter, approx


def summary(g: nx.Graph) -> dict[str, Any]:
    n, m = g.number_of_nodes(), g.number_of_edges()
    comps = nx.number_connected_components(g)
    lcc = _largest_cc(g)
    avg_path, diameter, approx = _path_stats(lcc)
    degrees = [d for _, d in g.degree()]
    try:
        assortativity = nx.degree_assortativity_coefficient(g)
        assortativity = None if np.isnan(assortativity) else assortativity
    except (ZeroDivisionError, ValueError):
        assortativity = None
    hist = np.bincount(degrees) if degrees else np.array([])
    return {
        "nodes": n,
        "edges": m,
        "density": nx.density(g),
        "components": comps,
        "lcc_fraction": lcc.number_of_nodes() / n if n else 0,
        "avg_degree": float(np.mean(degrees)) if degrees else 0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "avg_clustering": nx.average_clustering(g),
        "transitivity": nx.transitivity(g),
        "assortativity": assortativity,
        "avg_path_length": avg_path,
        "diameter": diameter,
        "path_stats_approx": approx,
        "degree_histogram": [int(c) for c in hist],
        "weighted": any(d.get("weight", 1) != 1 for _, _, d in g.edges(data=True)),
        "has_truth": has_truth(g),
    }


# ---------------------------------------------------------------- centrality


def _eigenvector(g: nx.Graph) -> dict[str, float]:
    try:
        return nx.eigenvector_centrality_numpy(g)
    except (nx.NetworkXException, ValueError, TypeError):
        return {n: 0.0 for n in g.nodes}


def centralities(g: nx.Graph) -> dict[str, dict[str, float]]:
    n = g.number_of_nodes()
    k = min(n, 200) if n > EXACT_LIMIT else None
    return {
        "degree": nx.degree_centrality(g),
        "betweenness": nx.betweenness_centrality(g, k=k, seed=0),
        "closeness": nx.closeness_centrality(g),
        "eigenvector": _eigenvector(g),
        "pagerank": nx.pagerank(g),
        "clustering": nx.clustering(g),
        "core": {k_: float(v) for k_, v in nx.core_number(g).items()},
    }


# ---------------------------------------------------------------- communities


def spectral_communities(g: nx.Graph, k: int | None = None, seed: int = 0) -> list[set]:
    """Normalised spectral clustering (Ng–Jordan–Weiss).

    Embeds nodes with the bottom eigenvectors of the symmetric normalised
    Laplacian, row-normalises, and runs k-means. When ``k`` is not given it is
    chosen by the largest eigengap among the first 10 eigenvalues.
    """
    nodes = list(g.nodes)
    n = len(nodes)
    if n < 3:
        return [set(nodes)]
    lap = nx.normalized_laplacian_matrix(g, nodelist=nodes).astype(float)
    n_eig = min(n - 1, max(k or 0, 10) + 1)
    if n <= 400:
        vals, vecs = np.linalg.eigh(lap.toarray())
        vals, vecs = vals[:n_eig], vecs[:, :n_eig]
    else:
        vals, vecs = eigsh(lap, k=n_eig, sigma=-1e-3, which="LM")
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
    if k is None:
        gaps = np.diff(vals[: min(len(vals), 11)])
        k = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 2
    k = max(2, min(k, n - 1))
    emb = vecs[:, :k]
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)
    groups: dict[int, set] = {}
    for node, lab in zip(nodes, labels):
        groups.setdefault(int(lab), set()).add(node)
    return list(groups.values())


def _truth_k(g: nx.Graph) -> int | None:
    return len({d["truth"] for _, d in g.nodes(data=True)}) if has_truth(g) else None


COMMUNITY_ALGORITHMS: dict[str, tuple[str, Callable[[nx.Graph], list[set]]]] = {
    "louvain": ("Louvain (modularity)", lambda g: nx.community.louvain_communities(g, seed=0)),
    "greedy": ("Clauset–Newman–Moore greedy", lambda g: list(nx.community.greedy_modularity_communities(g))),
    "label_propagation": ("Label propagation", lambda g: list(nx.community.asyn_lpa_communities(g, seed=0))),
    "spectral": ("Spectral clustering (eigengap k)", lambda g: spectral_communities(g)),
    "spectral_k": ("Spectral clustering (true k)", lambda g: spectral_communities(g, k=_truth_k(g))),
}


def communities(g: nx.Graph, algorithm: str) -> dict[str, Any]:
    if algorithm not in COMMUNITY_ALGORITHMS:
        raise ValueError(f"Unknown community algorithm {algorithm!r}")
    if algorithm == "spectral_k" and not has_truth(g):
        algorithm = "spectral"
    label, fn = COMMUNITY_ALGORITHMS[algorithm]
    parts = sorted(fn(g), key=len, reverse=True)  # community 0 = largest
    membership = {node: i for i, part in enumerate(parts) for node in part}
    result: dict[str, Any] = {
        "algorithm": algorithm,
        "label": label,
        "count": len(parts),
        "sizes": [len(p) for p in parts],
        "modularity": nx.community.modularity(g, parts),
        "membership": membership,
        "nmi": None,
        "ari": None,
    }
    if has_truth(g):
        nodes = list(g.nodes)
        truth = [g.nodes[n]["truth"] for n in nodes]
        pred = [membership[n] for n in nodes]
        result["nmi"] = normalized_mutual_info_score(truth, pred)
        result["ari"] = adjusted_rand_score(truth, pred)
        truth_parts: dict[Any, set] = {}
        for n in nodes:
            truth_parts.setdefault(g.nodes[n]["truth"], set()).add(n)
        result["truth_modularity"] = nx.community.modularity(g, truth_parts.values())
    return result


# ---------------------------------------------------------------- paths


def shortest_paths(g: nx.Graph, source: str, target: str, weighted: bool, limit: int = 10) -> dict[str, Any]:
    for node in (source, target):
        if node not in g:
            raise ValueError(f"Node {node!r} is not in the graph")
    # Edge weights are tie strengths, so the traversal cost is their inverse.
    weight = (lambda u, v, d: 1.0 / d.get("weight", 1.0)) if weighted else None
    try:
        gen = nx.all_shortest_paths(g, source, target, weight=weight)
        paths = []
        for p in gen:
            paths.append(p)
            if len(paths) >= limit:
                break
    except nx.NetworkXNoPath:
        return {"paths": [], "hops": None, "cost": None, "truncated": False}
    cost = None
    if weighted:
        cost = sum(1.0 / g[u][v].get("weight", 1.0) for u, v in zip(paths[0], paths[0][1:]))
    return {
        "paths": paths,
        "hops": len(paths[0]) - 1,
        "cost": cost,
        "truncated": len(paths) >= limit,
    }


# ---------------------------------------------------------------- robustness


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return self.size[ra]
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return self.size[ra]


def lcc_curve(g: nx.Graph, removal_order: list) -> list[float]:
    """Largest-component fraction after removing each prefix of ``removal_order``.

    Runs in near-linear time by replaying the removals backwards as node
    *additions* with a union-find, instead of recomputing components n times.
    ``curve[i]`` is the LCC fraction after ``i`` nodes have been removed.
    """
    n = g.number_of_nodes()
    index = {node: i for i, node in enumerate(removal_order)}
    dsu = _DSU(n)
    present = [False] * n
    best = 0
    curve = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        node = removal_order[i]
        present[i] = True
        best = max(best, 1)
        for nb in g.neighbors(node):
            j = index[nb]
            if present[j]:
                best = max(best, dsu.union(i, j))
        curve[i] = best / n
    return curve


def _adaptive_degree_order(g: nx.Graph) -> list:
    """Repeatedly remove the current highest-degree node (recomputed each step)."""
    h = g.copy()
    order = []
    buckets: dict[int, set] = {}
    deg = dict(h.degree())
    for node, d in deg.items():
        buckets.setdefault(d, set()).add(node)
    max_d = max(deg.values(), default=0)
    while deg:
        while max_d > 0 and not buckets.get(max_d):
            max_d -= 1
        node = min(buckets[max_d], key=str)  # deterministic tie-break
        buckets[max_d].remove(node)
        order.append(node)
        for nb in list(h.neighbors(node)):
            d = deg[nb]
            buckets[d].remove(nb)
            deg[nb] = d - 1
            buckets.setdefault(d - 1, set()).add(nb)
        h.remove_node(node)
        del deg[node]
    return order


def robustness(g: nx.Graph, cent: dict[str, dict[str, float]] | None = None, trials: int = 20) -> dict[str, Any]:
    """Simulate random failures vs targeted attacks.

    Reports the LCC-fraction curve for each strategy plus the robustness index
    R = mean LCC fraction over all removal steps (Schneider et al., PNAS 2011).
    """
    cent = cent or centralities(g)
    nodes = list(g.nodes)
    n = len(nodes)

    def by(metric: str) -> list:
        return sorted(nodes, key=lambda v: (-cent[metric][v], str(v)))

    rng = random.Random(0)
    random_orders = []
    for _ in range(trials):
        order = nodes[:]
        rng.shuffle(order)
        random_orders.append(order)
    orders = {
        "random": random_orders[0],
        "degree": by("degree"),
        "adaptive_degree": _adaptive_degree_order(g),
        "betweenness": by("betweenness"),
        "pagerank": by("pagerank"),
    }
    strategies = {k: lcc_curve(g, o) for k, o in orders.items() if k != "random"}
    strategies = {"random": np.mean([lcc_curve(g, o) for o in random_orders], axis=0).tolist(), **strategies}
    # Downsample for transport; keep endpoints.
    step = max(1, n // 200)
    idx = sorted(set(range(0, n + 1, step)) | {n})
    return {
        "n": n,
        "x": [i / n for i in idx],
        "curves": {k: [v[i] for i in idx] for k, v in strategies.items()},
        "R": {k: float(np.mean(v[1:])) if n else 0.0 for k, v in strategies.items()},
        "orders": orders,  # the random order is one sample; its curve is the mean of all trials
    }


def compare_communities(g: nx.Graph) -> dict[str, Any]:
    rows = []
    for key in COMMUNITY_ALGORITHMS:
        if key == "spectral_k" and not has_truth(g):
            continue
        start = time.perf_counter()
        r = communities(g, key)
        rows.append({
            "key": key,
            "label": r["label"],
            "count": r["count"],
            "modularity": r["modularity"],
            "nmi": r["nmi"],
            "ari": r["ari"],
            "ms": round((time.perf_counter() - start) * 1000, 1),
        })
    return {"rows": rows}
