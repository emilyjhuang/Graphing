"""Supervised link prediction with leakage-safe splits.

Pipeline
--------
1. Hide ``test_frac`` of the edges from G  ->  G_train          (test positives)
2. Hide another ``test_frac`` from G_train ->  G_fit            (fit positives)
3. Features for fit pairs are computed on G_fit; features for test pairs on
   G_train. The model therefore never sees a graph that contains the edge it
   is asked to score -- the most common bug in link-prediction demos.
4. Negatives are uniformly sampled non-edges of the *full* graph G.

Each classic heuristic is also scored on its own so the learned model has a
fair baseline to beat.
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

FEATURES = [
    ("common_neighbors", "Common neighbours"),
    ("jaccard", "Jaccard"),
    ("adamic_adar", "Adamic–Adar"),
    ("resource_allocation", "Resource allocation"),
    ("preferential_attachment", "Preferential attachment"),
    ("katz", "Katz index"),
    ("low_rank", "Low-rank (spectral) reconstruction"),
]


class LinkPredictionError(ValueError):
    pass


def _hide_edges(g: nx.Graph, count: int, rng: random.Random) -> tuple[nx.Graph, list[tuple]]:
    """Remove ``count`` random edges without isolating any node."""
    h = g.copy()
    edges = list(h.edges)
    rng.shuffle(edges)
    hidden = []
    for u, v in edges:
        if len(hidden) >= count:
            break
        if h.degree(u) > 1 and h.degree(v) > 1:
            h.remove_edge(u, v)
            hidden.append((u, v))
    return h, hidden


def _sample_non_edges(g: nx.Graph, count: int, rng: random.Random, exclude: set) -> list[tuple]:
    nodes = list(g.nodes)
    out: set[tuple] = set()
    max_possible = len(nodes) * (len(nodes) - 1) // 2 - g.number_of_edges()
    count = min(count, max_possible - len(exclude))
    attempts = 0
    while len(out) < count and attempts < count * 50:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        key = (u, v) if str(u) < str(v) else (v, u)
        if key in out or key in exclude or g.has_edge(u, v):
            continue
        out.add(key)
    return list(out)


def pair_features(g: nx.Graph, pairs: list[tuple], nodes: list) -> np.ndarray:
    """Vectorised topological features for node pairs, computed on ``g``.

    Uses dense matrix algebra (graphs are capped at 2k nodes), so each
    feature is a single matrix product rather than a Python loop per pair.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    a = nx.to_numpy_array(g, nodelist=nodes, weight=None, dtype=np.float64)
    deg = a.sum(axis=1)
    inv_log = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    inv_deg = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)

    ui = np.array([idx[u] for u, _ in pairs])
    vi = np.array([idx[v] for _, v in pairs])
    au, av = a[ui], a[vi]  # neighbour indicator rows, (pairs x n)
    cn = (au * av).sum(axis=1)
    aa = (au * av * inv_log).sum(axis=1)
    ra = (au * av * inv_deg).sum(axis=1)
    union = deg[ui] + deg[vi] - cn
    jac = np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
    pa = deg[ui] * deg[vi]

    # One eigendecomposition A = V diag(lambda) V^T serves both spectral features.
    vals, vecs = np.linalg.eigh(a)
    vu, vv = vecs[ui], vecs[vi]

    # Katz: sum_l beta^l A^l = V diag(1/(1 - beta*lambda) - 1) V^T, beta < 1/lambda_max.
    beta = 0.5 / max(vals[-1], 1e-9)
    katz_s = (vu * (1.0 / (1.0 - beta * vals) - 1.0) * vv).sum(axis=1)

    # Rank-k reconstruction of A from its leading eigenpairs (by |lambda|).
    k = max(1, min(16, len(nodes) - 2))
    top = np.argsort(-np.abs(vals))[:k]
    recon = (vu[:, top] * vals[top] * vv[:, top]).sum(axis=1)

    return np.column_stack([cn, jac, aa, ra, pa, katz_s, recon])


def _roc_points(y: np.ndarray, s: np.ndarray, max_points: int = 60) -> list[list[float]]:
    fpr, tpr, _ = roc_curve(y, s)
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return [[float(a), float(b)] for a, b in zip(fpr, tpr)]


def run(g: nx.Graph, test_frac: float = 0.1, seed: int = 0, top_k: int = 15) -> dict[str, Any]:
    if g.number_of_edges() < 20:
        raise LinkPredictionError("Need at least 20 edges for a meaningful split")
    test_frac = float(np.clip(test_frac, 0.05, 0.3))
    rng = random.Random(seed)
    nodes = list(g.nodes)
    count = max(5, int(round(test_frac * g.number_of_edges())))

    g_train, test_pos = _hide_edges(g, count, rng)
    g_fit, fit_pos = _hide_edges(g_train, count, rng)
    if len(test_pos) < 5 or len(fit_pos) < 5:
        raise LinkPredictionError("Graph is too sparse to hold out edges without isolating nodes")

    def canon(pairs):
        return {(u, v) if str(u) < str(v) else (v, u) for u, v in pairs}

    fit_neg = _sample_non_edges(g, len(fit_pos), rng, exclude=set())
    test_neg = _sample_non_edges(g, len(test_pos), rng, exclude=canon(fit_neg))

    x_fit = pair_features(g_fit, fit_pos + fit_neg, nodes)
    y_fit = np.r_[np.ones(len(fit_pos)), np.zeros(len(fit_neg))]
    test_pairs = test_pos + test_neg
    x_test = pair_features(g_train, test_pairs, nodes)
    y_test = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))]

    # Tree ensembles need data; on tiny graphs a regularised linear model on
    # log-scaled features generalises better.
    if len(y_fit) >= 300:
        model_label = "Gradient-boosted trees (all features)"
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
    else:
        model_label = "Logistic regression (all features)"
        model = make_pipeline(
            FunctionTransformer(lambda x: np.sign(x) * np.log1p(np.abs(x))),
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000),
        )
    model.fit(x_fit, y_fit)
    proba = model.predict_proba(x_test)[:, 1]

    methods = []
    for j, (key, label) in enumerate(FEATURES):
        s = x_test[:, j]
        methods.append({
            "key": key,
            "label": label,
            "auc": float(roc_auc_score(y_test, s)),
            "ap": float(average_precision_score(y_test, s)),
            "roc": _roc_points(y_test, s),
        })
    methods.append({
        "key": "model",
        "label": model_label,
        "auc": float(roc_auc_score(y_test, proba)),
        "ap": float(average_precision_score(y_test, proba)),
        "roc": _roc_points(y_test, proba),
    })

    imp = permutation_importance(model, x_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed)
    importances = sorted(
        ({"feature": label, "importance": float(m)} for (_, label), m in zip(FEATURES, imp.importances_mean)),
        key=lambda d: -d["importance"],
    )

    order = np.argsort(-proba)[:top_k]
    top = [
        {"source": test_pairs[i][0], "target": test_pairs[i][1], "score": float(proba[i]), "hidden_edge": bool(y_test[i])}
        for i in order
    ]
    return {
        "test_edges": len(test_pos),
        "fit_edges": len(fit_pos),
        "methods": methods,
        "importances": importances,
        "top_predictions": top,
        "precision_at_k": float(np.mean([t["hidden_edge"] for t in top])) if top else 0.0,
    }
