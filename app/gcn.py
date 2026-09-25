"""A two-layer Graph Convolutional Network (Kipf & Welling, ICLR 2017)
implemented from scratch in NumPy -- forward pass, hand-derived backprop and
Adam -- for semi-supervised node classification.

    A_hat = D^-1/2 (A + I) D^-1/2
    H     = ReLU(A_hat X W1)          hidden representation
    Z     = A_hat H W2                logits
    loss  = CE(softmax(Z)[labelled]) + weight_decay * ||W1||^2

With featureless input (X = I) the model can only learn from graph structure,
which is exactly the Karate-Club setting of the original paper.

Two baselines are trained on the same labelled nodes:
* harmonic-function label propagation (Zhu, Ghahramani & Lafferty 2003)
* logistic regression on a spectral (Laplacian eigenmap) embedding
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg  # noqa: F401  (registers sp.linalg)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from .datasets import has_truth


class GCNError(ValueError):
    pass


def normalized_adjacency(g: nx.Graph, nodes: list) -> sp.csr_matrix:
    a = nx.to_scipy_sparse_array(g, nodelist=nodes, weight=None, format="csr").astype(float)
    a = sp.csr_matrix(a) + sp.eye(len(nodes), format="csr")
    d_inv_sqrt = 1.0 / np.sqrt(np.asarray(a.sum(axis=1)).ravel())
    d = sp.diags(d_inv_sqrt)
    return (d @ a @ d).tocsr()


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class GCN:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator):
        # Glorot-uniform initialisation, as in the reference implementation.
        def glorot(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, size=(fan_in, fan_out))

        self.w1 = glorot(n_in, n_hidden)
        self.w2 = glorot(n_hidden, n_out)
        self.rng = rng

    def forward(self, a_hat, ax, dropout: float = 0.0, train: bool = False):
        pre = ax @ self.w1
        h = np.maximum(pre, 0)
        mask = None
        if train and dropout > 0:
            mask = (self.rng.random(h.shape) >= dropout) / (1 - dropout)
            h = h * mask
        ah = a_hat @ h
        z = ah @ self.w2
        return z, {"pre": pre, "h": h, "ah": ah, "mask": mask}

    def backward(self, a_hat, ax, cache, probs, y_onehot, labelled, weight_decay):
        n_lab = labelled.sum()
        dz = (probs - y_onehot) * labelled[:, None] / n_lab
        dw2 = cache["ah"].T @ dz
        dh = a_hat.T @ (dz @ self.w2.T)
        if cache["mask"] is not None:
            dh = dh * cache["mask"]
        dpre = dh * (cache["pre"] > 0)
        dw1 = ax.T @ dpre + 2 * weight_decay * self.w1
        return [dw1, dw2]


def _split(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    train = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = list(np.flatnonzero(labels == c))
        rng.shuffle(idx)
        train[idx[:per_class]] = True
    return train


def _harmonic(a: sp.csr_matrix, labels: np.ndarray, train: np.ndarray, n_classes: int) -> np.ndarray:
    """Solve L_uu f_u = W_ul f_l (harmonic label propagation), then apply class
    mass normalisation so a well-connected seed can't swallow every class."""
    deg = np.asarray(a.sum(axis=1)).ravel()
    lap = sp.diags(deg) - a
    u, l = np.flatnonzero(~train), np.flatnonzero(train)
    fl = np.eye(n_classes)[labels[l]]
    luu = lap[u][:, u] + sp.eye(len(u)) * 1e-9  # regularise isolated components
    rhs = a[u][:, l] @ fl
    fu = sp.linalg.spsolve(luu.tocsc(), rhs)
    fu = fu.reshape(len(u), n_classes)
    prior = fl.mean(axis=0)
    fu = fu * prior / np.maximum(fu.sum(axis=0), 1e-12)
    pred = labels.copy()
    pred[u] = fu.argmax(axis=1)
    return pred


def _spectral_lr(g: nx.Graph, nodes: list, labels: np.ndarray, train: np.ndarray, dim: int) -> np.ndarray:
    lap = nx.normalized_laplacian_matrix(g, nodelist=nodes).toarray()
    _, vecs = np.linalg.eigh(lap)
    emb = vecs[:, 1 : min(dim, len(nodes) - 1) + 1]
    clf = LogisticRegression(max_iter=2000).fit(emb[train], labels[train])
    return clf.predict(emb)


def run(
    g: nx.Graph,
    per_class: int = 1,
    hidden: int = 16,
    epochs: int = 200,
    lr: float = 0.01,
    dropout: float = 0.5,
    weight_decay: float = 5e-4,
    seed: int = 0,
    snapshots: int = 40,
) -> dict[str, Any]:
    if not has_truth(g):
        raise GCNError("Node classification needs ground-truth labels (try Karate Club or SBM)")
    nodes = list(g.nodes)
    classes = sorted({g.nodes[n]["truth"] for n in nodes}, key=str)
    labels = np.array([classes.index(g.nodes[n]["truth"]) for n in nodes])
    n, c = len(nodes), len(classes)
    per_class = int(np.clip(per_class, 1, 50))
    epochs = int(np.clip(epochs, 10, 1000))
    hidden = int(np.clip(hidden, 2, 64))

    train = _split(labels, per_class, seed)
    test = ~train
    if not test.any():
        raise GCNError("Every node is labelled; lower labelled-nodes-per-class")

    a_hat = normalized_adjacency(g, nodes)
    ax = a_hat.toarray()  # A_hat @ I: featureless input, precomputed once
    y = np.eye(c)[labels]
    rng = np.random.default_rng(seed)
    model = GCN(n, hidden, c, rng)
    opt = Adam([model.w1, model.w2], lr=lr)

    history = []
    hidden_snaps = []
    snap_every = max(1, epochs // snapshots)
    for epoch in range(1, epochs + 1):
        z, cache = model.forward(a_hat, ax, dropout, train=True)
        p = _softmax(z)
        loss = -np.log(p[train, labels[train]] + 1e-12).mean() + weight_decay * (model.w1 ** 2).sum()
        opt.step(model.backward(a_hat, ax, cache, p, y, train, weight_decay))

        z_eval, cache_eval = model.forward(a_hat, ax)
        pred = z_eval.argmax(axis=1)
        history.append({
            "epoch": epoch,
            "loss": float(loss),
            "train_acc": float((pred[train] == labels[train]).mean()),
            "test_acc": float((pred[test] == labels[test]).mean()),
        })
        if epoch == 1 or epoch % snap_every == 0 or epoch == epochs:
            hidden_snaps.append((epoch, cache_eval["h"].copy(), pred.copy()))

    # Project every snapshot with the *final* PCA basis so the animation shows
    # nodes moving through one fixed space instead of a re-fit per frame.
    pca = PCA(n_components=2, random_state=seed).fit(hidden_snaps[-1][1])
    frames = []
    for epoch, h, pred in hidden_snaps:
        xy = pca.transform(h)
        frames.append({"epoch": epoch, "xy": np.round(xy, 4).tolist(), "pred": pred.tolist()})

    final_pred = frames[-1]["pred"]
    harmonic = _harmonic(nx.to_scipy_sparse_array(g, nodelist=nodes, weight=None, format="csr").astype(float), labels, train, c)
    spectral = _spectral_lr(g, nodes, labels, train, dim=max(2, c))

    def acc(pred):
        pred = np.asarray(pred)
        return float((pred[test] == labels[test]).mean())

    return {
        "nodes": nodes,
        "classes": [str(x) for x in classes],
        "labels": labels.tolist(),
        "train_mask": train.tolist(),
        "history": history,
        "frames": frames,
        "final_pred": final_pred,
        "results": [
            {"key": "gcn", "label": "GCN (NumPy, from scratch)", "test_acc": acc(final_pred)},
            {"key": "harmonic", "label": "Harmonic label propagation + CMN", "test_acc": acc(harmonic)},
            {"key": "spectral_lr", "label": "Spectral embedding + LogReg", "test_acc": acc(spectral)},
            {"key": "majority", "label": "Majority-class guess", "test_acc": float(np.bincount(labels[test]).max() / test.sum())},
        ],
        "n_labelled": int(train.sum()),
    }


def run_with_benchmark(g: nx.Graph, repeats: int = 10, max_nodes: int = 500, **kwargs) -> dict[str, Any]:
    """Train once for the visualisation, then re-run every model on ``repeats``
    different random label splits and report mean ± std test accuracy, since a
    single split with one or two labels per class is very noisy."""
    result = run(g, **kwargs)
    result["benchmark"] = None
    if g.number_of_nodes() <= max_nodes and repeats > 1:
        base_seed = kwargs.get("seed", 0)
        scores: dict[str, list[float]] = {r["key"]: [] for r in result["results"]}
        for i in range(repeats):
            rep = run(g, **{**kwargs, "seed": base_seed + i, "snapshots": 1})
            for r in rep["results"]:
                scores[r["key"]].append(r["test_acc"])
        result["benchmark"] = {
            "repeats": repeats,
            "scores": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores.items()},
        }
    return result
