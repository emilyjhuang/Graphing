import networkx as nx
import numpy as np
import pytest

from app import analysis, gcn, link_prediction
from app.datasets import DatasetError, build_graph, parse_edge_list


def test_parse_edge_list_with_weights_header_and_labels():
    g = parse_edge_list("source,target,weight\na,b,2\nb,c\nc,a,0.5\n# labels\na,x\nb,x\nc,y\n")
    assert g.number_of_edges() == 3
    assert g["a"]["b"]["weight"] == 2
    assert {d["truth"] for _, d in g.nodes(data=True)} == {"x", "y"}


@pytest.mark.parametrize("text", ["", "a", "a,b,-1", "a,b\n# labels\na,x\n"])
def test_parse_edge_list_rejects_bad_input(text):
    with pytest.raises(DatasetError):
        parse_edge_list(text)


def test_params_are_clamped():
    g = build_graph({"key": "ba", "params": {"n": 10**9, "m": 2}})
    assert g.number_of_nodes() == 2000


def test_lcc_curve_matches_brute_force():
    g = nx.relabel_nodes(nx.gnp_random_graph(60, 0.05, seed=1), str)
    order = list(g.nodes)
    np.random.default_rng(0).shuffle(order)
    curve = analysis.lcc_curve(g, order)
    for i in range(len(order) + 1):
        h = g.copy()
        h.remove_nodes_from(order[:i])
        expected = max((len(c) for c in nx.connected_components(h)), default=0) / 60
        assert curve[i] == pytest.approx(expected)


def test_targeted_attack_beats_random_on_scale_free_graph():
    r = analysis.robustness(build_graph({"key": "ba", "params": {"n": 300, "m": 2}}))
    assert r["R"]["adaptive_degree"] < r["R"]["random"]


def test_spectral_recovers_planted_partition():
    g = build_graph({"key": "sbm", "params": {"blocks": 3, "block_size": 40, "p_in": 0.3, "p_out": 0.01}})
    assert analysis.communities(g, "spectral_k")["nmi"] > 0.95


def test_weighted_shortest_path_prefers_strong_ties():
    g = nx.Graph()
    g.add_edge("a", "b", weight=1)            # weak direct tie, cost 1
    g.add_edge("a", "c", weight=10)           # strong ties, cost 0.1 each
    g.add_edge("c", "b", weight=10)
    assert analysis.shortest_paths(g, "a", "b", weighted=False)["paths"] == [["a", "b"]]
    assert analysis.shortest_paths(g, "a", "b", weighted=True)["paths"] == [["a", "c", "b"]]


def test_link_prediction_split_has_no_leakage(monkeypatch):
    g = build_graph({"key": "lesmis"})
    seen = []
    original = link_prediction.pair_features

    def spy(graph, pairs, nodes):
        seen.append((graph, pairs))
        return original(graph, pairs, nodes)

    monkeypatch.setattr(link_prediction, "pair_features", spy)
    result = link_prediction.run(g, seed=3)
    (g_fit, fit_pairs), (g_train, test_pairs) = seen
    for graph, pairs in seen:
        assert not any(graph.has_edge(u, v) for u, v in pairs)
    assert g_fit.number_of_edges() < g_train.number_of_edges() < g.number_of_edges()
    assert result["methods"][-1]["auc"] > 0.8


def test_gcn_gradients_match_finite_differences():
    g = build_graph({"key": "karate"})
    nodes = list(g.nodes)
    a_hat = gcn.normalized_adjacency(g, nodes)
    ax = a_hat.toarray()
    rng = np.random.default_rng(0)
    model = gcn.GCN(len(nodes), 4, 2, rng)
    labels = rng.integers(0, 2, len(nodes))
    y = np.eye(2)[labels]
    mask = np.zeros(len(nodes), bool)
    mask[:10] = True
    wd = 1e-2

    def loss():
        z, _ = model.forward(a_hat, ax)
        p = gcn._softmax(z)
        return -np.log(p[mask, labels[mask]]).mean() + wd * (model.w1 ** 2).sum()

    z, cache = model.forward(a_hat, ax)
    grads = model.backward(a_hat, ax, cache, gcn._softmax(z), y, mask, wd)
    eps = 1e-6
    for w, grad in zip([model.w1, model.w2], grads):
        for idx in [(0, 0), (3, 1), (w.shape[0] - 1, w.shape[1] - 1)]:
            old = w[idx]
            w[idx] = old + eps; up = loss()
            w[idx] = old - eps; down = loss()
            w[idx] = old
            assert grad[idx] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-8)


def test_gcn_learns_sbm():
    g = build_graph({"key": "sbm"})
    result = gcn.run(g, per_class=3, epochs=200)
    assert result["results"][0]["test_acc"] > 0.85
    assert len(result["frames"][0]["xy"]) == g.number_of_nodes()


def test_gcn_requires_labels():
    with pytest.raises(gcn.GCNError):
        gcn.run(build_graph({"key": "lesmis"}))
