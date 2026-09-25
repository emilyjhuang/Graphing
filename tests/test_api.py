from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_index_and_datasets():
    assert client.get("/").status_code == 200
    body = client.get("/api/datasets").json()
    assert {d["key"] for d in body["datasets"]} >= {"karate", "sbm", "custom"}


def test_graph_endpoint():
    body = client.post("/api/graph", json={"dataset": {"key": "karate"}, "community": "louvain"}).json()
    assert body["summary"]["nodes"] == 34
    assert body["communities"]["nmi"] is not None
    assert {"betweenness", "pagerank", "community"} <= body["nodes"][0].keys()


def test_path_robustness_lp_gcn_endpoints():
    ds = {"key": "karate"}
    assert client.post("/api/path", json={"dataset": ds, "source": "0", "target": "33"}).json()["hops"] == 2
    assert "adaptive_degree" in client.post("/api/robustness", json={"dataset": ds}).json()["R"]
    assert client.post("/api/link-prediction", json={"dataset": ds}).status_code == 200
    assert client.post("/api/gcn", json={"dataset": ds, "epochs": 50}).status_code == 200


def test_errors_are_422():
    assert client.post("/api/graph", json={"dataset": {"key": "nope"}}).status_code == 422
    assert client.post("/api/graph", json={"dataset": {"key": "custom", "edges": ""}}).status_code == 422
    assert client.post("/api/path", json={"dataset": {"key": "karate"}, "source": "0", "target": "999"}).status_code == 422
    assert client.post("/api/gcn", json={"dataset": {"key": "lesmis"}}).status_code == 422


def test_compare_communities():
    rows = client.post("/api/communities/compare", json={"dataset": {"key": "sbm"}}).json()["rows"]
    assert {r["key"] for r in rows} == {"louvain", "greedy", "label_propagation", "spectral", "spectral_k"}
    assert all(r["nmi"] is not None for r in rows)
