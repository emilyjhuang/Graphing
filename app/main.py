"""Network Lab API: FastAPI service exposing graph analytics and graph ML."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import analysis, gcn, link_prediction
from .datasets import DatasetError, build_graph, catalog

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="Network Lab",
    description="Interactive network analysis and graph machine learning.",
    version="1.0.0",
)


class DatasetSpec(BaseModel):
    key: str = "karate"
    params: dict[str, float] = Field(default_factory=dict)
    edges: str | None = Field(default=None, max_length=1_000_000)


class GraphRequest(BaseModel):
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    community: str = "louvain"


class PathRequest(BaseModel):
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    source: str
    target: str
    weighted: bool = False


class DatasetRequest(BaseModel):
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)


class LinkPredictionRequest(BaseModel):
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    test_frac: float = Field(0.1, ge=0.05, le=0.3)
    seed: int = 0


class GCNRequest(BaseModel):
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    per_class: int = Field(1, ge=1, le=50)
    hidden: int = Field(16, ge=2, le=64)
    epochs: int = Field(200, ge=10, le=1000)
    lr: float = Field(0.01, gt=0, le=1)
    dropout: float = Field(0.5, ge=0, lt=1)
    seed: int = 0


def _graph(spec: DatasetSpec):
    try:
        return build_graph(spec.model_dump())
    except DatasetError as e:
        raise HTTPException(status_code=422, detail=str(e))


async def _timed(fn, *args, **kwargs) -> dict[str, Any]:
    """Run CPU-bound work off the event loop and report how long it took."""
    start = time.perf_counter()
    try:
        result = await run_in_threadpool(fn, *args, **kwargs)
    except (ValueError, DatasetError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    return {
        "datasets": catalog(),
        "community_algorithms": [
            {"key": k, "label": label} for k, (label, _) in analysis.COMMUNITY_ALGORITHMS.items()
        ],
    }


def _analyze(req: GraphRequest) -> dict[str, Any]:
    g = _graph(req.dataset)
    cent = analysis.centralities(g)
    comm = analysis.communities(g, req.community)
    nodes = [
        {
            "id": n,
            "truth": d.get("truth"),
            "community": comm["membership"][n],
            **{metric: round(values[n], 6) for metric, values in cent.items()},
        }
        for n, d in g.nodes(data=True)
    ]
    edges = [{"source": u, "target": v, "weight": d.get("weight", 1.0)} for u, v, d in g.edges(data=True)]
    comm.pop("membership")
    return {"summary": analysis.summary(g), "nodes": nodes, "edges": edges, "communities": comm}


@app.post("/api/graph")
async def graph(req: GraphRequest) -> dict[str, Any]:
    return await _timed(_analyze, req)


@app.post("/api/communities/compare")
async def compare_communities(req: DatasetRequest) -> dict[str, Any]:
    g = _graph(req.dataset)
    return await _timed(analysis.compare_communities, g)


@app.post("/api/path")
async def path(req: PathRequest) -> dict[str, Any]:
    g = _graph(req.dataset)
    return await _timed(analysis.shortest_paths, g, req.source, req.target, req.weighted)


@app.post("/api/robustness")
async def robustness(req: DatasetRequest) -> dict[str, Any]:
    g = _graph(req.dataset)
    return await _timed(analysis.robustness, g)


@app.post("/api/link-prediction")
async def link_pred(req: LinkPredictionRequest) -> dict[str, Any]:
    g = _graph(req.dataset)
    return await _timed(link_prediction.run, g, req.test_frac, req.seed)


@app.post("/api/gcn")
async def gcn_train(req: GCNRequest) -> dict[str, Any]:
    g = _graph(req.dataset)
    return await _timed(
        gcn.run_with_benchmark, g,
        per_class=req.per_class, hidden=req.hidden, epochs=req.epochs,
        lr=req.lr, dropout=req.dropout, seed=req.seed,
    )


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
