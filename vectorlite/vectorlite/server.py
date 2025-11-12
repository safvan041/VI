# vectorlite/vectorlite/server.py
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os

from .client import VectorLiteClient

class UpsertItem(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[dict] = None

class SearchRequest(BaseModel):
    query: List[float]
    k: int = 10
    metric: str = "cosine"

class UpsertBatchRequest(BaseModel):
    items: List[UpsertItem]

app = FastAPI(title="VectorLite HTTP API", version="0.1")

DB_PATH = os.environ.get("VECTORLITE_DB", "./vl_db")
DIM = os.environ.get("VECTORLITE_DIM")
DIM = int(DIM) if DIM else None

_client: Optional[VectorLiteClient] = None

def get_client() -> VectorLiteClient:
    global _client
    if _client is None:
        _client = VectorLiteClient(path=DB_PATH, dim=DIM)
    return _client

@app.get("/health")
def health():
    c = get_client()
    return {"status": "ok", "db_path": DB_PATH, "capacity_info": c.storage.get_capacity_info()}

@app.post("/upsert")
def upsert(item: UpsertItem):
    c = get_client()
    vec = np.asarray(item.vector, dtype="float32")
    try:
        c.upsert(item.id, vec, item.metadata or {})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": item.id, "status": "upserted"}

@app.post("/upsert_batch")
def upsert_batch(req: UpsertBatchRequest):
    c = get_client()
    items = [(it.id, np.asarray(it.vector, dtype="float32"), it.metadata or {}) for it in req.items]
    try:
        c.upsert_batch(items)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"count": len(items), "status": "ok"}

@app.get("/get/{id}")
def get_vec(id: str):
    c = get_client()
    r = c.get(id)
    if r is None:
        raise HTTPException(status_code=404, detail="id not found")
    r["vector"] = r["vector"].tolist()
    return r

@app.post("/search")
def search(req: SearchRequest):
    c = get_client()
    q = np.asarray(req.query, dtype="float32")
    try:
        out = c.search(q, k=req.k, metric=req.metric)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"results": out}

@app.post("/delete/{id}")
def delete(id: str):
    c = get_client()
    c.delete(id)
    return {"id": id, "status": "deleted"}

@app.post("/compact")
def compact():
    c = get_client()
    try:
        c.storage.compact()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compact failed: {e}")
    return {"status": "compacted", "capacity_info": c.storage.get_capacity_info()}
