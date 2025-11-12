import typer
import numpy as np
import json
from .client import VectorLiteClient

app = typer.Typer()

@app.command()
def init(path: str = "./vl_db", dim: int = 1536):
    vl = VectorLiteClient(path=path, dim=dim)
    typer.echo(f"Initialized VectorLite DB at {path} with dim={dim}")

@app.command()
def add(path: str, id: str, vector_file: str, metadata: str = "{}"):
    vl = VectorLiteClient(path=path)
    vec = np.load(vector_file).astype("float32")
    meta = json.loads(metadata)
    try:
        vl.add(id, vec, meta)
        typer.echo(f"Added {id}")
    except Exception as e:
        typer.echo(f"Error: {e}")

@app.command()
def search(path: str, query_file: str, k: int = 5, metric: str = "cosine"):
    vl = VectorLiteClient(path=path)
    q = np.load(query_file).astype("float32")
    res = vl.search(q, k=k, metric=metric)
    typer.echo(json.dumps(res, indent=2))

if __name__ == "__main__":
    app()
