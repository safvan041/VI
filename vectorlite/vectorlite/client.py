# vectorlite/client.py
import numpy as np
from typing import Optional, List, Dict, Iterable, Tuple
# prefer memmap storage
try:
    from .storage_memmap import MemmapStorage as DefaultStorage
except Exception:
    from .storage import FileStorage as DefaultStorage

from .index_exact import ExactIndex

class VectorLiteClient:
    def __init__(self, path: str, dim: Optional[int] = None, storage_class: Optional[type] = None):
        self.dim = dim
        storage_cls = storage_class or DefaultStorage
        try:
            self.storage = storage_cls(path, dim)
        except TypeError:
            self.storage = storage_cls(path)
            if dim is not None:
                self.dim = dim
        self._index = None

    def _ensure_index(self, rebuild: bool = False):
        if self._index is None or rebuild:
            def loader():
                ids = self.storage.list_ids()
                d = {}
                for _id in ids:
                    v = self.storage.load_vector(_id)
                    if v is None:
                        continue
                    d[_id] = v
                return d
            self._index = ExactIndex(loader)

    def add(self, id: str, vector, metadata: Optional[Dict] = None):
        vector = np.asarray(vector, dtype="float32")
        if self.dim is None:
            self.dim = vector.shape[0]
        if vector.shape[0] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {vector.shape[0]}")
        if self.storage.load_vector(id) is not None:
            raise KeyError(f"id {id} already exists, use upsert to replace")
        self.storage.save_vector(id, vector, metadata)
        self._ensure_index(rebuild=True)

    def upsert(self, id: str, vector, metadata: Optional[Dict] = None):
        vector = np.asarray(vector, dtype="float32")
        if self.dim is None:
            self.dim = vector.shape[0]
        if vector.shape[0] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {vector.shape[0]}")
        self.storage.save_vector(id, vector, metadata)
        self._ensure_index(rebuild=True)

    def upsert_batch(self, items: Iterable[Tuple[str, object, Optional[Dict]]]):
        """
        items: iterable of (id, vector, metadata)
        Performs bulk upsert using storage.save_vectors when available.
        """
        items = list(items)
        if not items:
            return
        # infer dim if needed
        if self.dim is None:
            self.dim = int(np.asarray(items[0][1]).shape[0])
        # prefer storage bulk method if available
        if hasattr(self.storage, "save_vectors"):
            self.storage.save_vectors(items)
        else:
            # fallback to simple loop
            for vid, vec, meta in items:
                self.upsert(vid, vec, meta)
        self._ensure_index(rebuild=True)

    def get(self, id: str):
        v = self.storage.load_vector(id)
        if v is None:
            return None
        return {"id": id, "vector": v, "metadata": self.storage.get_metadata(id)}

    def delete(self, id: str):
        self.storage.delete_vector(id)
        self._ensure_index(rebuild=True)

    def search(self, query, k: int = 10, metric: str = "cosine"):
        self._ensure_index()
        q = np.asarray(query, dtype="float32")
        results = self._index.search(q, k=k, metric=metric)
        out = []
        for _id, score in results:
            out.append({"id": _id, "score": score, "metadata": self.storage.get_metadata(_id)})
        return out
