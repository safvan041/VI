import numpy as np
from typing import List, Tuple, Optional
from .metrics import l2_distance_batch, cosine_similarity_batch

class ExactIndex:
    """
    Simple in-memory exact index. It loads vectors on init from provided loader function.
    For MVP we rebuild index on demand (simple and correct).
    """
    def __init__(self, loader_fn):
        """
        loader_fn() -> dict[id] = np.ndarray
        """
        self.loader_fn = loader_fn
        self._rebuild()

    def _rebuild(self):
        d = self.loader_fn()
        self.ids = list(d.keys())
        self.matrix = np.vstack([d[_id].astype("float32") for _id in self.ids]) if self.ids else np.zeros((0,0), dtype="float32")
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(self.ids)}
        self.dim = self.matrix.shape[1] if self.matrix.size else 0

    def search(self, query: np.ndarray, k: int = 10, metric: str = "cosine") -> List[Tuple[str, float]]:
        if self.matrix.size == 0:
            return []
        q = query.astype("float32")
        if metric == "cosine":
            scores = cosine_similarity_batch(self.matrix, q)  # higher is better
            if k >= scores.shape[0]:
                idx = np.argsort(-scores)
            else:
                idx = np.argpartition(-scores, k)[:k]
                idx = idx[np.argsort(-scores[idx])]
            return [(self.ids[i], float(scores[i])) for i in idx]
        elif metric == "l2":
            dists = l2_distance_batch(self.matrix, q)  # lower is better
            if k >= dists.shape[0]:
                idx = np.argsort(dists)
            else:
                idx = np.argpartition(dists, k)[:k]
                idx = idx[np.argsort(dists[idx])]
            return [(self.ids[i], float(dists[i])) for i in idx]
        else:
            raise ValueError("Unknown metric")
