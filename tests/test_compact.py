import numpy as np
import os, shutil
from vectorlite import VectorLiteClient

def test_compact(tmp_path):
    db = tmp_path / "db"
    vl = VectorLiteClient(path=db, dim=4)
    vl.upsert_batch([
        ("a", np.array([1,0,0,0], dtype="float32"), {}),
        ("b", np.array([0,1,0,0], dtype="float32"), {}),
        ("c", np.array([0,0,1,0], dtype="float32"), {}),
    ])
    vl.delete("b")
    from vectorlite.vectorlite.storage_memmap import MemmapStorage
    s = MemmapStorage(db)
    old_next = s.next_idx
    s.compact()
    assert s.next_idx < old_next
    ids = s.list_ids()
    assert "a" in ids and "c" in ids
