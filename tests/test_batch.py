# tests/test_batch.py
import numpy as np
import shutil, os
from vectorlite import VectorLiteClient

DB = "tests_batch_db"

def setup_module():
    if os.path.exists(DB):
        shutil.rmtree(DB)

def teardown_module():
    if os.path.exists(DB):
        shutil.rmtree(DB)

def test_upsert_batch_and_search():
    vl = VectorLiteClient(path=DB, dim=4)
    items = [
        ("x", np.array([1,0,0,0], dtype="float32"), {"name":"x"}),
        ("y", np.array([0,1,0,0], dtype="float32"), {"name":"y"}),
        ("z", np.array([0,0,1,0], dtype="float32"), {"name":"z"}),
    ]
    vl.upsert_batch(items)
    assert set(vl.storage.list_ids()) == {"x","y","z"}
    res = vl.search(np.array([1,0,0,0], dtype="float32"), k=1)
    assert res[0]["id"] == "x"
