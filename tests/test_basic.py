import numpy as np
import shutil
import os
from vectorlite import VectorLiteClient

DB = "tests_vldb"

def setup_module():
    if os.path.exists(DB):
        shutil.rmtree(DB)

def teardown_module():
    if os.path.exists(DB):
        shutil.rmtree(DB)

def test_add_search_get_delete():
    vl = VectorLiteClient(path=DB, dim=4)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
    vl.add("a", a, {"name":"vec-a"})
    vl.add("b", b, {"name":"vec-b"})
    # search for a
    res = vl.search(np.array([1,0,0,0], dtype="float32"), k=1, metric="cosine")
    assert res[0]["id"] == "a"
    g = vl.get("b")
    assert g["metadata"]["name"] == "vec-b"
    vl.delete("a")
    assert vl.get("a") is None
