# quick_test.py
from vectorlite import VectorLiteClient
import numpy as np
import os

DB = "./vl_db"
vl = VectorLiteClient(path=DB, dim=4)
print("storage info:", getattr(vl.storage, "get_capacity_info", lambda: None)())
# ensure a and b exist
vl.upsert("a", np.array([1.0,0,0,0], dtype="float32"), {"name":"vec-a"})
vl.upsert("b", np.array([0.0,1.0,0,0], dtype="float32"), {"name":"vec-b"})
vl.upsert("c", np.array([0.5,0.5,0,0], dtype="float32"), {"name":"vec-c"})
print("ids ->", vl.storage.list_ids())
print("get b ->", vl.get("b"))
print("search q ->", vl.search(np.array([1,0,0,0], dtype="float32"), k=3))
