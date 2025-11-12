# scripts/migrate_to_memmap.py
"""
Pack existing per-vector .npy files into a memmap-based DB and rebuild metadata.
Run from project root.
"""
import os
import glob
import numpy as np
from vectorlite.vectorlite.storage_memmap import MemmapStorage

SRC_DB = os.path.abspath("vl_db")
VECT_DIR = os.path.join(SRC_DB, "vectors")

if not os.path.isdir(SRC_DB):
    raise SystemExit("vl_db not found in project root")
files = sorted(glob.glob(os.path.join(VECT_DIR, "*.npy")))
print("Found vector files:", len(files))
if not files:
    raise SystemExit("No .npy vector files to migrate. Exiting.")

# Determine dim from first vector
first = np.load(files[0])
dim = first.shape[0]
print("Detected dim:", dim)

# Create a fresh new metadata.db by renaming existing one to keep backup safety.
old_db = os.path.join(SRC_DB, "metadata.db")
if os.path.exists(old_db):
    print("Found existing metadata.db; it will be kept as metadata.db.bak")
    os.replace(old_db, old_db + ".bak")

# create new memmap storage in place (it will create a fresh metadata.db)
ms = MemmapStorage(SRC_DB, dim=dim)

# Insert vectors in order
for f in files:
    vid = os.path.splitext(os.path.basename(f))[0]
    v = np.load(f).astype("float32")
    print("Packing", vid)
    ms.save_vector(vid, v, metadata={})

print("Migration done. capacity info:", ms.get_capacity_info())
print("Verify with: sqlite3 vl_db/metadata.db 'SELECT id, idx, dim FROM vectors;'")
