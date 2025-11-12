# rebuild_metadata.py
import os, json
import numpy as np
import sqlite3
from datetime import datetime

DB_DIR = os.path.abspath("vl_db")
VECT_DIR = os.path.join(DB_DIR, "vectors")
DB_PATH = os.path.join(DB_DIR, "metadata.db")

if not os.path.exists(DB_DIR):
    raise SystemExit("vl_db not found. Run from project root.")

if not os.path.isdir(VECT_DIR):
    raise SystemExit("No vectors directory found at vl_db/vectors")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute(
    "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, dim INTEGER, metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)
conn.commit()

files = [f for f in os.listdir(VECT_DIR) if f.endswith(".npy")]
print(f"Found {len(files)} vector files.")

for fname in files:
    vid = fname[:-4]  # remove .npy
    vec_path = os.path.join(VECT_DIR, fname)
    try:
        vec = np.load(vec_path)
    except Exception as e:
        print("SKIP (cannot load):", fname, e)
        continue
    dim = int(vec.shape[0])
    # Check if row exists
    cur.execute("SELECT id FROM vectors WHERE id = ?", (vid,))
    if cur.fetchone():
        # update dim if mismatch, but don't overwrite metadata
        cur.execute("UPDATE vectors SET dim = ? WHERE id = ?", (dim, vid))
    else:
        cur.execute("INSERT INTO vectors (id, dim, metadata, created_at) VALUES (?, ?, ?, ?)",
                    (vid, dim, json.dumps({}), datetime.utcnow().isoformat()))
        print("Inserted metadata row for:", vid)
conn.commit()
conn.close()
print("Done. Re-run sqlite3 query to confirm.")
