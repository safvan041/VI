import os
import json
import sqlite3
import numpy as np
from typing import Optional, Dict

class FileStorage:
    """
    Simple per-vector .npy storage + SQLite metadata.
    Replace later with memmap contiguous file for scale.
    """
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(self.path, exist_ok=True)
        self.vdir = os.path.join(self.path, "vectors")
        os.makedirs(self.vdir, exist_ok=True)
        self.db_path = os.path.join(self.path, "metadata.db")
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, dim INTEGER, metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        self.conn.commit()

    def save_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict]=None):
        assert isinstance(vector, np.ndarray)
        vec_path = os.path.join(self.vdir, f"{id}.npy")
        np.save(vec_path, vector.astype("float32"))
        meta_json = json.dumps(metadata or {})
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO vectors (id, dim, metadata) VALUES (?, ?, ?)", (id, int(vector.shape[0]), meta_json))
        self.conn.commit()

    def load_vector(self, id: str) -> Optional[np.ndarray]:
        vec_path = os.path.join(self.vdir, f"{id}.npy")
        if not os.path.exists(vec_path):
            return None
        return np.load(vec_path)

    def delete_vector(self, id: str):
        vec_path = os.path.join(self.vdir, f"{id}.npy")
        if os.path.exists(vec_path):
            os.remove(vec_path)
        cur = self.conn.cursor()
        cur.execute("DELETE FROM vectors WHERE id = ?", (id,))
        self.conn.commit()

    def list_ids(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM vectors")
        return [row[0] for row in cur.fetchall()]

    def get_metadata(self, id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT metadata FROM vectors WHERE id = ?", (id,))
        r = cur.fetchone()
        return json.loads(r[0]) if r else None
