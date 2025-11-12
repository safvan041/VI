# vectorlite/storage_memmap.py
import os
import json
import sqlite3
import numpy as np
from typing import Optional, Dict, List

DEFAULT_CHUNK = 1024  # vectors per chunk when growing

class MemmapStorage:
    """
    Single-file memmap storage for vectors + SQLite metadata.
    Layout:
      - vectors.dat : float32 array shape (capacity, dim)
      - metadata.db : SQLite table `vectors` (id TEXT PRIMARY KEY, idx INTEGER, dim INTEGER, metadata TEXT, created_at TIMESTAMP)
      - A small table `meta` stores capacity and next_idx
    """
    def __init__(self, path: str, dim: Optional[int] = None, initial_capacity: int = DEFAULT_CHUNK):
        self.path = os.path.abspath(path)
        os.makedirs(self.path, exist_ok=True)
        self.db_path = os.path.join(self.path, "metadata.db")
        self.data_path = os.path.join(self.path, "vectors.dat")
        self.dim = dim
        self._init_db()
        # load meta info
        meta = self._get_meta()
        if meta is None:
            # fresh DB: create memmap only if dim provided
            if dim is None:
                # dim unknown yet; wait until first write
                self.capacity = 0
                self.next_idx = 0
                self._mmap = None
            else:
                self.capacity = max(initial_capacity, 1)
                self.next_idx = 0
                self._create_memmap(self.capacity, self.dim)
                self._set_meta(self.capacity, self.next_idx)
        else:
            self.capacity = meta["capacity"]
            self.next_idx = meta["next_idx"]
            # dim might be known from stored vectors table
            if self.dim is None:
                cur = self.conn.cursor()
                cur.execute("SELECT dim FROM vectors LIMIT 1")
                r = cur.fetchone()
                if r:
                    self.dim = int(r[0])
            if self.capacity > 0 and self.dim is not None:
                self._load_memmap(self.capacity, self.dim)
            else:
                self._mmap = None

    # ---------- DB helpers ----------
    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, idx INTEGER, dim INTEGER, metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)"
        )
        self.conn.commit()

    def _get_meta(self) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT v FROM meta WHERE k = 'storage_meta'")
        r = cur.fetchone()
        if not r:
            return None
        return json.loads(r[0])

    def _set_meta(self, capacity: int, next_idx: int):
        cur = self.conn.cursor()
        meta = {"capacity": int(capacity), "next_idx": int(next_idx)}
        cur.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('storage_meta', ?)", (json.dumps(meta),))
        self.conn.commit()

    # ---------- memmap create/load ----------
    def _create_memmap(self, capacity: int, dim: int):
        # create new file with zeros
        shape = (int(capacity), int(dim))
        # use dtype float32
        dtype = np.float32
        # allocate file
        fp = np.memmap(self.data_path, dtype=dtype, mode="w+", shape=shape)
        fp[:] = 0.0
        del fp
        self._load_memmap(capacity, dim)

    def _load_memmap(self, capacity: int, dim: int):
        dtype = np.float32
        self._mmap = np.memmap(self.data_path, dtype=dtype, mode="r+", shape=(int(capacity), int(dim)))
        self.dim = int(dim)

    def _grow(self, min_needed: int = 1):
        # double capacity until it fits
        new_capacity = max(self.capacity * 2 if self.capacity > 0 else DEFAULT_CHUNK, self.capacity + min_needed)
        while new_capacity < self.next_idx + min_needed:
            new_capacity *= 2
        # create temp file and copy data
        tmp_path = self.data_path + ".tmp"
        shape = (int(new_capacity), int(self.dim))
        dtype = np.float32
        tmp = np.memmap(tmp_path, dtype=dtype, mode="w+", shape=shape)
        if self.capacity > 0:
            tmp[: self.capacity, :] = self._mmap[: self.capacity, :]
        tmp.flush()
        del tmp
        # replace file
        os.replace(tmp_path, self.data_path)
        # reload memmap
        self.capacity = new_capacity
        self._load_memmap(self.capacity, self.dim)
        self._set_meta(self.capacity, self.next_idx)

    # ---------- main API ----------
    def save_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None):
        vec = np.asarray(vector, dtype=np.float32)
        if self.dim is None:
            # first vector defines dim
            self.dim = int(vec.shape[0])
            # initialize memmap
            if self.capacity == 0:
                self.capacity = max(DEFAULT_CHUNK, 1)
                self._create_memmap(self.capacity, self.dim)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {vec.shape[0]}")
        # check if id exists
        cur = self.conn.cursor()
        cur.execute("SELECT idx FROM vectors WHERE id = ?", (id,))
        r = cur.fetchone()
        if r:
            idx = int(r[0])
            # overwrite row
            self._mmap[idx, :] = vec
            # update metadata (do not change created_at)
            cur.execute("UPDATE vectors SET dim=?, metadata=? WHERE id=?", (int(self.dim), json.dumps(metadata or {}), id))
            self.conn.commit()
            return
        # append new
        if self.next_idx >= self.capacity:
            self._grow(min_needed=1)
        idx = self.next_idx
        self._mmap[idx, :] = vec
        self._mmap.flush()
        cur.execute("INSERT INTO vectors (id, idx, dim, metadata) VALUES (?, ?, ?, ?)",
                    (id, int(idx), int(self.dim), json.dumps(metadata or {})))
        self.next_idx += 1
        self._set_meta(self.capacity, self.next_idx)
        self.conn.commit()
    
        def save_vectors(self, items):
            """
            items: iterable of (id:str, vector:np.ndarray, metadata:Optional[dict])
            Efficient bulk upsert: minimize grows and DB commits.
            """
            # Convert to list for counting and indexing
            items = list(items)
            if not items:
                return

            # validate dims and set self.dim if first time
            for _, v, _ in items:
                if self.dim is None:
                    self.dim = int(np.asarray(v).shape[0])
                if int(np.asarray(v).shape[0]) != self.dim:
                    raise ValueError(f"Dimension mismatch: expected {self.dim}")

            # ensure capacity
            needed = len([1 for _id, v, _m in items if self._id_not_exists(_id := _id if True else None)])  # placeholder; we'll recompute properly below

            # Determine how many *new* items will be appended (not overwrites)
            cur = self.conn.cursor()
            new_count = 0
            exists_map = {}
            for vid, vec, meta in items:
                cur.execute("SELECT idx FROM vectors WHERE id = ?", (vid,))
                r = cur.fetchone()
                if r:
                    exists_map[vid] = int(r[0])
                else:
                    new_count += 1

            if new_count > 0:
                if self.next_idx + new_count > self.capacity:
                    self._grow(min_needed=new_count)

            # Do writes in a transaction
            cur = self.conn.cursor()
            for vid, vec, meta in items:
                v = np.asarray(vec, dtype=np.float32)
                r = exists_map.get(vid)
                if r is not None:
                    # overwrite
                    idx = r
                    self._mmap[idx, :] = v
                    cur.execute("UPDATE vectors SET dim=?, metadata=? WHERE id=?", (int(self.dim), json.dumps(meta or {}), vid))
                else:
                    idx = self.next_idx
                    self._mmap[idx, :] = v
                    cur.execute("INSERT INTO vectors (id, idx, dim, metadata) VALUES (?, ?, ?, ?)",
                                (vid, int(idx), int(self.dim), json.dumps(meta or {})))
                    self.next_idx += 1

            # flush and commit once
            self._mmap.flush()
            self._set_meta(self.capacity, self.next_idx)
            self.conn.commit()


    def load_vector(self, id: str) -> Optional[np.ndarray]:
        cur = self.conn.cursor()
        cur.execute("SELECT idx FROM vectors WHERE id = ?", (id,))
        r = cur.fetchone()
        if not r:
            return None
        idx = int(r[0])
        return np.array(self._mmap[idx, :], dtype=np.float32)

    def delete_vector(self, id: str):
        # we won't compact the memmap; deletion removes metadata row only
        cur = self.conn.cursor()
        cur.execute("DELETE FROM vectors WHERE id = ?", (id,))
        self.conn.commit()

    def list_ids(self) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM vectors ORDER BY idx")
        return [row[0] for row in cur.fetchall()]

    def get_metadata(self, id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT metadata FROM vectors WHERE id = ?", (id,))
        r = cur.fetchone()
        return json.loads(r[0]) if r else None

    # utility for migration/testing
    def get_capacity_info(self):
        return {"capacity": self.capacity, "next_idx": self.next_idx, "dim": self.dim}
