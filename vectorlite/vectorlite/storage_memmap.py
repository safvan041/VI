# vectorlite/storage_memmap.py
import os
import json
import sqlite3
import numpy as np
import base64
import io
import struct
from typing import Optional, Dict, List, Iterable, Tuple
from datetime import datetime, timezone

from .lock import FileLock

DEFAULT_CHUNK = 1024  # vectors per chunk when growing
JOURNAL_FILENAME = "journal.json"
LOCK_FILENAME = "write.lock"


def _serialize_vector_to_b64(vec: np.ndarray) -> str:
    bio = io.BytesIO()
    np.save(bio, np.asarray(vec, dtype=np.float32))
    bio.seek(0)
    return base64.b64encode(bio.read()).decode("ascii")


def _deserialize_vector_from_b64(b64: str) -> np.ndarray:
    data = base64.b64decode(b64.encode("ascii"))
    bio = io.BytesIO(data)
    bio.seek(0)
    return np.load(bio)


class MemmapStorage:
    """
    Single-file memmap storage for vectors + SQLite metadata with simple locking,
    binary journaling, and recovery. Includes a compact() maintenance routine.
    """

    def __init__(self, path: str, dim: Optional[int] = None, initial_capacity: int = DEFAULT_CHUNK):
        self.path = os.path.abspath(path)
        os.makedirs(self.path, exist_ok=True)
        self.db_path = os.path.join(self.path, "metadata.db")
        self.data_path = os.path.join(self.path, "vectors.dat")
        self.journal_path = os.path.join(self.path, JOURNAL_FILENAME)
        self.lock_path = os.path.join(self.path, LOCK_FILENAME)
        self.dim = dim

        self._init_db()
        self._lock = FileLock(self.lock_path, timeout=30.0)

        meta = self._get_meta()
        if meta is None:
            if dim is None:
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

        # recovery if journal exists (binary or json)
        if os.path.exists(self.journal_path) or os.path.exists(self.journal_path + ".bin"):
            print("[recovery] journal found, replaying...")
            self._recover_from_journal()

    # ---------- DB helpers ----------
    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS vectors (
            id TEXT PRIMARY KEY,
            idx INTEGER,
            dim INTEGER,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        cur.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.commit()

    def _get_meta(self) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT v FROM meta WHERE k = 'storage_meta'")
        r = cur.fetchone()
        return json.loads(r[0]) if r else None

    def _set_meta(self, capacity: int, next_idx: int):
        cur = self.conn.cursor()
        meta = {"capacity": int(capacity), "next_idx": int(next_idx)}
        cur.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('storage_meta', ?)", (json.dumps(meta),))
        self.conn.commit()

    # ---------- memmap helpers ----------
    def _create_memmap(self, capacity: int, dim: int):
        shape = (int(capacity), int(dim))
        fp = np.memmap(self.data_path, dtype=np.float32, mode="w+", shape=shape)
        fp[:] = 0.0
        del fp
        self._load_memmap(capacity, dim)

    def _load_memmap(self, capacity: int, dim: int):
        self._mmap = np.memmap(self.data_path, dtype=np.float32, mode="r+", shape=(int(capacity), int(dim)))
        self.dim = int(dim)

    def _grow(self, min_needed: int = 1):
        new_capacity = max(self.capacity * 2 if self.capacity > 0 else DEFAULT_CHUNK, self.capacity + min_needed)
        while new_capacity < self.next_idx + min_needed:
            new_capacity *= 2
        tmp_path = self.data_path + ".tmp"
        tmp = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(int(new_capacity), int(self.dim)))
        if self.capacity > 0:
            tmp[: self.capacity, :] = self._mmap[: self.capacity, :]
        tmp.flush()
        del tmp
        os.replace(tmp_path, self.data_path)
        self.capacity = new_capacity
        self._load_memmap(self.capacity, self.dim)
        self._set_meta(self.capacity, self.next_idx)

    # ---------- journal removal helper ----------
    def _remove_journal(self):
        try:
            if os.path.exists(self.journal_path):
                os.remove(self.journal_path)
            if os.path.exists(self.journal_path + ".bin"):
                os.remove(self.journal_path + ".bin")
        except Exception:
            pass

    # ---------- binary journal ----------
    def _write_binary_journal(self, op: str, items):
        """Write a compact binary journal for fast recovery."""
        tmp_path = self.journal_path + ".bin.tmp"
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<I", len(items)))
            for vid, idx, vec, meta in items:
                vec = np.asarray(vec, dtype=np.float32)
                vec_bytes = vec.tobytes()
                meta_json = json.dumps(meta or {}).encode("utf8")
                id_bytes = vid.encode("utf8")
                f.write(struct.pack("<H", len(id_bytes)))
                f.write(id_bytes)
                f.write(struct.pack("<I", idx))
                f.write(struct.pack("<I", vec.shape[0]))
                f.write(struct.pack("<I", len(meta_json)))
                f.write(meta_json)
                f.write(struct.pack("<I", len(vec_bytes)))
                f.write(vec_bytes)
            footer = json.dumps({"op": op, "when": datetime.now(timezone.utc).isoformat()})
            footer_bytes = footer.encode("utf8")
            f.write(struct.pack("<I", len(footer_bytes)))
            f.write(footer_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.journal_path + ".bin")

    def _recover_from_binary_journal(self, jpath):
        with open(jpath, "rb") as f:
            count = struct.unpack("<I", f.read(4))[0]
            items = []
            for _ in range(count):
                id_len = struct.unpack("<H", f.read(2))[0]
                vid = f.read(id_len).decode("utf8")
                idx, dim = struct.unpack("<II", f.read(8))
                meta_len = struct.unpack("<I", f.read(4))[0]
                meta = json.loads(f.read(meta_len).decode("utf8"))
                vec_len = struct.unpack("<I", f.read(4))[0]
                vec_bytes = f.read(vec_len)
                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                items.append((vid, idx, vec, meta))
        with FileLock(self.lock_path, timeout=30.0):
            cur = self.conn.cursor()
            for vid, idx, vec, meta in items:
                if idx >= self.capacity:
                    self._grow(min_needed=idx - self.capacity + 1)
                self._mmap[idx, :] = vec
                cur.execute(
                    "INSERT OR REPLACE INTO vectors (id, idx, dim, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (vid, idx, int(self.dim), json.dumps(meta or {}), datetime.now(timezone.utc).isoformat()),
                )
            if items:
                self.next_idx = max(i[1] for i in items) + 1
            self._mmap.flush()
            self._set_meta(self.capacity, self.next_idx)
            self.conn.commit()
        os.remove(jpath)
        print(f"[recovery] binary journal replayed {len(items)} ops")

    # ---------- recovery ----------
    def _recover_from_journal(self):
        """Auto-detect binary or JSON journal and replay safely."""
        jbin = self.journal_path + ".bin"
        if os.path.exists(jbin):
            self._recover_from_binary_journal(jbin)
            return
        if not os.path.exists(self.journal_path):
            return
        with open(self.journal_path, "r", encoding="utf8") as f:
            payload = json.load(f)
        items = payload.get("items", [])
        with FileLock(self.lock_path, timeout=30.0):
            max_idx = max(item["idx"] for item in items) if items else -1
            if max_idx >= self.capacity:
                self._grow(min_needed=max_idx - self.capacity + 1)
            cur = self.conn.cursor()
            for it in items:
                vid = it["id"]
                idx = int(it["idx"])
                vec = _deserialize_vector_from_b64(it["vector_b64"])
                meta = it.get("metadata", {})
                self._mmap[idx, :] = vec
                cur.execute(
                    "INSERT OR REPLACE INTO vectors (id, idx, dim, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (vid, idx, int(self.dim), json.dumps(meta or {}), datetime.now(timezone.utc).isoformat()),
                )
            if items:
                self.next_idx = max_idx + 1
            self._mmap.flush()
            self._set_meta(self.capacity, self.next_idx)
            self.conn.commit()
        os.remove(self.journal_path)
        print(f"[recovery] JSON journal replayed {len(items)} ops")

    # ---------- main API ----------
    def save_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None):
        vec = np.asarray(vector, dtype=np.float32)
        with self._lock:
            if self.dim is None:
                self.dim = int(vec.shape[0])
                if self.capacity == 0:
                    self.capacity = max(DEFAULT_CHUNK, 1)
                    self._create_memmap(self.capacity, self.dim)
            if vec.shape[0] != self.dim:
                raise ValueError(f"Dimension mismatch: expected {self.dim}, got {vec.shape[0]}")

            cur = self.conn.cursor()
            cur.execute("SELECT idx FROM vectors WHERE id = ?", (id,))
            r = cur.fetchone()

            if r and r[0] is not None:
                idx = int(r[0])
                self._write_binary_journal("upsert", [(id, idx, vec, metadata or {})])
                self._mmap[idx, :] = vec
                self._mmap.flush()
                cur.execute(
                    "UPDATE vectors SET dim=?, metadata=? WHERE id=?",
                    (int(self.dim), json.dumps(metadata or {}), id),
                )
                self.conn.commit()
                self._set_meta(self.capacity, self.next_idx)
                # remove journal after success
                self._remove_journal()
                return

            if self.next_idx >= self.capacity:
                self._grow(min_needed=1)
            idx = self.next_idx
            self._write_binary_journal("insert", [(id, idx, vec, metadata or {})])
            self._mmap[idx, :] = vec
            self._mmap.flush()
            cur.execute(
                "INSERT OR REPLACE INTO vectors (id, idx, dim, metadata) VALUES (?, ?, ?, ?)",
                (id, idx, int(self.dim), json.dumps(metadata or {})),
            )
            self.next_idx += 1
            self._set_meta(self.capacity, self.next_idx)
            self.conn.commit()
            # remove journal after success
            self._remove_journal()

    def save_vectors(self, items: Iterable[Tuple[str, object, Optional[Dict]]]):
        items = list(items)
        if not items:
            return
        with self._lock:
            for _id, v, _m in items:
                if self.dim is None:
                    self.dim = int(np.asarray(v).shape[0])
                    if self.capacity == 0:
                        self.capacity = max(DEFAULT_CHUNK, 1)
                        self._create_memmap(self.capacity, self.dim)
                if int(np.asarray(v).shape[0]) != self.dim:
                    raise ValueError(f"Dimension mismatch: expected {self.dim}")

            cur = self.conn.cursor()
            exists_map = {}
            new_count = 0
            for vid, vec, meta in items:
                cur.execute("SELECT idx FROM vectors WHERE id = ?", (vid,))
                r = cur.fetchone()
                if r and r[0] is not None:
                    exists_map[vid] = int(r[0])
                else:
                    new_count += 1

            if new_count > 0 and (self.next_idx + new_count > self.capacity):
                self._grow(min_needed=new_count)

            journal_entries = []
            idx_assign = {}
            next_idx_local = self.next_idx
            for vid, vec, meta in items:
                if vid in exists_map:
                    idx = exists_map[vid]
                else:
                    idx = next_idx_local
                    idx_assign[vid] = idx
                    next_idx_local += 1
                journal_entries.append((vid, idx, np.asarray(vec, dtype=np.float32), meta or {}))

            self._write_binary_journal("bulk_upsert", journal_entries)
            for vid, idx, vec, meta in journal_entries:
                self._mmap[idx, :] = vec
                cur.execute(
                    "INSERT OR REPLACE INTO vectors (id, idx, dim, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (vid, idx, int(self.dim), json.dumps(meta or {}), datetime.now(timezone.utc).isoformat()),
                )

            if next_idx_local != self.next_idx:
                self.next_idx = next_idx_local
            self._mmap.flush()
            self._set_meta(self.capacity, self.next_idx)
            self.conn.commit()
            # remove journal after success
            self._remove_journal()

    def load_vector(self, id: str) -> Optional[np.ndarray]:
        cur = self.conn.cursor()
        cur.execute("SELECT idx FROM vectors WHERE id = ?", (id,))
        r = cur.fetchone()
        if not r or r[0] is None:
            return None
        idx = int(r[0])
        return np.array(self._mmap[idx, :], dtype=np.float32)

    def delete_vector(self, id: str):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM vectors WHERE id = ?", (id,))
            self.conn.commit()
            self._set_meta(self.capacity, self.next_idx)

    def list_ids(self) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM vectors ORDER BY idx")
        return [row[0] for row in cur.fetchall()]

    def get_metadata(self, id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT metadata FROM vectors WHERE id = ?", (id,))
        r = cur.fetchone()
        return json.loads(r[0]) if r else None

    # ---------- maintenance: compaction ----------
    def compact(self):
        """
        Rebuilds a new contiguous memmap and metadata table.
        Keeps only active (non-deleted) vectors.
        Safe to run anytime; acquires write lock.
        """
        with self._lock:
            print("[compact] starting compaction...")
            cur = self.conn.cursor()
            cur.execute("SELECT id, idx, dim, metadata FROM vectors ORDER BY idx")
            rows = cur.fetchall()
            # rows is list of (id, idx, dim, metadata)
            active = [(r[0], r[1], r[3]) for r in rows if r[1] is not None]
            if not active:
                print("[compact] nothing to compact.")
                return

            new_count = len(active)
            new_capacity = max(DEFAULT_CHUNK, new_count)
            tmp_data_path = self.data_path + ".compacting"
            # create temp memmap
            tmp = np.memmap(tmp_data_path, dtype=np.float32, mode="w+", shape=(int(new_capacity), int(self.dim)))
            new_idx = 0
            id_to_meta = {}
            # copy vectors from current memmap
            for vid, old_idx, meta_str in active:
                tmp[new_idx, :] = np.array(self._mmap[int(old_idx), :], dtype=np.float32)
                id_to_meta[vid] = meta_str
                new_idx += 1
            tmp.flush()
            del tmp

            # backup current DB and create new DB
            backup_db = self.db_path + ".bak_before_compact"
            os.replace(self.db_path, backup_db)
            # open new DB file and create schema
            self._init_db()  # this reinitializes self.conn to new metadata.db
            cur = self.conn.cursor()

            # insert new rows with contiguous idx
            idx_counter = 0
            for vid, _, meta_str in active:
                meta_json = meta_str or "{}"
                # ensure valid JSON string
                try:
                    _ = json.loads(meta_json)
                except Exception:
                    meta_json = "{}"
                cur.execute(
                    "INSERT INTO vectors (id, idx, dim, metadata) VALUES (?, ?, ?, ?)",
                    (vid, int(idx_counter), int(self.dim), meta_json),
                )
                idx_counter += 1

            self.conn.commit()
            # replace data file atomically
            os.replace(tmp_data_path, self.data_path)
            # reload memmap and update metadata
            self.capacity = new_capacity
            self.next_idx = idx_counter
            self._load_memmap(self.capacity, self.dim)
            self._set_meta(self.capacity, self.next_idx)
            # remove any journal if exists
            try:
                if os.path.exists(self.journal_path):
                    os.remove(self.journal_path)
                if os.path.exists(self.journal_path + ".bin"):
                    os.remove(self.journal_path + ".bin")
            except Exception:
                pass
            print(f"[compact] done. capacity={self.capacity}, next_idx={self.next_idx}")

    def get_capacity_info(self):
        return {"capacity": self.capacity, "next_idx": self.next_idx, "dim": self.dim}
