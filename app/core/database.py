import os
import json
import threading
import numpy as np
from app.core.config import BASE_DIR, MAX_HOT_MEMORIES, VECTOR_DIM
from app.utils.compat import _embed

class FastMemoryDB:
    def __init__(self):
        print(">> Booting High-Speed Vector Engine (Matrix Math + JSONL)...")
        self._lock = threading.RLock()
        self.db_file = os.path.join(BASE_DIR, "doremon_memory.jsonl")

        self.memories: list[dict] = []
        self.vectors:  list[list] = []
        self._matrix_dirty = False
        self.vector_matrix = np.empty((0, VECTOR_DIM), dtype=np.float32)

        if os.path.exists(self.db_file):
            with open(self.db_file, "r", encoding="utf-8") as f:
                for lineno, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw: continue
                    try:
                        record = json.loads(raw)
                        if "vector" not in record or "content" not in record:
                            continue
                        self.memories.append(record)
                        self.vectors.append(record["vector"])
                    except json.JSONDecodeError:
                        pass

        self.msg_count = len(self.memories)
        self._rebuild_matrix()
        print(f"[OK] Memory Database Online: {self.msg_count} records loaded.")

    def _rebuild_matrix(self):
        if self.vectors:
            self.vector_matrix = np.array(self.vectors, dtype=np.float32)
        else:
            self.vector_matrix = np.empty((0, VECTOR_DIM), dtype=np.float32)
        self._matrix_dirty = False

    def _get_matrix(self) -> np.ndarray:
        if self._matrix_dirty:
            self._rebuild_matrix()
        return self.vector_matrix

    def _trim_to_cap(self):
        if len(self.memories) > MAX_HOT_MEMORIES:
            self.memories = self.memories[-MAX_HOT_MEMORIES:]
            self.vectors  = self.vectors[-MAX_HOT_MEMORIES:]
            self._matrix_dirty = True

    def remember(self, role: str, content: str, batch: bool = False, precomputed_vector: list | None = None) -> dict | None:
        try:
            vector = precomputed_vector if precomputed_vector is not None else _embed(content)
        except Exception as e:
            print(f"   [ERROR] Embedding failed: {e}")
            return None

        with self._lock:
            self.msg_count += 1
            record = {"id": self.msg_count, "role": role, "content": content, "vector": vector}
            self.memories.append(record)
            self.vectors.append(vector)
            self._matrix_dirty = True
            self._trim_to_cap()

        if not batch:
            self._write_record(record)
        return record

    def batch_save(self, records: list[dict]):
        if not records: return
        try:
            with open(self.db_file, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"   [ERROR] batch_save failed: {e}")

    def recall(self, current_query: str | None = None, query_vector: list | None = None, limit: int = 3) -> str:
        with self._lock:
            if not self.memories: return "No past memory available yet."
            try:
                if query_vector is not None:
                    query_vec = np.array(query_vector, dtype=np.float32)
                elif current_query:
                    query_vec = np.array(_embed(current_query), dtype=np.float32)
                else: return ""
            except Exception:
                return ""

            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0: return ""

            mat = self._get_matrix()
            if mat.shape[0] != len(self.memories):
                self._rebuild_matrix()
                mat = self.vector_matrix

            if mat.shape[0] == 0: return ""

            matrix_norms = np.linalg.norm(mat, axis=1)
            matrix_norms[matrix_norms == 0] = 1e-10

            similarities = np.dot(mat, query_vec) / (matrix_norms * query_norm)
            top_k = min(limit, len(self.memories))
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            top_matches = [self.memories[i]["content"] for i in top_indices if similarities[i] > 0.50]
            if not top_matches: return ""
        return "\n---\n".join(top_matches)

    def clear(self):
        with self._lock: 
            self.memories.clear()
            self.vectors.clear()
            self.msg_count = 0
            self.vector_matrix = np.empty((0, VECTOR_DIM), dtype=np.float32)
            self._matrix_dirty = False
            if os.path.exists(self.db_file): os.remove(self.db_file)
        print(">> Memory wiped from RAM and Disk.")

    def _write_record(self, record: dict):
        try:
            with open(self.db_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError: pass