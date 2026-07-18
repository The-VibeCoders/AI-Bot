import os
import uuid
import chromadb
from app.core.config import BASE_DIR
from app.utils.compat import _embed

class ChromaMemoryDB:
    def __init__(self):
        print(">> Booting High-Speed Vector Engine (ChromaDB)...")
        self.db_path = os.path.join(BASE_DIR, "chroma_db")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        self.collection = self.client.get_or_create_collection(
            name="doremon_memories",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[OK] ChromaDB Online. Collection count: {self.collection.count()}")

    def remember(self, role: str, content: str, user_id: str, session_id: str, batch: bool = False, precomputed_vector: list | None = None) -> dict | None:
        try:
            vector = precomputed_vector if precomputed_vector is not None else _embed(content)
        except Exception as e:
            print(f"   [ERROR] Embedding failed: {e}")
            return None

        record_id = uuid.uuid4().hex
        
        metadata = {
            "role": role,
            "user_id": user_id,
            "session_id": session_id
        }
        
        self.collection.add(
            ids=[record_id],
            embeddings=[vector],
            documents=[content],
            metadatas=[metadata]
        )
        
        return {"id": record_id, "role": role, "content": content, "vector": vector}

    def batch_save(self, records: list[dict], user_id: str, session_id: str):
        if not records: return
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        for rec in records:
            ids.append(uuid.uuid4().hex)
            embeddings.append(rec["vector"])
            documents.append(rec["content"])
            metadatas.append({"role": rec.get("role", ""), "user_id": user_id, "session_id": session_id})
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def recall(self, user_id: str, current_query: str | None = None, query_vector: list | None = None, limit: int = 3) -> str:
        try:
            if query_vector is not None:
                query_vec = query_vector
            elif current_query:
                query_vec = _embed(current_query)
            else: 
                return ""
        except Exception:
            return ""

        if not query_vec:
            return ""

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=limit,
            where={"user_id": user_id}
        )

        if not results or not results['documents'] or not results['documents'][0]:
            return "No past memory available yet."

        top_matches = []
        for doc, dist in zip(results['documents'][0], results['distances'][0]):
            # distance < 0.70 is a more relaxed threshold for cosine similarity
            if dist < 0.70: 
                top_matches.append(doc)
        
        if not top_matches:
            return ""
            
        return "\n---\n".join(top_matches)

    def clear(self, user_id: str = None):
        if user_id:
            try:
                self.collection.delete(where={"user_id": user_id})
                print(f">> Memory wiped for user {user_id}.")
            except Exception as e:
                print(f"Failed to clear for user {user_id}: {e}")
        else:
            self.client.delete_collection("doremon_memories")
            self.collection = self.client.get_or_create_collection(
                name="doremon_memories",
                metadata={"hnsw:space": "cosine"}
            )
            print(">> All Memory wiped.")

    def get_recent(self, user_id: str, limit: int = 10) -> list[dict]:
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                limit=limit
            )
            memories = []
            if results and results.get("documents"):
                for doc, meta in zip(results["documents"], results["metadatas"]):
                    memories.append({
                        "content": doc,
                        "role": meta.get("role", "unknown")
                    })
            return memories
        except Exception:
            return []

    def recall_by_session(self, user_id: str, session_id: str, limit: int = 10) -> str:
        """Fetches the most recent documents for a specific session without similarity search."""
        try:
            results = self.collection.get(
                where={"$and": [{"user_id": user_id}, {"session_id": session_id}]},
                limit=limit
            )
            if not results or not results.get("documents"):
                return ""
            return "\n---\n".join(results["documents"])
        except Exception:
            return ""

    @property
    def msg_count(self):
        return self.collection.count()