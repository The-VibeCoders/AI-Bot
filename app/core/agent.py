import atexit
import json
import os
import uuid
import time
import ollama
from app.core.config import STM_WINDOW, SESSIONS_DIR
from app.core.database import FastMemoryDB
from app.utils.compat import _embed
from app.services.scraper_service import get_web_context
from app.services.pdf_service import ingest_pdf
from app.services.image_service import ImageService

class LocalDoremonMaster:
    def __init__(self):
        self.bot_name = "Doremon"
        self.db = FastMemoryDB()
        self.img_service = ImageService()
        self.sessions = {}
        self.session_id = uuid.uuid4().hex
        self.current_model = "deepseek-r1:7b"
        
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self.short_term_memory = self._get_stm(self.session_id)
        
        atexit.register(self._shutdown)

    def _get_stm(self, session_id: str) -> list:
        if session_id not in self.sessions: self.sessions[session_id] = []
        return self.sessions[session_id]

    def save_session(self, session_id=None):
        sid = session_id or self.session_id
        memory = self._get_stm(sid)
        if not memory: return
        title = next((msg["content"][:30] + "..." for msg in memory if msg["role"] == "user"), "New Chat")
        filepath = os.path.join(SESSIONS_DIR, f"{sid}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"id": sid, "title": title, "messages": memory, "timestamp": time.time()}, f)

    def load_session(self, session_id: str):
        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.session_id = data["id"]
                self.sessions[self.session_id] = data.get("messages", [])
                self.short_term_memory = self.sessions[self.session_id]
                return True
        return False

    def new_session(self, session_id=None):
        sid = session_id or uuid.uuid4().hex
        self.session_id = sid
        self.sessions[sid] = []
        self.short_term_memory = self.sessions[sid]

    def switch_model(self, model_name: str, session_id: str = None) -> str:
        self.current_model = model_name
        if session_id is None:
            for sid in self.sessions: self.sessions[sid] = []
            self.short_term_memory = []
        else:
            self._get_stm(session_id).clear()
            self.short_term_memory = self.sessions.get(self.session_id, [])
        return f"[SWITCH] Switched to '{self.current_model}'"

    def get_available_models(self) -> list[str]:
        try:
            resp = ollama.list()
            models_raw = resp.get("models", []) if isinstance(resp, dict) else resp.models
            return [m.model if hasattr(m, "model") else m["model"] for m in models_raw]
        except Exception: return []

    def chat_stream(self, user_msg: str, session_id: str = None):
        user_msg = user_msg.strip()
        if not user_msg: return
        
        sid = session_id or self.session_id
        stm = self._get_stm(sid)
        user_vec = _embed(user_msg) if True else None
        
        past_context = self.db.recall(query_vector=user_vec) if user_vec else ""
        should_search = any(t in user_msg.lower() for t in ["search", "what is", "who is", "city"]) or not past_context.strip()
        
        if should_search:
            web_data = get_web_context(user_msg)
            if web_data:
                past_context = f"Web:\n{web_data}"
                self.db.remember(role="system_document", content=f"Learned from Web:\n{web_data}")
        
        sys_prompt = f"You are {self.bot_name}, an autonomous local AI. \n=== LIVE CONTEXT ===\n{past_context}\n==================="
        active_context = [{"role": "system", "content": sys_prompt}] + stm + [{"role": "user", "content": user_msg}]
        
        try:
            stream = ollama.chat(model=self.current_model, messages=active_context, stream=True)
            ans_chunks = []
            for chunk in stream:
                token = chunk.message.content if hasattr(chunk, "message") else chunk.get("message", {}).get("content", "")
                ans_chunks.append(token)
                yield token
            
            ans = "".join(ans_chunks).strip()
            stm.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": ans}])
            if len(stm) > STM_WINDOW: del stm[:-STM_WINDOW]
            self.save_session()
            self.db.remember("user", user_msg, precomputed_vector=user_vec)
            self.db.remember("assistant", ans)
        except Exception as e:
            yield f"[ERROR] Failed: {e}"

    def read_legal_pdf(self, path: str): return ingest_pdf(path, self.db)
    def draw(self, prompt: str, seed=None): return self.img_service.draw(prompt, seed)
    def _shutdown(self): self.img_service.unload()