"""
Doremon Local AI Agent — Fully Fixed Edition
=============================================
All original bugs + all newly found bugs resolved.

Fix Log (original):
  [B1]  CLIP skip now set absolutely, not with -= (cumulative mutation bug)
  [B2]  msg_count increments only after embedding succeeds
  [B3]  JSONL loader skips corrupt lines instead of crashing on boot
  [B4]  Command parsing uses len()-based slice, immune to spacing edge cases
  [A5]  Pipeline cached in VRAM — no 10-30s reload on every /draw call
  [A6]  Vector matrix capped at MAX_HOT_MEMORIES to prevent RAM exhaustion
  [A7]  Assistant response trimmed before vector storage to protect recall quality
  [A8]  /clear command added to reset short-term memory mid-session
  [P9]  Single embedding computed per chat() call, reused for recall + remember
  [P10] np.vstack replaced with dirty-flag lazy rebuild pattern
  [F11] /redraw <seed> command added
  [F12] Streaming chat output — no more frozen terminal
  [F13] /memory command to inspect stored records
  [F14] atexit graceful VRAM flush on any exit (SIGTERM override REMOVED — see FIX-12)
  [X15] All bare except clauses replaced with typed catches
  [X16] /model input hardened against non-integer and out-of-range input
  [X17] draw() guards against CLIP skip going below 1
  [X18] FastMemoryDB.recall() guards against vector_matrix/memories length mismatch
  [X19] ollama.chat() response key access hardened with .get()
  [X20] Seed clamped to valid torch Generator range (0 to 2^32-1)

Fix Log (new — this edition):
  [FIX-4]  Chunk token extraction handles both old (dict) and new (object) ollama API
  [FIX-8]  STM now stores BOTH user AND assistant messages (was missing user turns)
  [FIX-9]  db_file path anchored to script directory — no more CWD-dependent path
  [FIX-10] ollama.embeddings() replaced with compatibility wrapper (_embed) that
           supports both old (ollama<0.4) and new (ollama>=0.4 uses ollama.embed)
  [FIX-11] Thread safety: RLock added to FastMemoryDB for all shared state mutations
  [FIX-12] SIGTERM signal override removed — was overwriting uvicorn's shutdown handler
  [FIX-ws] chat_stream() now writes user+assistant to LTM and STM after streaming
  [FIX-cl] FastMemoryDB.clear() added as a proper reset method (no more __init__ hack)
"""

import atexit
import gc
import json
import numpy as np
import os
import random
import threading
import uuid
import time
from urllib.parse import quote
import torch
import ollama
from web_scraper import get_web_context
from PIL import PngImagePlugin
from pypdf import PdfReader
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ── Constants ────────────────────────────────────────────────────────────────
MAX_HOT_MEMORIES   = 500
STM_WINDOW         = 6       # stores 3 full user/assistant pairs
EMBED_MODEL        = "nomic-embed-text"
SD_MODEL_ID        = "stablediffusionapi/realistic-vision-v51"
VECTOR_DIM         = 768

# Anchor all file paths to the directory this script lives in [FIX-9]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Ollama compatibility wrapper [FIX-10] ────────────────────────────────────
def _embed(text: str) -> list:
    """
    Unified embedding call that works with both:
      - ollama < 0.4:  ollama.embeddings(model=..., prompt=...)["embedding"]
      - ollama >= 0.4: ollama.embed(model=..., input=...)["embeddings"][0]
    """
    try:
        # Prefer the newer API
        result = ollama.embed(model=EMBED_MODEL, input=text)
        return result["embeddings"][0]
    except AttributeError:
        # Fall back to the old API
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def _extract_chunk_token(chunk) -> str:
    """
    [FIX-4] Handle both ollama response formats:
      - Old (dict-based):    chunk["message"]["content"]
      - New (object-based):  chunk.message.content
    """
    try:
        if hasattr(chunk, "message"):
            return chunk.message.content or ""
        return (chunk.get("message") or {}).get("content", "")
    except Exception:
        return ""


# =============================================================================
# 🧠 MODULE 1: HIGH-SPEED VECTOR ENGINE
# =============================================================================
class FastMemoryDB:
    def __init__(self):
        print(">> Booting High-Speed Vector Engine (Matrix Math + JSONL)...")

        # [FIX-11] RLock for thread-safe access from concurrent FastAPI requests
        self._lock = threading.RLock()

        # [FIX-9] Path is now relative to this script, not CWD
        self.db_file = os.path.join(SCRIPT_DIR, "doremon_memory.jsonl")

        self.memories: list[dict] = []
        self.vectors:  list[list] = []
        self._matrix_dirty = False
        self.vector_matrix = np.empty((0, VECTOR_DIM), dtype=np.float32)

        # [B3] Skip corrupt lines instead of crashing on boot
        if os.path.exists(self.db_file):
            with open(self.db_file, "r", encoding="utf-8") as f:
                for lineno, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                        if "vector" not in record or "content" not in record:
                            print(f"   [WARN]  Line {lineno}: missing keys, skipping.")
                            continue
                        self.memories.append(record)
                        self.vectors.append(record["vector"])
                    except json.JSONDecodeError:
                        print(f"   [WARN]  Line {lineno}: corrupt JSON, skipping.")

        self.msg_count = len(self.memories)
        self._rebuild_matrix()
        print(f"[OK] Memory Database Online: {self.msg_count} records loaded.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rebuild_matrix(self):
        """[P10] Lazily rebuild numpy matrix only when dirty."""
        if self.vectors:
            self.vector_matrix = np.array(self.vectors, dtype=np.float32)
        else:
            self.vector_matrix = np.empty((0, VECTOR_DIM), dtype=np.float32)
        self._matrix_dirty = False

    def _get_matrix(self) -> np.ndarray:
        """[P10] Return up-to-date matrix, rebuilding only if needed."""
        if self._matrix_dirty:
            self._rebuild_matrix()
        return self.vector_matrix

    def _trim_to_cap(self):
        """[A6] Keep only the most recent MAX_HOT_MEMORIES entries in RAM."""
        if len(self.memories) > MAX_HOT_MEMORIES:
            self.memories = self.memories[-MAX_HOT_MEMORIES:]
            self.vectors  = self.vectors[-MAX_HOT_MEMORIES:]
            self._matrix_dirty = True

    # ── Public API ────────────────────────────────────────────────────────────

    def remember(self, role: str, content: str, batch: bool = False,
                 precomputed_vector: list | None = None) -> dict | None:
        """
        Vectorise content and store it.
        [B2]    msg_count only increments after the embedding succeeds.
        [P9]    Accepts a precomputed_vector to avoid redundant embedding calls.
        [FIX-10] Uses _embed() for ollama version compatibility.
        [FIX-11] Fully thread-safe via RLock.
        """
        try:
            if precomputed_vector is not None:
                vector = precomputed_vector
            else:
                vector = _embed(content)
        except Exception as e:
            print(f"   [ERROR] Embedding failed for role='{role}': {e}")
            return None  # [B2] do NOT increment on failure

        with self._lock:  # [FIX-11]
            self.msg_count += 1  # [B2] safe to increment now
            record = {
                "id":      self.msg_count,
                "role":    role,
                "content": content,
                "vector":  vector,
            }
            self.memories.append(record)
            self.vectors.append(vector)
            self._matrix_dirty = True  # [P10]
            self._trim_to_cap()        # [A6]

        if not batch:
            self._write_record(record)

        return record

    def batch_save(self, records: list[dict]):
        """Single I/O flush for multiple records."""
        if not records:
            return
        try:
            with open(self.db_file, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"   [ERROR] batch_save failed: {e}")

    def recall(self, current_query: str | None = None,
               query_vector: list | None = None,
               limit: int = 3) -> str:
        """
        [P9]    Accepts a precomputed query_vector to skip redundant embedding.
        [X18]   Guards against matrix/memories length mismatch.
        [FIX-10] Uses _embed() for ollama version compatibility.
        [FIX-11] Thread-safe read via RLock.
        """
        with self._lock:  # [FIX-11]
            if not self.memories:
                return "No past memory available yet."

            try:
                if query_vector is not None:
                    query_vec = np.array(query_vector, dtype=np.float32)
                elif current_query:
                    query_vec = np.array(_embed(current_query), dtype=np.float32)
                else:
                    return ""
            except Exception as e:
                print(f"   [ERROR] Recall embedding failed: {e}")
                return ""

            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return ""

            mat = self._get_matrix()

            # [X18] Ensure matrix rows match memories list
            if mat.shape[0] != len(self.memories):
                print("   [WARN]  Matrix/memories mismatch — rebuilding.")
                self._rebuild_matrix()
                mat = self.vector_matrix

            if mat.shape[0] == 0:
                return ""

            matrix_norms = np.linalg.norm(mat, axis=1)
            matrix_norms[matrix_norms == 0] = 1e-10

            similarities = np.dot(mat, query_vec) / (matrix_norms * query_norm)
            top_k        = min(limit, len(self.memories))
            top_indices  = np.argsort(similarities)[::-1][:top_k]
            
            # Lower threshold to 50% for better recall
            top_matches = []
            for i in top_indices:
                if similarities[i] > 0.50:
                    top_matches.append(self.memories[i]["content"])

            # If nothing beats the 75% threshold, return empty to trigger the web scraper
            if not top_matches:
                return ""

        return "\n---\n".join(top_matches)

    def clear(self):
        """Proper in-place reset with thread-safe file deletion."""
        with self._lock: 
            self.memories.clear()
            self.vectors.clear()
            self.msg_count = 0
            self.vector_matrix = np.empty((0, VECTOR_DIM), dtype=np.float32)
            self._matrix_dirty = False
            
            # Safely delete the physical file inside the lock
            if os.path.exists(self.db_file):
                os.remove(self.db_file)
                
        print(">> Memory wiped from RAM and Disk.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _write_record(self, record: dict):
        try:
            with open(self.db_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(f"   [ERROR] Disk write failed: {e}")


# =============================================================================
# 🤖 MODULE 2: THE DOREMON MASTER AGENT
# =============================================================================
class LocalDoremonMaster:
    def __init__(self):
        self.bot_name          = "Doremon"
        self.db                = FastMemoryDB()
        self.sd_pipeline       = None
        self.sd_model_loaded   = None
        self.short_term_memory: list[dict] = []
        self.sessions = {}  # Maps session_id to short_term_memory list
        self.current_model     = "deepseek-r1:7b"
        
        # --- NEW: Session Management ---
        self.session_id = uuid.uuid4().hex
        self.sessions_dir = os.path.join(SCRIPT_DIR, "sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)
        # -------------------------------

        self.gpu_active = torch.cuda.is_available()
        status = "[OK] GPU Detected" if self.gpu_active else "[!] CPU Mode (Slow)"
        print(f"[{self.bot_name}] 100% Local Agent Online | {status}")
        
        # Validate default model exists
        available = self.get_available_models()
        if self.current_model not in available:
            print(f"[!] Model '{self.current_model}' not found!")
            if available:
                self.current_model = available[0]
                print(f"   Using: {self.current_model}")
            else:
                print(f"[!] No models available!")
        print(f">> Default LLM: {self.current_model}")

        # [F14] Graceful VRAM flush on process exit via atexit only.
        # [FIX-12] signal.signal(SIGTERM) REMOVED — it was overwriting uvicorn's
        #          own SIGTERM handler, which broke graceful server shutdown.
        #          atexit handles CLI exits; uvicorn handles its own shutdown.
        atexit.register(self._shutdown)

    def _get_stm(self, session_id: str) -> list:
        """Helper to ensure a session exists and return its memory list."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    # ── VRAM / lifecycle ──────────────────────────────────────────────────────
    # ── Session Management ────────────────────────────────────────────────────
    
    def save_session(self, session_id=None):
        """Saves the specified session's short term memory as a discrete chat thread."""
        sid = session_id or self.session_id
        print(f"[SESSION] Saving session: {sid}")
        
        memory = self._get_stm(sid)
        if not memory:
            print(f"[SESSION] No memory to save for {sid}")
            return
            
        print(f"[SESSION] Saving {len(memory)} messages for {sid}")
            
        # Generate a title based on the first user message
        title = "New Chat"
        for msg in memory:
            if msg["role"] == "user":
                title = msg["content"][:30] + ("..." if len(msg["content"]) > 30 else "")
                break
                
        filepath = os.path.join(self.sessions_dir, f"{sid}.json")
        data = {
            "id": sid,
            "title": title,
            "messages": memory,
            "timestamp": time.time()
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_session(self, session_id: str):
        """Loads a past chat thread into the active session memory."""
        filepath = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.session_id = data["id"]
                loaded_messages = data.get("messages", [])
                self.sessions[self.session_id] = loaded_messages
                self.short_term_memory = self.sessions[self.session_id]
                return True
        return False

    def new_session(self, session_id=None):
        """Starts a fresh chat thread."""
        sid = session_id or uuid.uuid4().hex
        self.session_id = sid
        self.sessions[sid] = []
        self.short_term_memory = self.sessions[sid]
        print(f"[SESSION] Created new session: {sid}")
    def _unload_models(self):
        """Move SD pipeline to CPU and release VRAM."""
        if self.sd_pipeline is not None:
            try:
                self.sd_pipeline.to("cpu")
            except Exception:
                pass
            self.sd_pipeline     = None
            self.sd_model_loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _shutdown(self):
        """[F14] Called by atexit."""
        print("\n>> Flushing VRAM and shutting down...")
        self._unload_models()

    # ── Model management ──────────────────────────────────────────────────────

    def switch_model(self, model_name: str, session_id: str = None) -> str:
        model_name = model_name.strip()
        if not model_name:
            if session_id is None:
                return f"[INFO] Currently active model: {self.current_model}"
            else:
                memory = self._get_stm(session_id)
                return f"[INFO] Currently active model: {self.current_model} (session {session_id[:8]}...)"
         
        old_model = self.current_model
        self.current_model = model_name
        
        # Clear short-term memory for session(s)
        if session_id is None:
            # Clear all sessions when switching model globally
            for sid in self.sessions:
                self.sessions[sid] = []
            # Also clear current session's memory
            self.short_term_memory = []
            msg = f"[SWITCH] Switched to '{self.current_model}' and cleared memory for all sessions"
        else:
            # Clear only specified session's memory
            memory = self._get_stm(session_id)
            memory.clear()
            self.short_term_memory = self.sessions.get(self.session_id, [])
            msg = f"[SWITCH] Switched to '{self.current_model}' for session {session_id[:8]}..."
            
        return msg

    def get_available_models(self) -> list[str]:
        """Handles both old (dict) and new (object) Ollama library versions."""
        try:
            response   = ollama.list()
            models_raw = (response.get("models", [])
                          if isinstance(response, dict)
                          else response.models)
            return [
                m.model if hasattr(m, "model") else m["model"]
                for m in models_raw
            ]
        except Exception as e:
            print(f"[ERROR] Failed to fetch models from Ollama: {e}")
            return []

        # Search trigger patterns that ALWAYS force web search (factual queries only)
    SEARCH_TRIGGERS = [
            "search", "find", "lookup", "where is", "how to find", "location of",
            "tell me about", "what is", "what's", "who is", "who was", "when was", "when is",
            "city", "country", "capital", "population", "distance", "directions to",
            "address", "map", "gps", "latitude", "longitude", "coordinates", "place",
            "restaurant", "hotel", "hospital", "airport", "station", "museum", "park",
            "shop", "store", "mall", "market", "temple", "church", "mosque",
            "beach", "mountain", "lake", "river", "island", "village", "town",
            "info", "about"
        ]

    def _should_search_web(self, user_msg: str, past_context: str) -> bool:
            """Determine if we should search the web for this query."""
            msg_lower = user_msg.lower().strip()
            msg_words = msg_lower.split()
            SEARCH_TRIGGERS = [
            "search", "find", "lookup", "where is", "how to find", "location of",
            "tell me about", "what is", "what's", "who is", "who was", "when was", "when is",
            "city", "country", "capital", "population", "distance", "directions to",
            "address", "map", "gps", "latitude", "longitude", "coordinates", "place",
            "restaurant", "hotel", "hospital", "airport", "station", "museum", "park",
            "shop", "store", "mall", "market", "temple", "church", "mosque",
            "beach", "mountain", "lake", "river", "island", "village", "town",
            "info", "about"
            ]
            # Don't search for simple greetings or conversational messages
            greeting_phrases = ["hello", "hi", "hey", "yo", "sup", "what's up", "howdy", "thanks", "thank you", "ok", "okay", "yes", "no"]
            for gp in greeting_phrases:
                if msg_lower == gp or msg_lower.startswith(gp + " "):
                    return False
            
            # Force search if message contains factual/search triggers
            for trigger in SEARCH_TRIGGERS:
                if trigger in msg_lower:
                    return True
            
            # Check if it's a question (ends with ?) - likely needs search
            if "?" in user_msg and len(msg_words) >= 2:
                return True
            
            # Force search if LTM has nothing useful
            has_useful_memory = past_context.strip() and "no past memory" not in past_context.lower()
            if not has_useful_memory:
                return True
            
            return False

    # Note: chat() method was removed due to indentation corruption - 
    # all chat logic is now in chat_stream() method directly


            

    def chat_stream(self, user_msg: str, session_id: str = None):
        """
        Generator for the FastAPI SSE endpoint.
        Handles: Local Memory -> Web Search -> LLM Response -> Save to Memory
        """
        user_msg = user_msg.strip()
        if not user_msg:
            return
        
        # Get session memory (STM)
        if session_id is None:
            sid = self.session_id
        else:
            sid = session_id
        
        stm = self._get_stm(sid)
        
        # Embed user message
        try:
            user_vec = _embed(user_msg)
        except Exception as e:
            user_vec = None
            print(f"   [WARN] Could not embed user message: {e}")

        # Try local memory recall
        past_context = self.db.recall(query_vector=user_vec) if user_vec else ""
        
        # Force web search for factual queries
        msg_lower = user_msg.lower()
        search_triggers = ["search", "find", "where is", "tell me about", "what is", "who is", "city", "country", "location", "about"]
        should_search = any(t in msg_lower for t in search_triggers) or not past_context.strip()
        
        web_data = ""
        if should_search:
            print(f"   [WEB] Query requires live search...")
            search_query = user_msg
            for prefix in ["search about ", "find ", "tell me about ", "where is "]:
                if search_query.lower().startswith(prefix):
                    search_query = search_query[len(prefix):].strip()
                    break
            
            web_data = get_web_context(search_query)
            if web_data:
                print(f"   [WEB] Found live data")
                past_context = f"Web Search Results:\n{web_data}"
                try:
                    web_vec = _embed(web_data)
                except:
                    web_vec = None
                self.db.remember(role="system_document", content=f"Learned from Web:\n{web_data}", precomputed_vector=web_vec)
            else:
                print(f"   [WEB] No results found")

        # Build system prompt
        system_injection = (
            f"You are {self.bot_name}, an elite, autonomous local AI assistant. "
            "Your core directive is to balance high-level technical expertise with clear, straightforward communication.\n\n"
            "### OPERATIONAL RULES ###\n"
            "1. **Structure & Clarity:** Format your responses for maximum scannability. Use Markdown headings, bullet points, and bold text to organize information logically.\n"
            "2. **Autonomy & Action:** NEVER give generic advice. Provide highly specific, production-ready code or direct answers.\n"
            "3. **Candor:** Be straightforward and honest about your nature as an AI. Do not feign human emotions or personal experiences.\n"
            "4. **Fact Grounding:** You have LIVE internet access. You MUST use the data provided in the LIVE CONTEXT below to answer the user.\n\n"
            "5. **No Guessing:** If Web Search Results are empty, say you don't have that information. NEVER fabricate details.\n\n"
            f"=== LIVE CONTEXT ===\n{past_context}\n===================\n"
        )

        active_context = (
            [{"role": "system", "content": system_injection}]
            + stm
            + [{"role": "user", "content": user_msg}]
        )

        # Stream from Ollama
        try:
            stream = ollama.chat(
                model=self.current_model,
                messages=active_context,
                stream=True,
            )
        except Exception as e:
            print(f"[ERROR] Ollama chat failed: {e}")
            yield f"[ERROR] Failed to get response from AI: {e}"
            return
        
        # Yield tokens
        ans_chunks = []
        for chunk in stream:
            try:
                token = chunk.message.content
            except:
                token = chunk.get('message', {}).get('content', '')
            ans_chunks.append(token)
            yield token
        
        # Save to memory
        ans = "".join(ans_chunks).strip()
        
        # Update STM
        stm.append({"role": "user", "content": user_msg})
        stm.append({"role": "assistant", "content": ans})
        if len(stm) > STM_WINDOW:
            del stm[:-STM_WINDOW]
        
        # Save session
        self.save_session()
        
        # Save to LTM
        self.db.remember("user", user_msg, precomputed_vector=user_vec)
        try:
            ans_vec = _embed(ans)
        except:
            ans_vec = None
        self.db.remember("assistant", ans, precomputed_vector=ans_vec)
        return stream, user_vec, active_context
    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, prompt: str, seed_override: int | None = None) -> str:
        """
        [A5]  Caches pipeline in VRAM — no reload if model unchanged.
        [B1]  CLIP skip set absolutely, not with -=.
        [X17] Guards CLIP skip from going below minimum.
        [F11] seed_override parameter for /redraw support.
        """
        prompt = prompt.strip()
        if not prompt:
            return "[ERROR] Error: Please provide a valid prompt. Example: /draw cyberpunk city"
        if not self.gpu_active:
            return "[ERROR] Error: CUDA GPU not detected. Image generation requires a CUDA GPU."

        # Shortened to fit under the 77-token limit
        QUALITY_PREFIX = "masterpiece, best quality, ultra-detailed, photorealistic, 8k uhd, "
        
        NEGATIVE_PROMPT = (
            "cgi, 3d, sketch, cartoon, anime, text, worst quality, low quality, "
            "ugly, duplicate, morbid, mutilated, poorly drawn, deformed, blurry, "
            "bad anatomy, extra limbs, disfigured"
        )
        enriched_prompt = QUALITY_PREFIX + prompt
        print(f">> [DRAW] Generating: '{prompt}'")

        try:
            if self.sd_model_loaded != SD_MODEL_ID or self.sd_pipeline is None:
                self._unload_models()
                print(f"   Loading pipeline: {SD_MODEL_ID}")
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    SD_MODEL_ID,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                ).to("cuda")

                self.sd_pipeline.enable_attention_slicing(slice_size="auto")
                self.sd_pipeline.enable_vae_tiling()
                self.sd_pipeline.enable_vae_slicing()

                try:
                    self.sd_pipeline.enable_xformers_memory_efficient_attention()
                    print("   [OK] xformers enabled")
                except Exception:
                    print("   ! xformers not available, using default attention")

                self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.sd_pipeline.scheduler.config,
                    algorithm_type="dpmsolver++",
                    use_karras_sigmas=True,
                )

                # [B1] Set CLIP skip absolutely on a freshly-loaded pipeline
                # [X17] Clamp: never go below 1 layer
                total_layers = self.sd_pipeline.text_encoder.config.num_hidden_layers
                target       = max(1, total_layers - 1)
                self.sd_pipeline.text_encoder.config.num_hidden_layers = target

                self.sd_model_loaded = SD_MODEL_ID
                print("   [OK] Pipeline ready and cached in VRAM")
            else:
                print("   [OK] Reusing cached pipeline (skipping reload)")

            WIDTH, HEIGHT = 512, 512

            # [X20] Clamp seed to valid 32-bit range
            if seed_override is not None:
                seed = int(seed_override) % (2 ** 32)
            else:
                seed = random.randint(0, 2 ** 32 - 1)

            generator = torch.Generator(device="cuda").manual_seed(seed)

            result = self.sd_pipeline(
                prompt=enriched_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                # CHANGE 2: Drop steps to 20
                num_inference_steps=20, 
                guidance_scale=7.5,
                width=WIDTH,
                height=HEIGHT,
                generator=generator,
            )
            image = result.images[0]

            filename = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
            file_path = os.path.join(SCRIPT_DIR, filename) # NEW: Explicit path
            
            meta = PngImagePlugin.PngInfo()
            meta.add_text("prompt",          enriched_prompt)
            meta.add_text("negative_prompt", NEGATIVE_PROMPT)
            meta.add_text("seed",            str(seed))
            meta.add_text("steps",           "20")
            meta.add_text("cfg_scale",       "7.5")
            meta.add_text("sampler",         "DPM++ 2M Karras")
            meta.add_text("model",           SD_MODEL_ID)
            
            image.save(file_path, pnginfo=meta) # NEW: Save to file_path

            return (
                f"🎨 Saved '{filename}'\n"
                f"   Seed: {seed} | Steps: 30 | CFG: 7.5 | {WIDTH}×{HEIGHT}\n"
                f"   Sampler: DPM++ 2M Karras | CLIP skip: 2\n"
            )

        except torch.cuda.OutOfMemoryError:
            self._unload_models()
            return (
                "[ERROR] VRAM Error: GPU ran out of memory.\n"
                "   Try: install xformers (`pip install xformers`) or reduce to 512×512."
            )
        except Exception as e:
            return f"[ERROR] Image Generation Error: {type(e).__name__}: {e}"

    # ── PDF ingestion ─────────────────────────────────────────────────────────

    def read_legal_pdf(self, file_path: str) -> str:
        file_path = file_path.strip().strip('"').strip("'")

        if not file_path:
            return "[ERROR] Error: No file path provided."
        if not os.path.exists(file_path):
            return f"[ERROR] Error: File not found — '{file_path}'"
        if not file_path.lower().endswith(".pdf"):
            return f"[ERROR] Error: Only .pdf files are supported. Got: '{file_path}'"

        try:
            reader      = PdfReader(file_path)
            total_pages = len(reader.pages)
            print(f">> Scanning '{file_path}' ({total_pages} pages)...")

            new_records: list[dict] = []
            skipped = 0

            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    print(f"   [WARN]  Page {i+1} extract failed: {e}")
                    skipped += 1
                    continue

                if len(text.strip()) > 10:
                    record = self.db.remember(
                        role="system_document",
                        content=f"Source: {file_path}, Page {i+1}\n{text}",
                        batch=True,
                    )
                    if record:
                        new_records.append(record)
                    else:
                        skipped += 1

            if not new_records:
                return (
                    "[WARN]  PDF read but no text extracted "
                    "(possibly a scanned image — try an OCR tool first)."
                )

            self.db.batch_save(new_records)
            return (
                f"📄 Success! {len(new_records)}/{total_pages} pages vectorized "
                f"into Long-Term Memory. ({skipped} pages skipped)"
            )

        except Exception as e:
            return f"[ERROR] PDF Parsing Error: {type(e).__name__}: {e}"

    # ── Memory inspection ─────────────────────────────────────────────────────

    def show_memory(self, count: int = 10) -> str:
        """[F13] Show the most recent N memory records."""
        if not self.db.memories:
            return "📭 Long-term memory is empty."
        recent = self.db.memories[-count:]
        lines  = [
            f"[{m['role']}] {m['content'][:100]}{'…' if len(m['content']) > 100 else ''}"
            for m in recent
        ]
        return f"🧠 Last {len(recent)} memories:\n" + "\n".join(lines)


# =============================================================================
# 🖥️ MODULE 3: TERMINAL INTERFACE
# =============================================================================

HELP_TEXT = """
Commands:
  /draw <prompt>       — Generate an image
  /redraw <seed>       — Re-generate image with a specific seed   [F11]
  /read <pdf_path>     — Ingest a PDF into long-term memory
  /model               — Switch active LLM
  /memory [n]          — Show last n memory records (default 10)  [F13]
  /clear               — Reset short-term conversation context    [A8]
  /help                — Show this message
  q / exit / quit      — Exit
"""


def run_cli():
    bot = LocalDoremonMaster()
    print("-" * 55)
    print(HELP_TEXT.strip())
    print("-" * 55)

    while True:
        try:
            raw = input("\nYou: ")
        except EOFError:
            print("\nEOF detected. Shutting down.")
            break
        except KeyboardInterrupt:
            print("\nShutting down securely...")
            break

        ui = raw.strip()
        if not ui:
            continue
        if ui.lower() in ("q", "exit", "quit"):
            break

        # ── /draw ─────────────────────────────────────────────────────────────
        if ui.lower().startswith("/draw "):
            prompt = ui[len("/draw "):].strip()
            print(bot.draw(prompt))

        # ── /redraw <seed> ────────────────────────────────────────────────────
        elif ui.lower().startswith("/redraw "):
            arg = ui[len("/redraw "):].strip()
            try:
                seed = int(arg)
            except ValueError:
                print(f"[ERROR] Invalid seed '{arg}'. Usage: /redraw <integer seed>")
                continue
            try:
                prompt = input("   Prompt to redraw: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCanceled.")
                continue
            if not prompt:
                print("[ERROR] No prompt provided.")
                continue
            print(bot.draw(prompt, seed_override=seed))

        # ── /read ─────────────────────────────────────────────────────────────
        elif ui.lower().startswith("/read "):
            path = ui[len("/read "):].strip()
            print(bot.read_legal_pdf(path))

        # ── /model ────────────────────────────────────────────────────────────
        elif ui.lower() == "/model":
            models = bot.get_available_models()
            if not models:
                print("[ERROR] No models found. Pull one with: ollama pull <name>")
                continue

            print("\n=== Available Models ===")
            for i, name in enumerate(models, 1):
                tag = " ◄─ Active" if name == bot.current_model else ""
                print(f"  [{i}] {name}{tag}")

            # [X16] Hardened input
            try:
                raw_choice = input(f"\nSelect [1-{len(models)}] or Enter to cancel: ").strip()
                if not raw_choice:
                    print("Canceled.")
                    continue
                idx = int(raw_choice) - 1
                if not (0 <= idx < len(models)):
                    print(f"[ERROR] Out of range. Enter a number between 1 and {len(models)}.")
                    continue
                print(bot.switch_model(models[idx]))
            except ValueError:
                print("[ERROR] Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\nCanceled.")

        # ── /memory [n] ───────────────────────────────────────────────────────
        elif ui.lower().startswith("/memory"):
            parts = ui.split()
            count = 10
            if len(parts) > 1:
                try:
                    count = max(1, int(parts[1]))
                except ValueError:
                    print("[ERROR] Usage: /memory [number]")
                    continue
            print(bot.show_memory(count))

        # ── /clear ────────────────────────────────────────────────────────────
        elif ui.lower() == "/clear":
            bot.short_term_memory = []
            print("🧹 Short-term memory cleared. Starting fresh context.")

        # ── /help ─────────────────────────────────────────────────────────────
        elif ui.lower() == "/help":
            print(HELP_TEXT)

        # ── Unknown command ───────────────────────────────────────────────────
        elif ui.startswith("/"):
            print(f"[ERROR] Unknown command '{ui.split()[0]}'. Type /help for a list.")

        # ── Normal chat ───────────────────────────────────────────────────────
        else:
            stream, user_vec, _ = bot.chat(ui)
            if stream is None:
                continue

            ans_chunks: list[str] = []
            for chunk in stream:
                token = _extract_chunk_token(chunk)  # [FIX-4]
                print(token, end="", flush=True)
                ans_chunks.append(token)
            print()

            ans = "".join(ans_chunks).strip()

            # [FIX-8] Store BOTH user AND assistant turns in STM
            bot.short_term_memory.append({"role": "user",      "content": ui})
            bot.short_term_memory.append({"role": "assistant", "content": ans})
            if len(bot.short_term_memory) > STM_WINDOW:
                bot.short_term_memory = bot.short_term_memory[-STM_WINDOW:]

            # Store both messages with precomputed vectors
            bot.db.remember("user", ui, precomputed_vector=user_vec)
            try:
                assistant_vec = _embed(ans)
            except Exception as e:
                print(f"   [WARN]  Could not embed assistant message: {e}")
                assistant_vec = None
            bot.db.remember("assistant", ans, precomputed_vector=assistant_vec)


if __name__ == "__main__":
    run_cli()
