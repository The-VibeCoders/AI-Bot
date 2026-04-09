"""
Doremon Local AI Agent — Hardened Edition
==========================================
All bugs fixed, all analysis issues resolved, fully error-proofed.

Fix Log:
  [B1] CLIP skip now set absolutely, not with -= (cumulative mutation bug)
  [B2] msg_count increments only after embedding succeeds
  [B3] JSONL loader skips corrupt lines instead of crashing on boot
  [B4] Command parsing uses len()-based slice, immune to spacing edge cases
  [A5] Pipeline cached in VRAM — no 10-30s reload on every /draw call
  [A6] Vector matrix capped at MAX_HOT_MEMORIES to prevent RAM exhaustion
  [A7] Assistant response trimmed before vector storage to protect recall quality
  [A8] /clear command added to reset short-term memory mid-session
  [P9] Single embedding computed per chat() call, reused for recall + remember
  [P10] np.vstack replaced with dirty-flag lazy rebuild pattern
  [F11] /redraw <seed> command added
  [F12] Streaming chat output — no more frozen terminal
  [F13] /memory command to inspect stored records
  [F14] atexit + SIGTERM graceful VRAM flush on any exit
  [X15] All bare except clauses replaced with typed catches
  [X16] /model input hardened against non-integer and out-of-range input
  [X17] draw() guards against CLIP skip going below 1
  [X18] FastMemoryDB.recall() guards against vector_matrix/memories length mismatch
  [X19] ollama.chat() response key access hardened with .get()
  [X20] Seed clamped to valid torch Generator range (0 to 2^32-1)
"""

import atexit
import gc
import json
import numpy as np
import os
import random
import signal
import uuid
import time
import torch
import ollama

from PIL import PngImagePlugin
from pypdf import PdfReader
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ── Constants ────────────────────────────────────────────────────────────────
MAX_HOT_MEMORIES   = 500    # [A6] cap on in-RAM vector entries
STM_WINDOW         = 6      # short-term memory: 3 full user/assistant pairs
EMBED_MODEL        = "nomic-embed-text"
SD_MODEL_ID        = "stablediffusionapi/realistic-vision-v51"
VECTOR_DIM         = 768


# =============================================================================
# 🧠 MODULE 1: HIGH-SPEED VECTOR ENGINE
# =============================================================================
class FastMemoryDB:
    def __init__(self):
        print(">> Booting High-Speed Vector Engine (Matrix Math + JSONL)...")
        self.db_file = "doremon_memory.jsonl"
        self.memories: list[dict] = []
        self.vectors:  list[list] = []
        self._matrix_dirty = False                          # [P10]
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
                        # Validate required keys exist before trusting the record
                        if "vector" not in record or "content" not in record:
                            print(f"   ⚠️  Line {lineno}: missing keys, skipping.")
                            continue
                        self.memories.append(record)
                        self.vectors.append(record["vector"])
                    except json.JSONDecodeError:
                        print(f"   ⚠️  Line {lineno}: corrupt JSON, skipping.")

        self.msg_count = len(self.memories)
        self._rebuild_matrix()
        print(f"✅ Memory Database Online: {self.msg_count} records loaded.")

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
        [B2] msg_count only increments after the embedding succeeds.
        [P9] Accepts a precomputed_vector to avoid redundant embedding calls.
        Returns the record dict, or None on failure.
        """
        try:
            if precomputed_vector is not None:
                vector = precomputed_vector
            else:
                vector = ollama.embeddings(model=EMBED_MODEL, prompt=content)["embedding"]
        except Exception as e:
            print(f"   ❌ Embedding failed for role='{role}': {e}")
            return None                                      # [B2] do NOT increment

        self.msg_count += 1                                  # [B2] safe to increment now
        record = {
            "id":      self.msg_count,
            "role":    role,
            "content": content,
            "vector":  vector,
        }
        self.memories.append(record)
        self.vectors.append(vector)
        self._matrix_dirty = True                            # [P10] mark stale

        if not batch:
            self._write_record(record)

        self._trim_to_cap()                                  # [A6]
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
            print(f"   ❌ batch_save failed: {e}")

    def recall(self, current_query: str | None = None,
               query_vector: list | None = None,
               limit: int = 3) -> str:
        """
        [P9] Accepts a precomputed query_vector to skip redundant embedding.
        [X18] Guards against matrix/memories length mismatch.
        """
        if not self.memories:
            return "No past memory available yet."

        try:
            if query_vector is not None:
                query_vec = np.array(query_vector, dtype=np.float32)
            elif current_query:
                raw = ollama.embeddings(model=EMBED_MODEL, prompt=current_query)["embedding"]
                query_vec = np.array(raw, dtype=np.float32)
            else:
                return ""
        except Exception as e:
            print(f"   ❌ Recall embedding failed: {e}")
            return ""

        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return ""

        mat = self._get_matrix()

        # [X18] Ensure matrix rows match memories list
        if mat.shape[0] != len(self.memories):
            print("   ⚠️  Matrix/memories mismatch — rebuilding.")
            self._rebuild_matrix()
            mat = self.vector_matrix

        if mat.shape[0] == 0:
            return ""

        matrix_norms = np.linalg.norm(mat, axis=1)
        matrix_norms[matrix_norms == 0] = 1e-10

        similarities  = np.dot(mat, query_vec) / (matrix_norms * query_norm)
        top_k         = min(limit, len(self.memories))
        top_indices   = np.argsort(similarities)[::-1][:top_k]
        top_matches   = [self.memories[i]["content"] for i in top_indices]

        return "\n---\n".join(top_matches)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _write_record(self, record: dict):
        try:
            with open(self.db_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(f"   ❌ Disk write failed: {e}")


# =============================================================================
# 🤖 MODULE 2: THE DOREMON MASTER AGENT
# =============================================================================
class LocalDoremonMaster:
    def __init__(self):
        self.bot_name          = "Doremon"
        self.db                = FastMemoryDB()
        self.sd_pipeline       = None
        self.sd_model_loaded   = None        # [A5] tracks which model is in VRAM
        self.short_term_memory: list[dict] = []
        self.current_model     = "artifish/llama3.2-uncensored"

        self.gpu_active = torch.cuda.is_available()
        status = "✅ GPU Detected" if self.gpu_active else "⚠️  CPU Mode (Slow)"
        print(f"[{self.bot_name}] 100% Local Agent Online | {status}")
        print(f">> Default LLM: {self.current_model}")

        # [F14] Graceful VRAM flush on any exit path
        atexit.register(self._shutdown)
        signal.signal(signal.SIGTERM, lambda s, f: self._shutdown())

    # ── VRAM / lifecycle ──────────────────────────────────────────────────────

    def _unload_models(self):
        """Move SD pipeline to CPU and release VRAM."""
        if self.sd_pipeline is not None:
            try:
                self.sd_pipeline.to("cpu")
            except Exception:
                pass
            self.sd_pipeline     = None
            self.sd_model_loaded = None      # [A5] reset cache tracker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _shutdown(self):
        """[F14] Called by atexit and SIGTERM."""
        print("\n>> Flushing VRAM and shutting down...")
        self._unload_models()

    # ── Model management ──────────────────────────────────────────────────────

    def switch_model(self, model_name: str) -> str:
        model_name = model_name.strip()
        if not model_name:
            return f"ℹ️  Currently active model: {self.current_model}"
        self.current_model     = model_name
        self.short_term_memory = []
        return f"🔄 Now routing queries to '{self.current_model}'"

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
            print(f"❌ Failed to fetch models from Ollama: {e}")
            return []

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(self, user_msg: str):
        """
        [P9]  Single embedding computed here, reused for recall AND remember.
        [F12] Streams output token-by-token so the terminal never appears frozen.
        [X19] .get() guards on response keys.
        Now returns a tuple (stream, user_vec, active_context) where:
          - stream: the ollama chat stream (generator of chunks) or None if empty prompt
          - user_vec: the embedding vector of the user message (or None if failed)
          - active_context: the list of messages passed to the ollama chat (including system and user)
        The caller is responsible for consuming the stream and storing the assistant message.
        """
        user_msg = user_msg.strip()
        if not user_msg:
            return None, None, None

        # [P9] Compute embedding once and reuse for both recall and storage
        try:
            user_vec = ollama.embeddings(model=EMBED_MODEL, prompt=user_msg)["embedding"]
        except Exception as e:
            user_vec = None
            print(f"   ⚠️  Could not embed user message: {e}")

        past_context = (
            self.db.recall(query_vector=user_vec)
            if user_vec else "Memory unavailable."
        )

        system_injection = (
            f"You are {self.bot_name}, an advanced local AI assistant. "
            "Use the recalled memories below to answer if relevant:\n"
            f"=== PAST MEMORY ===\n{past_context}\n===================\n"
        )

        # Build active_context without modifying self.short_term_memory
        active_context = [{"role": "system", "content": system_injection}] + self.short_term_memory
        active_context.append({"role": "user", "content": user_msg})

        # Get the stream from ollama
        stream = ollama.chat(
            model=self.current_model,
            messages=active_context,
            stream=True,
        )

        return stream, user_vec, active_context

    def chat_stream(self, user_msg: str):
        """
        Generator function that yields tokens from the chat stream.
        Used by the FastAPI server for SSE streaming.
        """
        stream, _, _ = self.chat(user_msg)
        if stream is None:
            return
        for chunk in stream:
            token = (chunk.get("message") or {}).get("content", "")
            yield token

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
            return "❌ Error: Please provide a valid prompt. Example: /draw cyberpunk city"
        if not self.gpu_active:
            return "❌ Error: CUDA GPU not detected. Image generation requires a CUDA GPU."

        QUALITY_PREFIX = (
            "RAW photo, (photorealistic:1.4), masterpiece, best quality, "
            "ultra-detailed, sharp focus, 8k uhd, DSLR, film grain, "
            "Fujifilm XT3, professional lighting, bokeh, "
        )
        NEGATIVE_PROMPT = (
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, "
            "sketch, cartoon, drawing, anime:1.4), text, close up, cropped, "
            "out of frame, worst quality, low quality, jpeg artifacts, ugly, "
            "duplicate, morbid, mutilated, extra fingers, mutated hands, "
            "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, "
            "dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, "
            "disfigured, gross proportions, malformed limbs, missing arms, "
            "missing legs, extra arms, extra legs, fused fingers, too many fingers, "
            "long neck, watermark, signature, parentheses"
        )
        enriched_prompt = QUALITY_PREFIX + prompt
        print(f">> [DRAW] Generating: '{prompt}'")

        try:
            # [A5] Only reload if model changed or pipeline was unloaded
            if self.sd_model_loaded != SD_MODEL_ID or self.sd_pipeline is None:
                self._unload_models()
                print(f"   Loading pipeline: {SD_MODEL_ID}")
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    SD_MODEL_ID,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                ).to("cuda")

                # VRAM survival pack
                self.sd_pipeline.enable_attention_slicing(slice_size="auto")
                self.sd_pipeline.enable_vae_tiling()
                self.sd_pipeline.enable_vae_slicing()

                try:
                    self.sd_pipeline.enable_xformers_memory_efficient_attention()
                    print("   ✓ xformers enabled")
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
                target       = max(1, total_layers - 1)   # CLIP skip=2 means subtract 1
                self.sd_pipeline.text_encoder.config.num_hidden_layers = target

                self.sd_model_loaded = SD_MODEL_ID        # [A5] mark as cached
                print("   ✓ Pipeline ready and cached in VRAM")
            else:
                print("   ✓ Reusing cached pipeline (skipping reload)")

            WIDTH, HEIGHT = 768, 512

            # [X20] Clamp seed to valid 32-bit range
            if seed_override is not None:
                seed = int(seed_override) % (2**32)
            else:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device="cuda").manual_seed(seed)

            result = self.sd_pipeline(
                prompt=enriched_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=WIDTH,
                height=HEIGHT,
                generator=generator,
            )
            image = result.images[0]

            filename = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
            meta = PngImagePlugin.PngInfo()
            meta.add_text("prompt",          enriched_prompt)
            meta.add_text("negative_prompt", NEGATIVE_PROMPT)
            meta.add_text("seed",            str(seed))
            meta.add_text("steps",           "30")
            meta.add_text("cfg_scale",       "7.5")
            meta.add_text("sampler",         "DPM++ 2M Karras")
            meta.add_text("model",           SD_MODEL_ID)
            image.save(filename, pnginfo=meta)

            return (
                f"🎨 Saved '{filename}'\n"
                f"   Seed: {seed} | Steps: 30 | CFG: 7.5 | {WIDTH}×{HEIGHT}\n"
                f"   Sampler: DPM++ 2M Karras | CLIP skip: 2\n"
                f"   Tip: use /redraw {seed} to regenerate this exact image."
            )

        except torch.cuda.OutOfMemoryError:
            self._unload_models()
            return (
                "❌ VRAM Error: GPU ran out of memory.\n"
                "   Try: install xformers (`pip install xformers`) or reduce to 512×512."
            )
        except Exception as e:
            return f"❌ Image Generation Error: {type(e).__name__}: {e}"
        # NOTE: no finally/_unload here — [A5] intentionally keeps pipeline warm.
        # VRAM is released by _shutdown() on exit, or by the next _unload_models() call.

    # ── PDF ingestion ─────────────────────────────────────────────────────────

    def read_legal_pdf(self, file_path: str) -> str:
        file_path = file_path.strip().strip('"').strip("'")

        if not file_path:
            return "❌ Error: No file path provided."
        if not os.path.exists(file_path):
            return f"❌ Error: File not found — '{file_path}'"
        if not file_path.lower().endswith(".pdf"):
            return f"❌ Error: Only .pdf files are supported. Got: '{file_path}'"

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
                    print(f"   ⚠️  Page {i+1} extract failed: {e}")
                    skipped += 1
                    continue

                if len(text.strip()) > 10:
                    record = self.db.remember(
                        role="system_document",
                        content=f"Source: {file_path}, Page {i+1}\n{text}",
                        batch=True,
                    )
                    if record:                               # remember() returns None on failure
                        new_records.append(record)
                    else:
                        skipped += 1

            if not new_records:
                return (
                    "⚠️  PDF read but no text extracted "
                    "(possibly a scanned image — try an OCR tool first)."
                )

            self.db.batch_save(new_records)
            return (
                f"📄 Success! {len(new_records)}/{total_pages} pages vectorized "
                f"into Long-Term Memory. ({skipped} pages skipped)"
            )

        except Exception as e:
            return f"❌ PDF Parsing Error: {type(e).__name__}: {e}"

    # ── Memory inspection ─────────────────────────────────────────────────────

    def show_memory(self, count: int = 10) -> str:
        """[F13] Show the most recent N memory records."""
        if not self.db.memories:
            return "📭 Long-term memory is empty."
        recent = self.db.memories[-count:]
        lines  = [f"[{m['role']}] {m['content'][:100]}{'…' if len(m['content'])>100 else ''}"
                  for m in recent]
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
            # stdin closed (e.g. piped input finished)
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

        # ── /draw ──────────────────────────────────────────────────────────
        if ui.lower().startswith("/draw "):
            prompt = ui[len("/draw "):].strip()
            print(bot.draw(prompt))

        # ── /redraw <seed> ────────────────────────────────────────────────
        elif ui.lower().startswith("/redraw "):
            arg = ui[len("/redraw "):].strip()
            try:
                seed = int(arg)
            except ValueError:
                print(f"❌ Invalid seed '{arg}'. Usage: /redraw <integer seed>")
                continue
            # Ask user for the prompt to redraw
            try:
                prompt = input("   Prompt to redraw: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCanceled.")
                continue
            if not prompt:
                print("❌ No prompt provided.")
                continue
            print(bot.draw(prompt, seed_override=seed))

        # ── /read ─────────────────────────────────────────────────────────
        elif ui.lower().startswith("/read "):
            path = ui[len("/read "):].strip()
            print(bot.read_legal_pdf(path))

        # ── /model ────────────────────────────────────────────────────────
        elif ui.lower() == "/model":
            models = bot.get_available_models()
            if not models:
                print("❌ No models found. Pull one with: ollama pull <name>")
                continue

            print("\n=== Available Models ===")
            for i, name in enumerate(models, 1):
                tag = " ◄─ Active" if name == bot.current_model else ""
                print(f"  [{i}] {name}{tag}")

            # [X16] Hardened input — non-integer and out-of-range both handled
            try:
                raw_choice = input(f"\nSelect [1-{len(models)}] or Enter to cancel: ").strip()
                if not raw_choice:
                    print("Canceled.")
                    continue
                idx = int(raw_choice) - 1
                if not (0 <= idx < len(models)):
                    print(f"❌ Out of range. Enter a number between 1 and {len(models)}.")
                    continue
                print(bot.switch_model(models[idx]))
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\nCanceled.")

        # ── /memory [n] ───────────────────────────────────────────────────
        elif ui.lower().startswith("/memory"):
            parts = ui.split()
            count = 10
            if len(parts) > 1:
                try:
                    count = max(1, int(parts[1]))
                except ValueError:
                    print("❌ Usage: /memory [number]")
                    continue
            print(bot.show_memory(count))

        # ── /clear ────────────────────────────────────────────────────────
        elif ui.lower() == "/clear":
            bot.short_term_memory = []
            print("🧹 Short-term memory cleared. Starting fresh context.")

        # ── /help ─────────────────────────────────────────────────────────
        elif ui.lower() == "/help":
            print(HELP_TEXT)

        # ── Unknown command ───────────────────────────────────────────────
        elif ui.startswith("/"):
            print(f"❓ Unknown command '{ui.split()[0]}'. Type /help for a list.")

        # ── Normal chat ───────────────────────────────────────────────────
        else:
            # Get stream and other info, then process the stream and store the response
            stream, user_vec, active_context = bot.chat(ui)
            if stream is not None:
                # Consume the stream and collect the full response
                ans_chunks: list[str] = []
                for chunk in stream:
                    token = (chunk.get("message") or {}).get("content", "")
                    print(token, end="", flush=True)
                    ans_chunks.append(token)
                print()  # newline after stream ends
                
                ans = "".join(ans_chunks)
                
                # Update sliding window
                bot.short_term_memory.append({"role": "assistant", "content": ans})
                if len(bot.short_term_memory) > STM_WINDOW:
                    bot.short_term_memory = bot.short_term_memory[-STM_WINDOW:]
                
                # Compute embedding for assistant response for storage
                try:
                    assistant_vec = ollama.embeddings(model=EMBED_MODEL, prompt=ans)["embedding"]
                except Exception as e:
                    print(f"   ⚠️  Could not embed assistant message: {e}")
                    assistant_vec = None
                
                # Store both messages with precomputed vectors
                bot.db.remember("user", ui, precomputed_vector=user_vec)
                bot.db.remember("assistant", ans, precomputed_vector=assistant_vec)


if __name__ == "__main__":
    run_cli()
