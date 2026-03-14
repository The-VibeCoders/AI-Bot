"""
Doremon Local AI Agent — High-Performance Edition
==================================================
Optimizations:
  1. [MATH] Vectorized Cosine Similarity (Eliminated Python loops)
  2. [DISK] Switched to JSONL (O(1) append-only file writes)
  3. [MEMORY] Restored Short-Term Sliding Window Context
  4. [DISK] Batched disk writes for PDF parsing
  5. [FEATURE] Dynamic Model Switching Menu
"""

import gc
import json
import numpy as np
import os
import uuid
import time
import torch
import ollama

from pypdf import PdfReader
from diffusers import StableDiffusionPipeline


# ==========================================
# 🧠 MODULE 1: HIGH-SPEED VECTOR ENGINE
# ==========================================
class FastMemoryDB:
    def __init__(self):
        print(">> Booting High-Speed Vector Engine (Matrix Math + JSONL)...")
        # JSONL (JSON Lines) is infinitely faster for appending data
        self.db_file = "doremon_memory.jsonl" 
        self.memories = []
        self.vectors = []
        
        # Load existing memory instantly
        if os.path.exists(self.db_file):
            with open(self.db_file, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    self.memories.append(record)
                    self.vectors.append(record["vector"])
        
        # Pre-compile the matrix for lightning-fast math
        self.vector_matrix = np.array(self.vectors, dtype=np.float32) if self.vectors else np.empty((0, 768))
        self.msg_count = len(self.memories)
        print(f"✅ Memory Database Online: {self.msg_count} records.")

    def remember(self, role, content, batch=False):
        """Converts text to vector and appends instantly."""
        self.msg_count += 1
        vector = ollama.embeddings(model='nomic-embed-text', prompt=content)['embedding']
        
        record = {"id": self.msg_count, "role": role, "content": content, "vector": vector}
        self.memories.append(record)
        
        # Dynamically add to our Numpy Matrix without looping
        vec_np = np.array([vector], dtype=np.float32)
        if self.vector_matrix.shape[0] == 0:
            self.vector_matrix = vec_np
        else:
            self.vector_matrix = np.vstack([self.vector_matrix, vec_np])
        
        # Instant O(1) disk write (Unless batching for PDFs)
        if not batch:
            with open(self.db_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                
        return record

    def batch_save(self, records):
        """Saves multiple records to disk in a single I/O operation."""
        with open(self.db_file, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def recall(self, current_query, limit=3):
        """Vectorized Cosine Similarity math."""
        if self.msg_count == 0:
            return "No past memory available yet."
            
        query_vec = np.array(ollama.embeddings(model='nomic-embed-text', prompt=current_query)['embedding'], dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0: return ""
        
        # Calculate distance against ALL memories simultaneously using Matrix Algebra
        matrix_norms = np.linalg.norm(self.vector_matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1e-10 # Prevent division by zero
        
        similarities = np.dot(self.vector_matrix, query_vec) / (matrix_norms * query_norm)
        
        # Grab the indices of the highest scores
        top_indices = np.argsort(similarities)[::-1][:limit]
        top_matches = [self.memories[i]["content"] for i in top_indices]
        
        return "\n---\n".join(top_matches)


# ==========================================
# 🤖 MODULE 2: THE DOREMON MASTER AGENT
# ==========================================
class LocalDoremonMaster:
    def __init__(self):
        self.bot_name = "Doremon"
        self.db = FastMemoryDB()
        self.sd_pipeline = None
        self.short_term_memory = [] # Restored sliding window context
        
        # Default model configuration
        self.current_model = 'artifish/llama3.2-uncensored'

        self.gpu_active = torch.cuda.is_available()
        status = "✅ GPU Detected" if self.gpu_active else "⚠️ CPU Mode (Slow)"
        print(f"[{self.bot_name}] 100% Local Agent Online | {status}")
        print(f">> Default LLM: {self.current_model}")

    def _unload_models(self):
        """Strict VRAM Management."""
        if self.sd_pipeline is not None:
            self.sd_pipeline.to("cpu")
            self.sd_pipeline = None
        gc.collect()
        if self.gpu_active:
            torch.cuda.empty_cache()

    def switch_model(self, model_name: str) -> str:
        model_name = model_name.strip()
        if not model_name:
            return f"ℹ️ Current active model is: {self.current_model}"
        
        self.current_model = model_name
        # Clear short-term memory to prevent context conflicts between different models
        self.short_term_memory = [] 
        return f"🔄 System updated: Now routing queries to '{self.current_model}'"

    def get_available_models(self) -> list:
        """Fetches all downloaded models directly from the local Ollama instance."""
        try:
            response = ollama.list()
            return [m['model'] for m in response.get('models', [])]
        except Exception as e:
            print(f"❌ Failed to fetch models from Ollama: {str(e)}")
            return []

    def chat(self, user_msg: str) -> str:
        if not user_msg.strip(): return ""

        # 1. Pull deep historical context
        past_context = self.db.recall(user_msg)
        system_injection = (
            "You are Doremon, an advanced local AI assistant. "
            "Use the recalled memories to answer if relevant:\n"
            f"=== PAST MEMORY ===\n{past_context}\n===================\n"
        )

        # 2. Update short term memory
        self.short_term_memory.append({"role": "user", "content": user_msg})
        
        # 3. Combine System Prompt + Sliding Window
        active_context = [{"role": "system", "content": system_injection}] + self.short_term_memory

        try:
            # Use the dynamic model variable instead of hardcoded string
            res = ollama.chat(model=self.current_model, messages=active_context)
            ans = res['message']['content']

            # 4. Save to short term memory
            self.short_term_memory.append({"role": "assistant", "content": ans})
            if len(self.short_term_memory) > 4:
                self.short_term_memory = self.short_term_memory[-4:]

            # 5. Save to Long Term Vector DB
            self.db.remember("user", user_msg)
            self.db.remember("assistant", ans)
            return ans
            
        except Exception as e:
            return f"❌ Chat Error: {str(e)}\nMake sure you have pulled '{self.current_model}' via Ollama."

    # --- FEATURE: UNCENSORED ART GENERATION ---
    # --- FEATURE: VRAM-OPTIMIZED ART GENERATION ---
    """def draw(self, prompt: str) -> str:
        # 1. Input Validation
        if not prompt or not prompt.strip():
            return "❌ Error: Please provide a valid prompt. Example: /draw cyberpunk city"
            
        if not self.gpu_active:
            return "❌ Error: CUDA GPU not found. Image generation requires your RTX 3050 Ti."

        # 2. Strict VRAM Flush (Crucial to clear space for the higher quality generation)
        self._unload_models()
        print(f">> Generating High-Quality Art: '{prompt}'")

        try:
            # 3. Load the model directly into GPU in fp16 (half-precision)
            model_id = "stablediffusionapi/realistic-vision-v51"
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,        # Bypasses local censorship
                requires_safety_checker=False
            ).to("cuda")

            # 4. Optimized Generation Call for 4GB cards
            # INCREASED QUALITY: steps changed from 20 to 40
            # OPTIMIZED RESOLUTION: width=768, height=512 (Cinematic Widescreen)
            # GUIDANCE: Set to 8.0 to strongly adhere to complex prompt details
            image = self.sd_pipeline(
                prompt, 
                num_inference_steps=80,  # DOUBLED steps for convergence and fine detail
                guidance_scale=8.0,      # Pushes the model to stick closer to prompt text
                width=1920,               # Widescreen landscape (Optimized for 4GB)
                height=1080
            ).images[0]

            # 5. Create a unique, collision-proof filename
            filename = f"gen_{int(time.time())}_{uuid.uuid4().hex[:4]}.png"
            image.save(filename)

            return f"🎨 Widescreen Image successfully saved as '{filename}'"
            
        except Exception as e:
            return f"❌ Image Generation Error: {str(e)}"
            
        finally:
            # 6. Immediate VRAM Dump to prevent GPU lag in chat
            self._unload_models()
"""
    # --- FEATURE: LEGAL/OSINT PDF INGESTION ---
    def read_legal_pdf(self, file_path: str) -> str:
        # 1. Clean the file path
        file_path = file_path.strip().strip('"').strip("'")
        
        # 2. File Validation
        if not os.path.exists(file_path):
            return f"❌ Error: Could not find the file '{file_path}' in this folder."
        if not file_path.lower().endswith(".pdf",".docx",".doc",".ppt",".pptx",".svg",".png",".img",".mp3",".mp4"):
            return f"❌ Error: '{file_path}' is not a valid PDF file."

        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            print(f">> Scanning '{file_path}' ({total_pages} pages)...")
            
            new_records = []
            
            # 3. Page-by-Page Extraction
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                
                # Only save pages that actually contain readable text
                if len(text.strip()) > 10:
                    # 4. Convert text to vector math, but delay writing to the hard drive
                    record = self.db.remember(
                        role="system_document", 
                        content=f"Source Document: {file_path}, Page {i + 1}\n{text}", 
                        batch=True
                    )
                    new_records.append(record)

            if not new_records:
                return "⚠️ PDF was read, but it contained no extractable text (it might be a scanned image)."
                
            # 5. High-Speed Disk Write
            self.db.batch_save(new_records)
            
            return f"📄 Success! {len(new_records)}/{total_pages} pages were vectorized and saved to your Long-Term Memory."
            
        except Exception as e:
            return f"❌ PDF Parsing Error: {str(e)}"

# ==========================================
# 🖥️ MODULE 3: TERMINAL INTERFACE
# ==========================================
if __name__ == "__main__":
    bot = LocalDoremonMaster()
    print("-" * 50)
    print("Commands: /draw <prompt> | /read <pdf_path> | /model")
    print("-" * 50)

    while True:
        try:
            ui = input("\nYou: ").strip()
            if not ui: continue
            if ui.lower() in ['q', 'exit', 'quit']: break

            if ui.startswith('/draw '):
                print(bot.draw(ui[6:]))
                
            elif ui.startswith('/read '):
                print(bot.read_legal_pdf(ui[6:]))
                
            elif ui == '/model':
                # 1. Fetch models
                models = bot.get_available_models()
                if not models:
                    print("❌ No models found. Have you pulled any via the terminal?")
                    continue
                
                # 2. Print the interactive menu
                print("\n=== Available Models ===")
                for i, m_name in enumerate(models, 1):
                    # Highlight the currently active model
                    active_tag = " <--(Active)" if m_name == bot.current_model else ""
                    print(f"[{i}] {m_name}{active_tag}")
                
                # 3. Get user selection
                try:
                    choice = input(f"\nSelect a model [1-{len(models)}] or press Enter to cancel: ").strip()
                    if not choice:
                        print("Canceled.")
                        continue
                    
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        print(bot.switch_model(models[idx]))
                    else:
                        print("❌ Invalid selection. Out of range.")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
                    
            else:
                print(f"Doremon: {bot.chat(ui)}")
                
        except KeyboardInterrupt:
            print("\nShutting down securely...")
            break
