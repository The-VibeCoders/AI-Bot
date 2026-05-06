"""
Doremon Local AI — FastAPI Backend (Fixed Edition)
===================================================
All server-side bugs resolved.

Fix Log:
  [FIX-3]  CORS: removed allow_credentials=True (incompatible with allow_origins="*")
  [FIX-5]  __main__ now binds to 0.0.0.0 so it's reachable inside Docker
  [FIX-7]  /memory/wipe uses bot.db.clear() instead of calling __init__() directly
  [FIX-9]  Wipe now uses bot.db.db_file (consistent path) instead of re-computing it
  [FIX-13] Memory stats now accurately reflect state (fixed once chat_stream saves STM)
"""

import json
import logging
import os
import shutil
import tempfile
import uuid
from pydantic import BaseModel
import re
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bot import LocalDoremonMaster
import uvicorn

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Doremon Local AI", version="2.1")

# [FIX-3] allow_credentials=True is forbidden with allow_origins="*" by the
# CORS spec. Browsers block such responses. Removed allow_credentials entirely
# (it defaults to False), which is correct for a local-only app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Bot singleton ─────────────────────────────────────────────────────────────
logger.info("Booting Doremon backend…")
try:
    bot = LocalDoremonMaster()
    logger.info("Doremon ready.")
except Exception as e:
    logger.error(f"Failed to initialize Doremon: {e}")
    raise


# ── Static files ──────────────────────────────────────────────────────────────
index_path = os.path.join(SCRIPT_DIR, "index.html")
if not os.path.exists(index_path):
    logger.warning(f"index.html not found at {index_path}")


@app.get("/", response_class=FileResponse)
def index():
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


# ── SSE streaming chat ────────────────────────────────────────────────────────
@app.get("/chat")
def chat(message: str = Query(..., min_length=1), session_id: str = Query(None)):
    """
    Server-Sent Events endpoint.
    Token push:   data: {"token": "..."}\n\n
    End signal:   data: [DONE]\n\n

    After the stream completes, bot.chat_stream() internally saves both
    user and assistant messages to STM and LTM for the given session.
    If no session_id is provided, the default session is used (for backward compatibility).
    """
    def event_stream():
        try:
            for token in bot.chat_stream(message, session_id=session_id):
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.error(f"Stream error: {exc}")
            yield f"data: {json.dumps({'token': f' [ERROR] Error: {exc}'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Model management ──────────────────────────────────────────────────────────
@app.get("/models")
def list_models():
    return {
        "models": bot.get_available_models(),
        "active": bot.current_model,
    }


@app.post("/models/switch")
def switch_model(model: str = Query(...), session_id: str = Query(None)):
    available = bot.get_available_models()
    if model not in available:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
    msg = bot.switch_model(model, session_id=session_id)
    return {"message": msg, "active": bot.current_model}


# ── PDF ingestion ─────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    suffix  = f"_{uuid.uuid4().hex[:8]}.pdf"
    tmp_path = os.path.join(tempfile.gettempdir(), f"doremon{suffix}")
    try:
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        logger.info(f"Processing PDF upload: {file.filename}")
        result  = bot.read_legal_pdf(tmp_path)
        success = not result.startswith("[ERROR]")
        return {"success": success, "message": result}
    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Memory management ─────────────────────────────────────────────────────────
@app.get("/memory/stats")
def memory_stats():
    return {
        "total_records":    bot.db.msg_count,
        "short_term_turns": len(bot.short_term_memory),
    }


@app.delete("/memory/wipe")
def wipe_memory():
    """
    [FIX-7]  Uses bot.db.clear() — a proper reset method, not __init__().
    [FIX-9]  Uses bot.db.db_file so the path is always consistent with where
             the bot actually writes (anchored to SCRIPT_DIR in FastMemoryDB).
    """
    try:
        db_path = bot.db.db_file  # [FIX-9] authoritative path from the db instance
        if os.path.exists(db_path):
            os.remove(db_path)
        bot.db.clear()            # [FIX-7] proper reset, not __init__()
        bot.short_term_memory = []
        logger.info("Memory wiped successfully")
        return {"message": "Memory wiped successfully."}
    except Exception as e:
        logger.error(f"Memory wipe error: {e}")
        raise HTTPException(status_code=500, detail=f"Memory wipe failed: {str(e)}")

class DrawRequest(BaseModel):
    prompt: str
    seed: int | None = None

@app.post("/draw")
def draw_image(req: DrawRequest):
    try:
        result_text = bot.draw(req.prompt, seed_override=req.seed)
        match = re.search(r"Saved '(gen_.*?\.png)'", result_text)
        filename = match.group(1) if match else None
        return {"success": True, "message": result_text, "filename": filename}
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{filename}")
def get_image(filename: str):
    if not filename.startswith("gen_") or not filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid file request")
    file_path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found on disk")
    return FileResponse(file_path)

@app.delete("/memory/context")
def clear_context():
    bot.new_session()
    return {"message": "Short-term chat context cleared. Starting fresh."}

@app.get("/memory/recent")
def view_memory(limit: int = 10):
    if not bot.db.memories:
        return {"memories": []}
    return {"memories": bot.db.memories[-limit:]}

# ── Session Management ────────────────────────────────────────────────────────
@app.get("/sessions")
def get_sessions():
    """Returns a list of all saved chat threads."""
    sessions = []
    if os.path.exists(bot.sessions_dir):
        for filename in os.listdir(bot.sessions_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(bot.sessions_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        sessions.append({
                            "id": data.get("id"),
                            "title": data.get("title", "Chat"),
                            "timestamp": data.get("timestamp", 0)
                        })
                except Exception:
                    continue
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"sessions": sessions, "active_id": bot.session_id}

@app.post("/sessions/new")
def new_session():
    bot.new_session()
    return {"message": "Started new chat thread.", "id": bot.session_id}

@app.post("/sessions/{session_id}")
def load_session(session_id: str):
    success = bot.load_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session loaded.", "messages": bot.short_term_memory}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # [FIX-5] Bind to 0.0.0.0 so the server is reachable inside Docker.
    #         Previously used 127.0.0.1 which is unreachable from outside
    #         the container even with -p 8000:8000.
    uvicorn.run(app, host="0.0.0.0", port=8000)
