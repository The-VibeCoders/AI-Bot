"""
Doremon Local AI — FastAPI Backend
====================================
Replaces Streamlit with a proper async server.
- Persistent bot instance (never re-initialises)
- True SSE streaming (tokens arrive as they're generated)
- PDF upload endpoint
- Model switch endpoint
"""

import json
import logging
import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bot import LocalDoremonMaster
import uvicorn

# ── Logging setup ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Script directory for relative paths ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Doremon Local AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000", "http://127.0.0.1"],  # Update for production
    allow_methods=["GET", "POST", "DELETE"],
    allow_credentials=True,
)

# ── Bot lives here forever — never re-initialises between requests ───────────
logger.info("Booting Doremon backend…")
try:
    bot = LocalDoremonMaster()
    logger.info("Doremon ready.")
except Exception as e:
    logger.error(f"Failed to initialize Doremon: {e}")
    raise


# ── Static files (serves index.html) ─────────────────────────────────────────
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
def chat(message: str = Query(..., min_length=1)):
    """
    Server-Sent Events endpoint.
    Each token is pushed as:   data: {"token": "..."}\n\n
    End of stream is signalled: data: [DONE]\n\n
    """
    def event_stream():
        try:
            for token in bot.chat_stream(message):
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'token': f' ❌ Error: {exc}'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables Nginx buffering if proxied
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
def switch_model(model: str = Query(...)):
    if model not in bot.get_available_models():
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
    msg = bot.switch_model(model)
    return {"message": msg, "active": bot.current_model}


# ── PDF ingestion ─────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Write to a temp file then pass the path to the bot
    suffix = f"_{uuid.uuid4().hex[:8]}.pdf"
    tmp_path = os.path.join(tempfile.gettempdir(), f"doremon{suffix}")
    try:
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        logger.info(f"Processing PDF upload: {file.filename}")
        result = bot.read_legal_pdf(tmp_path)
        success = not result.startswith("❌")
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
        "total_records": bot.db.msg_count,
        "short_term_turns": len(bot.short_term_memory),   # sliding window, not full DB
    }


@app.delete("/memory/wipe")
def wipe_memory():
    try:
        db_path = os.path.join(SCRIPT_DIR, "doremon_memory.jsonl")
        if os.path.exists(db_path):
            os.remove(db_path)
        # Reinitialize memory safely
        bot.short_term_memory = []
        if hasattr(bot.db, 'clear'):
            bot.db.clear()
        else:
            # Fallback: recreate db instance
            bot.db.__init__()
        logger.info("Memory wiped successfully")
        return {"message": "Memory wiped successfully."}
    except Exception as e:
        logger.error(f"Memory wipe error: {e}")
        raise HTTPException(status_code=500, detail=f"Memory wipe failed: {str(e)}")


# ── Run directly with: python server.py ──────────────────────────────────────
if __name__ == "__main__":

    # Pass the app object directly — avoids uvicorn re-importing this module
    # which would initialise LocalDoremonMaster() a second time.
    uvicorn.run(app, host="127.0.0.1", port=8000)
