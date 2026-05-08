import os
import json
import uuid
import tempfile
import shutil
import re
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.agent import LocalDoremonMaster
from app.core.config import BASE_DIR

app = FastAPI(title="Doremon Local AI Modular", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = LocalDoremonMaster()

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.get("/chat")
def chat(message: str = Query(...), session_id: str = Query(None)):
    def event_stream():
        try:
            for token in bot.chat_stream(message, session_id):
                yield f"data: {json.dumps({'token': token})}\n\n"
        finally:
            yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/models")
def list_models():
    return {"models": bot.get_available_models(), "active": bot.current_model}

@app.post("/models/switch")
def switch_model(model: str = Query(...), session_id: str = Query(None)):
    msg = bot.switch_model(model, session_id=session_id)
    return {"message": msg, "active": bot.current_model}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex[:8]}.pdf")
    with open(tmp_path, "wb") as fh: shutil.copyfileobj(file.file, fh)
    res = bot.read_legal_pdf(tmp_path)
    os.remove(tmp_path)
    return {"success": not res.startswith("[ERROR]"), "message": res}

class DrawRequest(BaseModel):
    prompt: str
    seed: int | None = None

@app.post("/draw")
def draw_image(req: DrawRequest):
    msg, filename = bot.draw(req.prompt, req.seed)
    return {"success": bool(filename), "message": msg, "filename": filename}

@app.get("/images/{filename}")
def get_image(filename: str):
    return FileResponse(os.path.join(BASE_DIR, filename))

@app.get("/memory/stats")
def memory_stats():
    return {"total_records": bot.db.msg_count}

@app.delete("/memory/wipe")
def wipe_memory():
    bot.db.clear()
    return {"message": "Memory wiped."}

@app.get("/sessions")
def get_sessions():
    sessions = []
    sessions_dir = os.path.join(BASE_DIR, "sessions")
    if os.path.exists(sessions_dir):
        for f in os.listdir(sessions_dir):
            if f.endswith(".json"):
                with open(os.path.join(sessions_dir, f)) as file:
                    sessions.append(json.load(file))
    sessions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return {"sessions": sessions, "active_id": bot.session_id}

@app.post("/sessions/new")
def new_session():
    bot.new_session()
    return {"message": "Started new chat thread.", "id": bot.session_id}

@app.post("/sessions/{session_id}")
def load_session(session_id: str):
    bot.load_session(session_id)
    return {"message": "Session loaded.", "messages": bot.short_term_memory}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)