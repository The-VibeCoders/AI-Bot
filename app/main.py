import os
import json
import uuid
import tempfile
import shutil
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.agent import LocalDoremonMaster
from app.core.config import BASE_DIR, LOCAL_USER
from app.core.providers import key_store
from app.core.security import get_current_user_id
from app.services.image_editing import image_editor
from app.services import pdf_preview
import httpx
import re
import threading

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

class ModelAddRequest(BaseModel):
    model_id: str
    provider: str = "ollama"
    api_key: str | None = None


class RemoveModelRequest(BaseModel):
    model_id: str


class ProviderAddRequest(BaseModel):
    name: str
    type: str = "openai_compatible"
    api_key: str
    base_url: str | None = None


@app.get("/chat")
def chat(message: str = Query(...), session_id: str = Query(None), attachments: str = Query(None)):
    if session_id in ("null", "undefined", ""):
        session_id = None
    def event_stream():
        try:
            # Pass attachments to chat_stream
            for event in bot.chat_stream(user_id=LOCAL_USER, user_msg=message, session_id=session_id, attachments=attachments):
                yield f"data: {event}\n\n"
        finally:
            yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/cancel")
def cancel_stream(session_id: str = Query(...)):
    if session_id in ("null", "undefined", ""):
        return {"message": "Invalid session ID"}
    bot.cancel_stream(session_id)
    return {"message": "Stream cancelled"}

@app.get("/models")
def list_models():
    raw = bot.get_available_models()
    cloud = []
    local = []
    for m in raw:
        prefix, raw_id = m.split(":", 1) if ":" in m else (None, m)
        if prefix and key_store.get_provider_for_model(raw_id):
            cloud.append(m)
        else:
            local.append(m)
    return {"models": raw, "active": bot.get_current_model(LOCAL_USER), "cloud": cloud, "local": local}

@app.post("/models/switch")
def switch_model(model: str = Query(...), session_id: str = Query(None)):
    if session_id in ("null", "undefined", ""):
        session_id = None
    msg = bot.switch_model(user_id=LOCAL_USER, model_name=model, session_id=session_id)
    return {"message": msg, "active": bot.get_current_model(LOCAL_USER)}

@app.post("/models/add")
def add_model(req: ModelAddRequest):
    if req.provider == "ollama":
        try:
            import ollama
            ollama.pull(req.model_id)
            return {"message": f"Successfully pulled model {req.model_id}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to pull model: {e}")
    else:
        pinfo = bot.provider_manager.list_providers()
        known = {p["name"] for p in pinfo}
        if req.provider not in known:
            raise HTTPException(status_code=400, detail=f"Provider '{req.provider}' not found. Add it first via POST /providers/add")
        bot.provider_manager.save_key(req.provider, req.model_id, req.api_key)
        return {"message": f"Added {req.provider}:{req.model_id}"}

@app.post("/models/remove")
def remove_model(req: RemoveModelRequest):
    ok = bot.provider_manager.remove_model(req.model_id)
    if ok:
        return {"message": f"Removed {req.model_id}"}
    raise HTTPException(status_code=404, detail="Model not found")

@app.get("/providers")
def list_providers():
    return {"providers": bot.provider_manager.list_providers()}

@app.post("/providers/add")
def add_provider(req: ProviderAddRequest):
    name = req.name.strip().lower()
    if not name:
        raise HTTPException(status_code=400, detail="Provider name required")
    if name in ("ollama", "openai", "anthropic", "gemini"):
        raise HTTPException(status_code=400, detail=f"Cannot override built-in provider '{name}'")
    supported = ("openai", "anthropic", "openai_compatible", "gemini")
    if req.type not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported provider type '{req.type}'. Supported: {', '.join(supported)}")
    bot.provider_manager.save_provider(name, req.type, req.api_key, req.base_url)
    detected = []
    try:
        detected = bot.provider_manager.detect_models(req.type, req.api_key, req.base_url)
        for m in detected:
            key_store.save_key(name, m, req.api_key)
    except Exception:
        pass
    return {"message": f"Provider '{name}' added", "name": name, "type": req.type, "detected_models": detected}

@app.post("/providers/remove")
def remove_provider(name: str = Query(...)):
    name = name.strip().lower()
    if name in ("ollama", "openai", "anthropic", "gemini"):
        raise HTTPException(status_code=400, detail=f"Cannot remove built-in provider '{name}'")
    ok = bot.provider_manager.remove_provider(name)
    if ok:
        return {"message": f"Provider '{name}' removed"}
    raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")


@app.get("/ollama-cloud-models")
def get_ollama_cloud_models():
    """Fetch available Ollama cloud models by scraping the Ollama website."""
    try:
        response = httpx.get("https://ollama.com/search?c=cloud", timeout=15.0)
        response.raise_for_status()
        html = response.text
        
        # Extract model names from href="/library/xxx" links
        model_matches = re.findall(r'href="/library/([^"]+)"', html)
        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for model in model_matches:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)
        
        # Also try to get descriptions by looking for nearby text
        models_with_info = []
        for model_slug in unique_models[:50]:  # Limit to first 50 to avoid overload
            # Try to find description near the model link
            pattern = rf'href="/library/{re.escape(model_slug)}"[^>]*>([^<]+)</a>[^<]*<p[^>]*>([^<]*)</p>'
            desc_match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            description = ""
            if desc_match:
                # Clean up the description
                description = re.sub(r'\s+', ' ', desc_match.group(2)).strip()
                if not description:
                    description = desc_match.group(1).strip()
            else:
                # Fallback: just use the model slug as description
                description = model_slug.replace('-', ' ').title()
            
            models_with_info.append({
                "name": model_slug,
                "description": description[:200]  # Limit description length
            })
        
        return {"models": models_with_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Ollama cloud models: {str(e)}")


@app.post("/models/pull")
def pull_ollama_model(model_id: str = Query(...)):
    """Pull an Ollama model (local or cloud) in the background."""
    def pull_model():
        try:
            import ollama
            ollama.pull(model_id)
        except Exception as e:
            # Log error but don't crash the background thread
            print(f"Error pulling model {model_id}: {e}")
    
    # Run pull in background thread to avoid blocking the API response
    thread = threading.Thread(target=pull_model, daemon=True)
    thread.start()
    
    return {"message": f"Pull started for {model_id}. This may take a while."}

SUPPORTED_UPLOAD_EXTS = {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.txt', '.py', '.js', '.ts', '.html', '.css', '.json', '.md', '.csv', '.xml', '.yaml', '.yml', '.sh', '.bat', '.ps1'}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Query(None)):
    if session_id in ("null", "undefined", ""):
        session_id = None

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_UPLOAD_EXTS:
        return {"success": False, "message": f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_UPLOAD_EXTS))}", "attachment": None}

    filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    upload_path = os.path.join(BASE_DIR, "uploads", filename)

    with open(upload_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    sid = session_id or bot.get_active_session(LOCAL_USER)
    msg = ""
    attachment = bot.add_attachment(LOCAL_USER, filename, session_id=sid, display_name=file.filename)

    if ext == ".pdf":
        msg = bot.read_legal_pdf(upload_path, user_id=LOCAL_USER, session_id=sid)
        pdf_info = pdf_preview.get_pdf_info(upload_path)
        thumb_url = pdf_preview.generate_thumbnail(upload_path)
        attachment["pdf_info"] = pdf_info
        attachment["thumbnail_url"] = thumb_url
        attachment["type"] = "pdf"
    elif ext in {'.txt', '.py', '.js', '.ts', '.html', '.css', '.json', '.md', '.csv', '.xml', '.yaml', '.yml', '.sh', '.bat', '.ps1'}:
        try:
            with open(upload_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            bot.db.remember("system_document", f"File: {file.filename}\n\n{content[:5000]}", user_id=LOCAL_USER, session_id=sid)
            msg = f"📄 Read '{file.filename}' ({len(content)} bytes)"
        except Exception as e:
            msg = f"[ERROR] Failed to read file: {e}"
        attachment["type"] = "text"
    elif ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}:
        msg = f"🖼️ Image '{file.filename}' uploaded ({os.path.getsize(upload_path) / 1024:.1f} KB)"
        attachment["type"] = "image"
        attachment["thumbnail_url"] = f"/uploads/{filename}"
    else:
        msg = f"📎 File '{file.filename}' uploaded ({os.path.getsize(upload_path) / 1024:.1f} KB)"

    return {"success": not msg.startswith("[ERROR]"), "message": msg, "attachment": attachment}

@app.get("/uploads/{filename}")
def get_upload(filename: str):
    filepath = os.path.join(BASE_DIR, "uploads", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="File not found")

class ImageEditRequest(BaseModel):
    action: str
    filepath: str | None = None
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    degrees: float | None = None
    radius: int | None = None
    factor: float | None = None
    text: str | None = None
    font_size: int | None = None
    color: str | None = None
    border_width: int | None = None
    border_color: str | None = None

class AIEditRequest(BaseModel):
    filepath: str
    prompt: str
    strength: float = 0.6
    seed: int | None = None
    high_quality: bool = True

@app.post("/image/upload")
async def upload_image(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in image_editor.SUPPORTED_FORMATS:
        return {"success": False, "message": f"Unsupported format. Supported: {', '.join(image_editor.SUPPORTED_FORMATS)}"}

    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex[:8]}{ext}")
    with open(tmp_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    return {"success": True, "filepath": tmp_path, "filename": file.filename}

@app.post("/image/edit")
def edit_image(req: ImageEditRequest, user_id: str = Depends(get_current_user_id)):
    if not req.filepath:
        return {"success": False, "message": "filepath required"}

    color_tuple = None
    if req.color:
        try:
            parts = req.color.split(',')
            color_tuple = tuple(int(p.strip()) for p in parts)
        except Exception:
            color_tuple = (255, 255, 255)

    border_color_tuple = None
    if req.border_color:
        try:
            parts = req.border_color.split(',')
            border_color_tuple = tuple(int(p.strip()) for p in parts)
        except Exception:
            border_color_tuple = (0, 0, 0)

    actions = {
        "resize": lambda: image_editor.resize(req.filepath, req.width or 800, req.height or 600),
        "crop": lambda: image_editor.crop(req.filepath, req.x or 0, req.y or 0, req.width or 100, req.height or 100),
        "rotate": lambda: image_editor.rotate(req.filepath, req.degrees or 90),
        "flip_horizontal": lambda: image_editor.flip_horizontal(req.filepath),
        "flip_vertical": lambda: image_editor.flip_vertical(req.filepath),
        "blur": lambda: image_editor.blur(req.filepath, req.radius or 2),
        "sharpen": lambda: image_editor.sharpen(req.filepath, req.factor or 1.5),
        "brightness": lambda: image_editor.adjust_brightness(req.filepath, req.factor or 1.2),
        "contrast": lambda: image_editor.adjust_contrast(req.filepath, req.factor or 1.2),
        "grayscale": lambda: image_editor.grayscale(req.filepath),
        "text": lambda: image_editor.add_text(req.filepath, req.text or "", req.x or 10, req.y or 10,
                                              req.font_size or 24, color_tuple or (255, 255, 255)),
        "border": lambda: image_editor.add_border(req.filepath, req.border_width or 5, border_color_tuple or (0, 0, 0)),
    }

    if req.action not in actions:
        return {"success": False, "message": f"Unknown action: {req.action}"}

    result = actions[req.action]()
    if result:
        return {"success": True, "filepath": result}
    return {"success": False, "message": "Edit failed"}

@app.post("/image/ai-edit")
def ai_edit_image(req: AIEditRequest, user_id: str = Depends(get_current_user_id)):
    msg, filename = bot.img_service.edit_image(req.filepath, req.prompt, req.strength, req.seed, req.high_quality)
    return {"success": bool(filename), "message": msg, "filename": filename}

class DrawRequest(BaseModel):
    prompt: str
    seed: int | None = None

@app.post("/draw")
def draw_image(req: DrawRequest, user_id: str = Depends(get_current_user_id)):
    msg, filename = bot.draw(req.prompt, req.seed)
    return {"success": bool(filename), "message": msg, "filename": filename}

@app.get("/images/{filename}")
def get_image(filename: str):
    return FileResponse(os.path.join(BASE_DIR, filename))

@app.get("/memory/stats")
def memory_stats(user_id: str = Depends(get_current_user_id)):
    records = bot.db.collection.get(where={"user_id": user_id})
    count = len(records.get("ids", [])) if records else 0
    return {"total_records": count}

@app.delete("/memory/wipe")
def wipe_memory(user_id: str = Depends(get_current_user_id)):
    bot.db.clear(user_id=user_id)
    return {"message": "Memory wiped for your account."}

@app.delete("/memory/context")
def clear_context(session_id: str = Query(None), user_id: str = Depends(get_current_user_id)):
    if session_id in ("null", "undefined", ""):
        session_id = None
    sid = session_id or bot.get_active_session(user_id)
    bot._get_stm(user_id, sid).clear()
    bot.save_session(user_id, sid, force=True)
    return {"message": "Context cleared."}

@app.get("/memory/recent")
def get_recent_memories(limit: int = 10, user_id: str = Depends(get_current_user_id)):
    memories = bot.db.get_recent(user_id=user_id, limit=limit)
    return {"memories": memories}

@app.get("/sessions")
def get_sessions(user_id: str = Depends(get_current_user_id)):
    sessions = []
    user_dir = os.path.join(BASE_DIR, "sessions", user_id)
    if os.path.exists(user_dir):
        for f in os.listdir(user_dir):
            if f.endswith(".json"):
                with open(os.path.join(user_dir, f)) as file:
                    sessions.append(json.load(file))
    sessions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return {"sessions": sessions, "active_id": bot.get_active_session(user_id)}

@app.post("/sessions/new")
def new_session(user_id: str = Depends(get_current_user_id)):
    bot.new_session(user_id=user_id)
    return {"message": "Started new chat thread.", "id": bot.get_active_session(user_id)}

@app.post("/sessions/{session_id}")
def load_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    if session_id in ("null", "undefined", ""):
        # Clean up orphaned files with invalid session IDs
        user_dir = os.path.join(BASE_DIR, "sessions", user_id)
        if os.path.exists(user_dir):
            for f in os.listdir(user_dir):
                if f in ("null.json", "undefined.json"):
                    os.remove(os.path.join(user_dir, f))
        return {"message": "Invalid session ID", "messages": []}
    success = bot.load_session(user_id, session_id)
    if not success:
        return {"message": "Session not found."}
    
    state = bot._get_user_state(user_id)
    session_data = state["sessions"].get(session_id, {"messages": [], "attachments": []})
    messages = session_data["messages"]
    attachments = session_data["attachments"]
    
    return {"message": "Session loaded.", "messages": messages, "attachments": attachments}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    user_dir = os.path.join(BASE_DIR, "sessions", user_id)
    filepath = os.path.join(user_dir, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"message": "Session deleted"}
    return {"message": "Session not found"}

# ── Personality & Work Directory & Git Endpoints ────────────────────

class ApproveRequest(BaseModel):
    approved: bool

class SetWorkDirRequest(BaseModel):
    path: str

class SwitchProjectRequest(BaseModel):
    project_id: str

@app.get("/personalities")
def list_personalities():
    from app.personalities.registry import PersonalityRegistry
    personalities = PersonalityRegistry.list()
    return {"personalities": [
        {"id": p.id, "name": p.name, "description": p.description, "icon": p.icon}
        for p in personalities
    ]}

@app.post("/personalities/switch")
def switch_personality(personality_id: str = Query(...)):
    msg = bot.set_personality(LOCAL_USER, personality_id)
    return {"message": msg}

@app.get("/work-dir")
def get_work_dir():
    return {"work_dir": bot.get_work_dir(LOCAL_USER)}

@app.post("/work-dir/set")
def set_work_dir(req: SetWorkDirRequest):
    msg = bot.set_work_dir(LOCAL_USER, req.path)
    return {"message": msg}

@app.post("/git/undo")
def git_undo():
    msg = bot.git_undo(LOCAL_USER)
    return {"message": msg}

@app.post("/git/redo")
def git_redo():
    msg = bot.git_redo(LOCAL_USER)
    return {"message": msg}

@app.get("/projects")
def get_projects():
    projects = bot.get_recent_projects(LOCAL_USER)
    return {"projects": projects}

@app.post("/projects/switch")
def switch_project(req: SwitchProjectRequest):
    projects = bot.get_recent_projects(LOCAL_USER)
    target = next((p for p in projects if p.get("id") == req.project_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Project not found")
    msg = bot.set_work_dir(LOCAL_USER, target["path"])
    return {"message": msg, "path": target["path"]}

@app.delete("/projects/{project_id}")
def delete_project(project_id: str):
    bot.remove_recent_project(LOCAL_USER, project_id)
    return {"message": "Project removed from recent list"}

@app.post("/approve/{req_id}")
def approve_tool(req_id: str, req: ApproveRequest):
    bot.approve_tool(req_id, req.approved)
    return {"message": "Approval recorded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)