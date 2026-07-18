# Doremon Local AI Agent

A local-first AI assistant with chat, web search, image generation/editing, PDF ingestion, and memory — all running on your machine.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI (app/main.py)                       │
│  Port 8000 · SSE streaming · JWT auth · SQLite + ChromaDB       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  Agent   │  │  Image   │  │  Image   │  │   Web Scraper  │  │
│  │  Core    │  │  Service │  │  Editor  │  │   (DuckDuckGo) │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────────────┐  │
│  │ ChromaDB │  │  PDF     │  │  Ollama (external process)    │  │
│  │ Memory   │  │  Ingest  │  │  deepseek-r1:7b + nomic-embed │  │
│  └──────────┘  └──────────┘  └───────────────────────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────────────┐  │
│  │ SQLite   │  │ Sessions │  │  Stable Diffusion            │  │
│  │ (users)  │  │ (JSON)   │  │  (CUDA 12.8 / CPU fallback)  │  │
│  └──────────┘  └──────────┘  └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                         ▲
                         │ HTTP / SSE
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Frontend (static/index.html)                    │
│   Vanilla JS · EventSource streaming · Dark UI · Image Editor   │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
mini project v2/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server, routes, auth
│   ├── core/
│   │   ├── agent.py         # Chat logic, session management, model orchestration
│   │   ├── config.py        # Settings (env vars, model IDs, secrets)
│   │   ├── database.py      # ChromaDB vector memory (recall/remember)
│   │   ├── security.py      # JWT creation, password hashing, auth middleware
│   │   └── user_db.py       # SQLite User model (SQLAlchemy)
│   ├── services/
│   │   ├── image_service.py # Stable Diffusion text-to-image + img2img
│   │   ├── image_editing.py # Pillow-based image manipulation (crop, resize, etc.)
│   │   ├── pdf_service.py   # PDF text extraction + vector ingestion
│   │   └── scraper_service.py # DuckDuckGo search + webpage scraping
│   └── utils/
│       └── compat.py        # Ollama embedding + chunk token extraction
├── static/
│   ├── index.html           # SPA frontend
│   ├── styles.css           # Dark theme UI
│   └── App.js               # Frontend logic (chat, auth, sessions, image editor)
├── sessions/                # Chat session JSON files (per-user subdirectories)
├── chroma_db/               # ChromaDB persistent vector store
├── cli.py                   # CLI interface for testing without frontend
├── requirements.txt         # Python dependencies
└── users.db                 # SQLite database (users table)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.14, FastAPI, Uvicorn |
| LLM | Ollama (`deepseek-r1:7b` default) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Auth | JWT (python-jose), bcrypt passwords |
| User DB | SQLite via SQLAlchemy |
| Image Gen | Stable Diffusion 1.5 (diffusers, CUDA 12.8) |
| Image Edit | Pillow (resize, crop, rotate, blur, etc.) |
| Streaming | Server-Sent Events (EventSource) |
| Frontend | Vanilla JS, no frameworks |
| Web Search | DuckDuckGo Search SDK + BeautifulSoup |

## API Reference

### Auth

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/register` | No | Create account (password ≥ 6 chars) |
| `POST` | `/login` | No | Login, returns `access_token` + `refresh_token` |
| `POST` | `/token/refresh` | Bearer | Exchange refresh token for new access token |

### Chat

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/chat?message=...&session_id=...` | Bearer/Query | SSE streaming chat |
| `GET` | `/models` | Bearer | List available Ollama models |
| `POST` | `/models/switch?model=...&session_id=...` | Bearer | Switch active model |

### Memory

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/memory/stats` | Bearer | Per-user memory count |
| `DELETE` | `/memory/wipe` | Bearer | Wipe user's vector memory |
| `DELETE` | `/memory/context` | Bearer | Clear current session context |
| `GET` | `/memory/recent?limit=10` | Bearer | Recent memories |

### Sessions

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/sessions` | Bearer | List sessions with active ID |
| `POST` | `/sessions/new` | Bearer | Create new session |
| `POST` | `/sessions/{id}` | Bearer | Load a session |
| `DELETE` | `/sessions/{id}` | Bearer | Delete a session |

### Images

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/draw` | Bearer | Generate image from prompt (SD) |
| `POST` | `/image/upload` | Bearer | Upload image for editing |
| `POST` | `/image/edit` | Bearer | Apply edits (crop, resize, etc.) |
| `POST` | `/image/ai-edit` | Bearer | AI image-to-image editing |
| `GET` | `/images/{filename}` | No | Serve generated/edited images |

### Files

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/upload` | Bearer | Upload & ingest PDF |

## Features

### Chat
- Streaming responses via SSE with real-time token display
- Session management (create, switch, delete conversations)
- Web search integration (auto-triggered on search-like queries)
- Long-term memory via ChromaDB vector recall
- Markdown rendering (code blocks, inline code)

### Image Generation
- Stable Diffusion 1.5 (`realistic-vision-v51` model)
- GPU acceleration via CUDA 12.8 (RTX 3050 Ti: ~3.5s for 20 steps)
- CPU fallback when no GPU available
- Seed-based reproducibility

### Image Editor
- Traditional edits: resize, crop, rotate, flip, blur, sharpen
- Color adjustments: brightness, contrast, grayscale
- Text overlay, borders
- AI-powered img2img editing with Stable Diffusion

### PDF Ingestion
- Extract text from PDFs page by page
- Vectorize and store in ChromaDB for later recall
- Batch embedding for efficiency

### Auth System
- JWT-based authentication (1 week expiry)
- Refresh token mechanism (1 day expiry)
- Auto-refresh on 401 in frontend (transparent to user)
- Password minimum length validation (6 chars)

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally (default model: `deepseek-r1:7b`)
- NVIDIA GPU with CUDA (optional, for image generation)

### Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install embedding model (required)
ollama pull nomic-embed-text

# Pull chat model (optional, deepseek-r1:7b default)
ollama pull deepseek-r1:7b
```

### Configuration

Set environment variables (optional):

```bash
set DOREMON_SECRET=your-secure-secret-key  # JWT signing key (dev default provided)
```

### Running

```bash
python -m app.main
```

Open http://localhost:8000 in a browser. Register an account and start chatting.

### CLI Mode

```bash
python cli.py
```

## GPU Image Generation

The image service auto-detects CUDA. If you have an NVIDIA GPU:

```bash
# PyTorch with CUDA 12.8 is already installed
torch==2.11.0+cu128
```

20-step 512×512 images generate in ~3.5 seconds on RTX 3050 Ti (4GB VRAM).

## Current Status

### Working
- Chat with streaming, session management, context persistence
- Web search integration (DuckDuckGo)
- Vector memory (ChromaDB recall + remember)
- PDF ingestion
- Image generation on GPU (~3.5s per image)
- Image editing (traditional + AI img2img)
- JWT auth with refresh tokens
- Auto-refresh on 401 in frontend
- Session CRUD (create, read, delete)

### Known Issues

| Issue | Location | Status |
|-------|----------|--------|
| Image service double-initialized | `main.py:31` + `agent.py:18` | Two instances created |
| No multi-user session isolation in ChromaDB `get_recent` | `database.py:111-126` | Needs user_id filter |
| Corrupted session files silently skipped | `agent.py:49` | No logging |
| Unicode emojis cause console errors on Windows | `image_service.py` | cp1252 encoding |
| Deprecation warnings for VAE methods | `image_service.py` | Using deprecated API |
| No upload file size limits | `main.py` `/upload` + `/image/upload` | No max_size enforcement |
| `switch_model` wipes all sessions when no session_id given | `agent.py:106-109` | Destructive behavior |
| SSE token in URL query param (EventSource limitation) | `App.js:285` | Exposed in server logs |

### Fixed (Recent)

| Fix | Date |
|-----|------|
| GPU image generation (replaced CPU-only PyTorch with CUDA 12.8) | 2026-05-20 |
| SECRET_KEY reads from `DOREMON_SECRET` env var | 2026-05-20 |
| Token refresh endpoint + auto-refresh on 401 | 2026-05-20 |
| Password validation (≥6 chars) | 2026-05-20 |
| DELETE endpoint for sessions | 2026-05-20 |
| memory_stats filtered per-user | 2026-05-20 |
| Session loading bugs (active_id tracking, orphan cleanup, null rejection) | 2026-05-20 |
| Markdown rendering for assistant messages | 2026-05-20 |
| Stop button for streaming responses | 2026-05-20 |
| Static file routing (fixed 404s) | 2026-05-20 |
