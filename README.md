# Doremon Local AI Agent

Doremon is a local-first AI assistant with a FastAPI backend and a vanilla-JS web UI. It supports chat with long-term vector memory, web search, PDF ingestion, image generation/editing, and an autonomous **Coding Agent** mode that can read/write files, run shell commands, and manage git — all through a pluggable multi-provider LLM layer (Ollama, OpenAI, Anthropic, Gemini, Mistral, Cohere, or any OpenAI-compatible endpoint).

> This is a single-local-user app — there is no login/registration system. All requests operate as one local account (`local_user`).

## 🌟 Key Features

- **Multi-Provider LLM Support:** Chat with local models via [Ollama](https://ollama.ai/) (default) or plug in cloud providers — OpenAI, Anthropic, Google Gemini, Mistral, Cohere, or any OpenAI-compatible API. Provider API keys are added at runtime through the UI/API and stored encrypted on disk.
- **Personalities:**
  - **Standard** — general-purpose chat assistant with web search, memory, and PDF analysis.
  - **Coding Agent** — an autonomous engineering assistant that can list directories, read/write/patch files, run shell commands (with user approval), and use git (`status`, `commit`, `undo`, `redo`) inside a configurable working directory.
- **Advanced Memory System:** Vector-based long-term memory (LTM) via ChromaDB, plus a sliding-window short-term memory (STM) per chat session.
- **Live Web Search:** Auto-triggered DuckDuckGo search with page scraping (via BeautifulSoup) when a query needs live information.
- **Image Generation & Editing:** Local Stable Diffusion pipeline (`realistic-vision-v51`) for text-to-image and AI-assisted image-to-image editing, plus a traditional Pillow-based editor (resize, crop, rotate, flip, blur, sharpen, brightness/contrast, text overlay, borders).
- **PDF & File Ingestion:** Upload PDFs, code, or text files; content is parsed/vectorized into long-term memory for later recall.
- **Project & Session Management:** Multiple chat sessions (create/switch/delete), a recent-projects list, and per-session file attachments.
- **Streaming Chat UI:** FastAPI backend streams tokens over Server-Sent Events (SSE) to a dark-themed, framework-free web frontend, with a stop/cancel button for in-flight responses.

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn, Python
- **LLM Providers:** Ollama, OpenAI, Anthropic, Google Gemini, Mistral, Cohere (via their respective SDKs), plus generic OpenAI-compatible endpoints
- **Vector Memory:** ChromaDB (persistent, cosine similarity)
- **Image Generation/Editing:** PyTorch, Diffusers, Transformers, Accelerate, Pillow
- **Web Scraping:** DuckDuckGo Search (`ddgs`), BeautifulSoup4, Requests, httpx
- **PDF Handling:** pypdf
- **Secrets Storage:** Provider API keys encrypted at rest (Fernet) in `provider_keys.json`
- **Frontend:** HTML5, CSS3, vanilla JavaScript (`static/App.js`), SSE via `EventSource`

## 📁 Project Structure

```
project/
├── app/
│   ├── main.py                     # FastAPI app, all HTTP/SSE routes
│   ├── core/
│   │   ├── agent.py                # Chat orchestration, sessions, tool-call loop, git ops
│   │   ├── config.py                # Paths, model defaults, tunables
│   │   ├── database.py              # ChromaDB vector memory (remember/recall)
│   │   ├── security.py              # Local-user resolver (no auth — single user)
│   │   └── providers/
│   │       ├── manager.py           # Resolves model id -> provider instance
│   │       ├── key_store.py         # Encrypted storage for provider API keys
│   │       ├── base.py               # LLMProvider interface + error codes
│   │       ├── ollama_provider.py
│   │       ├── openai_provider.py
│   │       ├── anthropic_provider.py
│   │       ├── gemini_provider.py
│   │       ├── mistral_provider.py
│   │       ├── cohere_provider.py
│   │       └── openai_compatible_provider.py
│   ├── personalities/
│   │   ├── registry.py               # Personality discovery/lookup
│   │   ├── base.py                    # BasePersonality interface
│   │   ├── standard.py                 # Standard chatbot personality
│   │   └── coding_agent.py             # Coding Agent personality + file/shell/git tools
│   ├── services/
│   │   ├── image_service.py            # Stable Diffusion text-to-image / img2img
│   │   ├── image_editing.py            # Pillow-based image editing
│   │   ├── pdf_service.py              # PDF text extraction + vector ingestion
│   │   ├── pdf_preview.py              # PDF info + thumbnail generation
│   │   └── scraper_service.py          # Web search + scraping
│   └── utils/
│       └── compat.py                   # Embedding helper + streaming chunk parsing
├── static/
│   ├── index.html                      # SPA frontend
│   ├── styles.css                      # Dark theme UI
│   └── App.js                          # Frontend chat/session/image-editor logic
├── cli.py                              # Terminal chat client (no web UI required)
└── requirements.txt                    # Python dependencies
```

*(Runtime-generated, not part of the repo: `sessions/`, `chroma_db/`, `uploads/`, `provider_keys.json`, `.secret_key`.)*

## 🚀 Installation & Setup

**1. Install Prerequisites**
Python 3.10+ is required. A CUDA-capable GPU is recommended (but not required — there's a CPU fallback) for image generation. If you want to use local models, install and run [Ollama](https://ollama.ai/).

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Pull a Local LLM (optional, if using Ollama)**

```bash
ollama pull llama3.2
```

> Note: cloud providers (OpenAI, Anthropic, Gemini, Mistral, Cohere, or an OpenAI-compatible endpoint) can be added and used instead of/alongside Ollama — see **Adding Cloud Providers** below. No provider is hardcoded as required at startup.

**4. Run the Server**

```bash
python -m app.main
```

This starts the FastAPI backend on `0.0.0.0:8000`.

## 💻 Usage & UI Guide

Open `http://localhost:8000` in your browser.

**Available Tools:**

- 💬 **Chat:** Talk naturally. Doremon automatically pulls in relevant memory, recently uploaded documents, and live web search results as needed.
- 🧠 **Switch Personality:** Choose between **Standard** chat and **Coding Agent** mode from the UI.
- ⌨️ **Coding Agent:** Set a working directory, then ask Doremon to read/write/patch files, run shell commands (you'll be prompted to approve each command), or manage git history (`git_status`, `git_commit`, `git_undo`, `git_redo`).
- 🎨 **Generate Images:** Enter a prompt and click the palette icon to generate with Stable Diffusion.
- 🖌️ **Edit Images:** Upload an image for traditional edits (crop, resize, rotate, filters, text, borders) or AI-assisted img2img editing.
- 📄 **Upload Files:** Upload PDFs, code, or text files — content is ingested into memory for later recall in that session.
- 🧹 **Clear Context:** Reset the short-term context for the current session.
- ⚙️ **Manage Models & Providers:** Add/remove local Ollama models, or register a cloud provider (with API key) and pick from its detected models.
- 🧠 **Manage Memory:** View memory stats, browse recent memories, or wipe all stored memory.
- 📂 **Sessions & Projects:** Create, switch, and delete chat sessions; switch between recent working-directory "projects."

### Adding Cloud Providers

From the UI (or via `POST /providers/add`), register a provider with a name, type (`openai`, `anthropic`, `gemini`, or `openai_compatible`), and API key. Once added, its models become selectable from the model list. API keys are encrypted before being written to disk.

### CLI Mode

For a terminal-only chat client that doesn't require the web UI:

```bash
python cli.py
```

CLI commands:
```
/draw <prompt>    - Generate an image
/read <path>      - Ingest a PDF into memory
/model <name>     - Switch LLM model
/list             - List all available models
/clear            - Clear current chat context
/wipe             - Wipe all vector memory
/help             - Show the guide
/quit             - Exit
```

## License

See [LICENSE](LICENSE) for details.
