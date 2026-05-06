# Doremon Local AI Agent

Doremon is a high-performance, fully local AI assistant designed for privacy and speed. It combines a powerful vector-based memory engine, local LLM orchestration via Ollama, image generation, and live web search into a single cohesive platform accessible via a sleek web interface.

## 🌟 Key Features

* **100% Local Processing:** Powered by Ollama for LLMs (defaulting to `deepseek-r1:7b`) and Hugging Face Diffusers for image generation. Everything stays on your machine.
* **Advanced Memory System:** Features a high-speed vector engine using NumPy and JSONL for Long-Term Memory (LTM), and a sliding window for Short-Term Memory (STM).
* **Live Web Search:** Autonomously searches the web using DuckDuckGo (`ddgs`) and scrapes context using BeautifulSoup when factual queries require live internet access.
* **Image Generation:** Integrated Stable Diffusion pipeline (`realistic-vision-v51`) for generating images directly in the chat with caching for fast VRAM inference.
* **PDF Document Ingestion:** Read and vectorize local PDF files directly into the agent's long-term memory for local Retrieval-Augmented Generation (RAG).
* **FastAPI Backend & Modern GUI:** Serves a responsive, dark-themed web UI with session management, markdown support, Server-Sent Events (SSE) streaming chat, and graphical tools to wipe or inspect memory.

## 🛠️ Tech Stack

* **Backend:** FastAPI, Uvicorn, Python
* **AI & ML Orchestration:** Ollama, PyTorch, Diffusers, Transformers
* **Vector Database:** NumPy (Matrix Math), JSONL
* **Web Scraping:** DuckDuckGo Search (`ddgs`), BeautifulSoup4, Requests
* **Frontend:** HTML5, CSS3 (Inter & JetBrains Mono fonts), Vanilla JavaScript

## 🚀 Installation & Setup

**1. Install Prerequisites** Ensure you have Python installed and a CUDA-capable GPU for image generation. You also need [Ollama](https://ollama.ai/) installed and running locally on your machine.

**2. Install Dependencies**
Install all required Python packages using the provided requirements file:
bash
pip install -r requirements.txt

3. Pull an LLM
Before running the agent, pull your preferred model via Ollama (Doremon defaults to DeepSeek):
Bash
ollama pull deepseek-r1:7b

4. Run the Server
Start the FastAPI backend, which will automatically bind to 0.0.0.0:8000 to ensure it is reachable locally or inside Docker:
Bash
python server.py


💻 Usage & UI Guide
Access the web UI by navigating to http://localhost:8000 in your web browser.

Available Tools:

💬 Chat & Search: Speak naturally. Doremon will evaluate if it needs to search the web automatically for factual or location-based questions.

🎨 Generate Images: Type a visual prompt and click the palette icon (🎨) in the chat bar.

📄 Upload PDF: Click the document icon (📄) to upload a PDF file and vectorize its contents into the agent's knowledge base.

🧹 Clear Context: Click the broom icon (🧹) to reset the short-term conversation context for a fresh start.

⚙️ Change Model: Use the sidebar to switch between any downloaded Ollama models on the fly.

🧠 Manage Memory: View recent vectorized memories or completely wipe the database using the sidebar utilities.

(For terminal power users, you can bypass the web UI and run the CLI mode directly via the core bot.py script.)
