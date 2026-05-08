import ollama
from app.core.config import EMBED_MODEL

def _embed(text: str) -> list:
    """Unified embedding call for both old and new Ollama API versions."""
    try:
        result = ollama.embed(model=EMBED_MODEL, input=text)
        return result["embeddings"][0]
    except AttributeError:
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

def _extract_chunk_token(chunk) -> str:
    """Extract token from stream chunk supporting dict or object formats."""
    try:
        if hasattr(chunk, "message"):
            return chunk.message.content or ""
        return (chunk.get("message") or {}).get("content", "")
    except Exception:
        return ""