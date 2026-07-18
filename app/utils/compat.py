import ollama
from app.core.config import EMBED_MODEL

def _embed(text: str) -> list:
    """Unified embedding call for both old and new Ollama API versions."""
    try:
        result = ollama.embed(model=EMBED_MODEL, input=text)
        embeddings = result.get("embeddings", [])
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        # Fallback for older API versions
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    except Exception as e:
        # Final fallback
        return [0.0] * 768  # Return zero vector of standard embedding size

def _extract_chunk_token(chunk) -> str:
    """Extract token from stream chunk supporting dict or object formats."""
    try:
        if hasattr(chunk, "message"):
            return chunk.message.content or ""
        return (chunk.get("message") or {}).get("content", "")
    except Exception:
        return ""