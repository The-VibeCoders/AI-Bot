import os
from pypdf import PdfReader
from app.core.database import ChromaMemoryDB

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Splits text into overlapping chunks for better vector retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_pdf(file_path: str, db: ChromaMemoryDB, user_id: str, session_id: str) -> str:
    file_path = file_path.strip().strip('"').strip("'")
    if not os.path.exists(file_path) or not file_path.lower().endswith(".pdf"):
        return f"[ERROR] File not found or invalid format: '{file_path}'"

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        new_records, skipped = [], 0

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                skipped += 1; continue
                
            if len(text.strip()) <= 10:
                skipped += 1
                continue

            # Chunk the page text for better RAG performance
            chunks = _chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                content = f"Source: {os.path.basename(file_path)}, Page {i+1}, Chunk {chunk_idx+1}\n{chunk}"
                record = db.remember(role="system_document", content=content, user_id=user_id, session_id=session_id, batch=True)
                if record:
                    new_records.append(record)
                else:
                    skipped += 1
        
        if not new_records: return "[WARN] PDF read but no text extracted."
        db.batch_save(new_records, user_id=user_id, session_id=session_id)
        return f"📄 Success! {len(new_records)} chunks from {total_pages} pages vectorized into LTM. ({skipped} skipped)"
    except Exception as e:
        return f"[ERROR] PDF Parsing Error: {e}"