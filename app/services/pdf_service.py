import os
from pypdf import PdfReader
from app.core.database import FastMemoryDB

def ingest_pdf(file_path: str, db: FastMemoryDB) -> str:
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
                
            if len(text.strip()) > 10:
                record = db.remember(role="system_document", content=f"Source: {file_path}, Page {i+1}\n{text}", batch=True)
                if record: new_records.append(record)
                else: skipped += 1

        if not new_records: return "[WARN] PDF read but no text extracted."
        db.batch_save(new_records)
        return f"📄 Success! {len(new_records)}/{total_pages} pages vectorized into LTM. ({skipped} skipped)"
    except Exception as e:
        return f"[ERROR] PDF Parsing Error: {e}"