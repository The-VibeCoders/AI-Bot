import os
import uuid
from app.core.config import BASE_DIR

THUMB_DIR = os.path.join(BASE_DIR, "static", "thumbnails")
os.makedirs(THUMB_DIR, exist_ok=True)


def _get_thumbnail_path(pdf_filename: str) -> str:
    base = os.path.splitext(pdf_filename)[0]
    return os.path.join(THUMB_DIR, f"{base}_thumb.png")


def generate_thumbnail(pdf_path: str) -> str | None:
    """Generate a thumbnail image of the first PDF page. Returns the URL path or None."""
    thumb_path = _get_thumbnail_path(os.path.basename(pdf_path))
    if os.path.exists(thumb_path):
        return f"/static/thumbnails/{os.path.basename(thumb_path)}"

    try:
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(0.25, 0.25))
        pix.save(thumb_path)
        doc.close()
        return f"/static/thumbnails/{os.path.basename(thumb_path)}"
    except Exception:
        return None


def get_pdf_info(pdf_path: str) -> dict:
    """Return metadata about a PDF file."""
    info = {
        "page_count": 0,
        "file_size": 0,
        "file_size_display": "",
        "has_thumbnail": False,
    }
    if not os.path.exists(pdf_path):
        return info

    info["file_size"] = os.path.getsize(pdf_path)
    info["file_size_display"] = _format_size(info["file_size"])
    thumb_url = _get_thumbnail_path(os.path.basename(pdf_path))
    info["has_thumbnail"] = os.path.exists(thumb_url)

    try:
        import fitz
        doc = fitz.open(pdf_path)
        info["page_count"] = len(doc)
        doc.close()
    except Exception:
        pass

    return info


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
