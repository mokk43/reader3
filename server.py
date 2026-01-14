import os
import pickle
import json
import time
from functools import lru_cache
from typing import Optional, AsyncGenerator
import asyncio
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry
from translation import (
    load_translation_config, save_translation_config,
    segment_html_to_blocks, compute_source_hash,
    load_cached_translation, write_cached_translation,
    ollama_translate_single, sanitize_model_name
)

# Thread pool for running blocking translation calls
_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the book folders located?
BOOKS_DIR = "."

# Track in-progress translation jobs to prevent duplicates
# Key: (book_id, chapter_index, tlang, model)
_translation_jobs: dict[tuple, str] = {}  # value: "translating" | "error:message"
_translation_lock = Lock()


# --- Pydantic Models for API ---

class TranslationConfigUpdate(BaseModel):
    prompt_template: Optional[str] = None
    default_target_language: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_url: Optional[str] = None


class TranslationRequest(BaseModel):
    tlang: str

@lru_cache(maxsize=10)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from the pickle file.
    Cached so we don't re-read the disk on every click.
    """
    file_path = os.path.join(BOOKS_DIR, folder_name, "book.pkl")
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            book = pickle.load(f)
        return book
    except Exception as e:
        print(f"Error loading book {folder_name}: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []

    # Scan directory for folders ending in '_data' that have a book.pkl
    if os.path.exists(BOOKS_DIR):
        for item in os.listdir(BOOKS_DIR):
            if item.endswith("_data") and os.path.isdir(item):
                # Try to load it to get the title
                book = load_book_cached(item)
                if book:
                    books.append({
                        "id": item,
                        "title": book.metadata.title,
                        "author": ", ".join(book.metadata.authors),
                        "chapters": len(book.spine)
                    })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})

@app.get("/read/{book_id}", response_class=HTMLResponse)
async def redirect_to_first_chapter(book_id: str):
    """Helper to just go to chapter 0."""
    return await read_chapter(book_id=book_id, chapter_index=0)

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def read_chapter(request: Request, book_id: str, chapter_index: int):
    """The main reader interface."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    current_chapter = book.spine[chapter_index]

    # Calculate Prev/Next links
    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "current_chapter": current_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "prev_idx": prev_idx,
        "next_idx": next_idx
    })

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    img_path = os.path.join(BOOKS_DIR, safe_book_id, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)


# --- Translation API ---

@app.get("/api/translation/config")
async def get_translation_config():
    """Returns current translation configuration."""
    config = load_translation_config()
    return JSONResponse(config)


@app.post("/api/translation/config")
async def update_translation_config(update: TranslationConfigUpdate):
    """Update translation configuration."""
    config = load_translation_config()
    
    if update.prompt_template is not None:
        config["prompt_template"] = update.prompt_template
    if update.default_target_language is not None:
        config["default_target_language"] = update.default_target_language
    if update.ollama_model is not None:
        config["ollama_model"] = update.ollama_model
    if update.ollama_url is not None:
        config["ollama_url"] = update.ollama_url
    
    save_translation_config(config)
    return JSONResponse({"status": "ok"})


@app.get("/api/books/{book_id}/chapters/{chapter_index}/translation")
async def get_chapter_translation(book_id: str, chapter_index: int, tlang: str = None):
    """
    Get translation for a chapter.
    Returns: {status: missing|ready|error|translating, translations?: [...]}
    """
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    config = load_translation_config()
    if tlang is None:
        tlang = config.get("default_target_language", "en")
    
    model = config.get("ollama_model", "llama3.2")
    job_key = (book_id, chapter_index, tlang, model)
    
    # Check if job is in progress
    with _translation_lock:
        job_status = _translation_jobs.get(job_key)
    
    if job_status == "translating":
        return JSONResponse({"status": "translating"})
    elif job_status and job_status.startswith("error:"):
        error_msg = job_status[6:]
        return JSONResponse({"status": "error", "message": error_msg})
    
    # Check cache
    cached = load_cached_translation(book_id, chapter_index, tlang, model)
    if cached:
        # Validate cache by checking source hash
        chapter_html = book.spine[chapter_index].content
        blocks = segment_html_to_blocks(chapter_html)
        current_hash = compute_source_hash(blocks)
        
        if cached.get("source_hash") == current_hash:
            return JSONResponse({
                "status": "ready",
                "translations": cached["translations"]
            })
    
    return JSONResponse({"status": "missing"})


async def _translate_single_async(text: str, config: dict, target_language: str) -> str:
    """Run single translation in thread pool to avoid blocking."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, 
        ollama_translate_single, 
        text, 
        config, 
        target_language
    )


async def _stream_translations(book_id: str, chapter_index: int, tlang: str) -> AsyncGenerator[str, None]:
    """
    Generator that yields SSE events for each translated paragraph.
    Translates one paragraph at a time with 700ms pause between.
    """
    config = load_translation_config()
    model = config.get("ollama_model", "llama3.2")
    
    book = load_book_cached(book_id)
    if not book or chapter_index >= len(book.spine):
        yield f"data: {json.dumps({'type': 'error', 'message': 'Book or chapter not found'})}\n\n"
        return
    
    chapter_html = book.spine[chapter_index].content
    blocks = segment_html_to_blocks(chapter_html)
    total = len(blocks)
    
    if not blocks:
        # No translatable content - send completion
        source_hash = compute_source_hash(blocks)
        write_cached_translation(book_id, chapter_index, tlang, model, source_hash, [])
        yield f"data: {json.dumps({'type': 'complete', 'total': 0})}\n\n"
        return
    
    # Send initial info
    yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"
    
    translations = []
    
    for i, block in enumerate(blocks):
        try:
            # Translate single paragraph
            translated = await _translate_single_async(block, config, tlang)
            translations.append(translated)
            
            # Send this translation
            yield f"data: {json.dumps({'type': 'translation', 'index': i, 'text': translated, 'total': total})}\n\n"
            
            # Pause 700ms before next paragraph (except for last one)
            if i < total - 1:
                await asyncio.sleep(0.7)
                
        except Exception as e:
            error_msg = str(e)
            yield f"data: {json.dumps({'type': 'error', 'index': i, 'message': error_msg})}\n\n"
            # Continue with next paragraph despite error
            translations.append(f"[Error: {error_msg}]")
    
    # Cache all translations
    source_hash = compute_source_hash(blocks)
    write_cached_translation(book_id, chapter_index, tlang, model, source_hash, translations)
    
    # Send completion
    yield f"data: {json.dumps({'type': 'complete', 'total': total})}\n\n"


@app.get("/api/books/{book_id}/chapters/{chapter_index}/translation/stream")
async def stream_chapter_translation(book_id: str, chapter_index: int, tlang: str = None):
    """
    Stream translation for a chapter using Server-Sent Events.
    Translates one paragraph at a time with 700ms pause between.
    """
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    config = load_translation_config()
    if tlang is None:
        tlang = config.get("default_target_language", "en")
    
    return StreamingResponse(
        _stream_translations(book_id, chapter_index, tlang),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8123")
    uvicorn.run(app, host="127.0.0.1", port=8123)
