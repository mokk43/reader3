# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview
reader3 is a lightweight, self-hosted EPUB reader designed for reading books chapter-by-chapter with LLMs. It extracts EPUB content into a structured format, serves it via a FastAPI web server, and provides a clean reading interface optimized for copying text to LLMs.

## Development Philosophy
This project is intentionally minimal and ephemeral. The author designed it as a "vibe coded" illustration and does not intend to maintain or improve it. Code changes should respect this simplicity—avoid over-engineering, complex dependencies, or architectural changes that increase complexity.

## Commands

### Processing EPUB files
```bash
uv run reader3.py <epub_file>
```
This parses an EPUB file and creates a `<book_name>_data/` directory containing:
- `book.pkl` - Serialized Book object with metadata, spine, TOC, and images
- `images/` - Extracted and sanitized image files

### Running the web server
```bash
uv run server.py
```
Starts the FastAPI server at `http://localhost:8123`. The server:
- Scans for all `*_data` directories in the current folder
- Shows a library view at `/`
- Provides reader interface at `/read/{book_id}/{chapter_index}`

### Python environment
Python version is locked to 3.10 (see `.python-version`). The project uses [uv](https://docs.astral.sh/uv/) for dependency management. All dependencies are defined in `pyproject.toml`.

## Architecture

### Core Data Structures
The project uses dataclasses defined in `reader3.py`:

**Book** - Top-level container (pickled to disk)
- `metadata: BookMetadata` - Title, authors, language, etc.
- `spine: List[ChapterContent]` - Linear reading order of physical EPUB files
- `toc: List[TOCEntry]` - Navigation tree (may be hierarchical)
- `images: Dict[str, str]` - Maps original EPUB paths to local `images/` paths
- `source_file`, `processed_at`, `version` - Metadata

**ChapterContent** - Represents one physical file from EPUB spine
- `href` - Original filename (e.g., `text/chapter01.html`)
- `content` - Cleaned HTML with rewritten image paths
- `text` - Plain text extraction for search/LLM usage
- `order` - Zero-indexed position in linear reading order

**TOCEntry** - One node in the table of contents
- `title` - Display name
- `href` - Original reference (may include `#anchor`)
- `file_href` - Just the filename part
- `anchor` - Fragment identifier if present
- `children` - Recursive list for nested TOC

### Processing Pipeline (reader3.py)
1. Load EPUB using `ebooklib`
2. Extract metadata from Dublin Core fields
3. Extract and sanitize images to `images/` directory
4. Parse TOC structure recursively (or build fallback from spine)
5. Process each spine item:
   - Decode content as UTF-8
   - Parse with BeautifulSoup
   - Rewrite image `src` attributes to point to local `images/`
   - Remove scripts, styles, forms, and other dangerous tags
   - Extract plain text for LLM context
6. Serialize entire Book object to `book.pkl`

### Web Server (server.py)
FastAPI application with three main routes:

**`GET /`** - Library view
- Scans for `*_data` folders in current directory
- Loads each `book.pkl` to display metadata
- Uses `lru_cache` to avoid repeated disk reads

**`GET /read/{book_id}/{chapter_index}`** - Reader interface
- Loads Book object from cache
- Serves chapter content by spine index (0-based)
- Renders Jinja2 template with TOC sidebar and navigation
- JavaScript maps TOC filenames to spine indices for navigation

**`GET /read/{book_id}/images/{image_name}`** - Image serving
- Serves images from `{book_id}/images/` directory
- Sanitizes paths to prevent directory traversal

### Template Architecture
Located in `templates/`:

**library.html** - Grid of book cards with basic styling

**reader.html** - Two-column layout:
- Left sidebar: TOC tree with recursive Jinja2 macro
- Right: Chapter content with Previous/Next navigation
- JavaScript function `findAndGo()` maps TOC entries to spine indices
- Active chapter highlighting based on filename matching

## Key Design Decisions

### Spine vs TOC
The EPUB spec separates the linear reading order (spine) from the navigation structure (TOC). This project:
- Uses spine indices (`0`, `1`, `2`...) for URL routing and navigation
- Displays the TOC tree for structure
- Maps between them via `href` matching in JavaScript

### Image Path Rewriting
EPUB images use relative paths that vary by structure. The processor:
- Extracts all images to flat `images/` directory
- Sanitizes filenames (removes special chars)
- Builds a map with both full path and basename as keys
- Rewrites `<img src>` attributes during HTML processing
- Serves images via special route that includes `book_id`

### HTML Cleaning
The `clean_html_content()` function removes potentially dangerous or useless elements:
- Scripts, styles, iframes, videos
- Forms, buttons, inputs
- HTML comments
- Preserves semantic HTML and formatting

### Caching Strategy
`load_book_cached()` uses `@lru_cache(maxsize=10)` to keep recently accessed books in memory, avoiding repeated pickle deserialization.

## File Structure
```
reader3/
├── reader3.py          # EPUB processing and data structures
├── server.py           # FastAPI web server
├── pyproject.toml      # Dependencies (ebooklib, beautifulsoup4, fastapi, uvicorn, jinja2)
├── .python-version     # 3.10
├── templates/
│   ├── library.html    # Book grid view
│   └── reader.html     # Two-column reader interface
└── *_data/            # Generated book directories (gitignored)
    ├── book.pkl       # Serialized Book object
    └── images/        # Extracted images
```

## Working with EPUBs
- Download from sources like [Project Gutenberg](https://www.gutenberg.org/)
- Process with `uv run reader3.py <file.epub>`
- This creates `<name>_data/` directory
- Delete the `*_data` folder to remove a book from the library
- The server automatically detects all `*_data` directories on startup

## Debugging Tips
- If TOC is empty, the code builds a fallback from spine filenames
- If images don't display, check the `images/` directory and the image map in the pickle
- If TOC navigation doesn't work, check JavaScript console for `spineMap` mismatches
- The server redacts sensitive data in output (shows `*********` for localhost)
