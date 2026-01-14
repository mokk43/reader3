"""
Translation helpers for reader3.

Handles:
- Segmentation of HTML into translatable blocks
- Caching translations to disk
- Ollama API communication
"""

import hashlib
import json
import os
import re
import urllib.request
import urllib.error
from typing import Optional

from bs4 import BeautifulSoup

# Default configuration
DEFAULT_CONFIG = {
    "ollama_url": os.environ.get("READER3_OLLAMA_URL", "http://127.0.0.1:11434"),
    "ollama_model": os.environ.get("READER3_OLLAMA_MODEL", "llama3.2"),
    "default_target_language": "en",
    "prompt_template": """Translate the following text blocks to {{TARGET_LANGUAGE}}.

IMPORTANT INSTRUCTIONS:
- Return ONLY a valid JSON array of translated strings
- The array must have exactly the same number of elements as the input
- Preserve paragraph structure and formatting
- Do not add explanations or notes

Input blocks:
{{BLOCKS_JSON}}

Output (JSON array of translated strings):"""
}

CONFIG_FILE = "translation_config.json"


def load_translation_config() -> dict:
    """Load translation config from disk, or return defaults."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            print(f"Error loading translation config: {e}")
    return DEFAULT_CONFIG.copy()


def save_translation_config(config: dict) -> None:
    """Save translation config to disk."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def segment_html_to_blocks(html: str) -> list[str]:
    """
    Extract translatable text blocks from HTML content.
    
    Tags included: p, h1, h2, h3, blockquote, li
    De-dup rule: Include li only if it contains no p descendants.
    Returns blocks in document order.
    """
    soup = BeautifulSoup(html, "html.parser")
    blocks = []
    seen_elements = set()
    
    # Tags we consider translatable
    translatable_tags = ["p", "h1", "h2", "h3", "blockquote", "li"]
    
    for tag in soup.find_all(translatable_tags):
        # Skip if we've already processed this element (nested case)
        if id(tag) in seen_elements:
            continue
        
        # For li: skip if it contains p descendants (avoid double-counting)
        if tag.name == "li":
            if tag.find("p"):
                continue
        
        # Get text content
        text = tag.get_text(strip=True)
        if text:
            blocks.append(text)
            seen_elements.add(id(tag))
    
    return blocks


def compute_source_hash(blocks: list[str]) -> str:
    """Compute SHA256 hash of source blocks for cache validation."""
    content = json.dumps(blocks, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in file paths."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model)


def translation_cache_path(book_id: str, chapter_index: int, tlang: str, model: str) -> str:
    """
    Get the path to the translation cache file.
    Path: {book_id}/translations/{tlang}/{model_sanitized}/chapter_{chapter_index}.json
    """
    safe_model = sanitize_model_name(model)
    return os.path.join(
        book_id,
        "translations",
        tlang,
        safe_model,
        f"chapter_{chapter_index}.json"
    )


def load_cached_translation(book_id: str, chapter_index: int, tlang: str, model: str) -> Optional[dict]:
    """
    Load cached translation if it exists.
    Returns dict with keys: source_hash, translations
    """
    path = translation_cache_path(book_id, chapter_index, tlang, model)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading translation cache: {e}")
    return None


def write_cached_translation(book_id: str, chapter_index: int, tlang: str, model: str, 
                              source_hash: str, translations: list[str]) -> None:
    """Write translation to cache."""
    path = translation_cache_path(book_id, chapter_index, tlang, model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    cache_data = {
        "source_hash": source_hash,
        "translations": translations,
        "version": "1"
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def build_prompt(blocks: list[str], target_language: str, template: str) -> str:
    """Build the translation prompt from template and blocks."""
    blocks_json = json.dumps(blocks, ensure_ascii=False, indent=2)
    
    prompt = template.replace("{{TARGET_LANGUAGE}}", target_language)
    prompt = prompt.replace("{{BLOCKS_JSON}}", blocks_json)
    
    # Ensure BLOCKS_JSON is in the prompt (append if missing)
    if "{{BLOCKS_JSON}}" not in template and blocks_json not in prompt:
        prompt += f"\n\nInput blocks:\n{blocks_json}\n\nOutput (JSON array):"
    
    return prompt


def extract_json_array(text: str) -> Optional[list[str]]:
    """Extract JSON array from Ollama response text."""
    # Try to parse the whole response as JSON first
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON array in the response
    # Look for patterns like [...] 
    matches = re.findall(r'\[[\s\S]*?\]', text)
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            continue
    
    return None


def build_single_prompt(text: str, target_language: str) -> str:
    """Build a simple prompt for translating a single paragraph."""
    return f"""Translate the following text to {target_language}.

IMPORTANT: Return ONLY the translated text, nothing else. No explanations, no quotes, no prefixes.

Text to translate:
{text}

Translation:"""


def ollama_translate_single(text: str, config: dict, target_language: str) -> str:
    """
    Translate a single text block using Ollama API.
    Returns the translated text directly.
    """
    if not text.strip():
        return ""
    
    ollama_url = config.get("ollama_url", DEFAULT_CONFIG["ollama_url"])
    model = config.get("ollama_model", DEFAULT_CONFIG["ollama_model"])
    
    prompt = build_single_prompt(text, target_language)
    
    url = f"{ollama_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "top_k": 20,
            "top_p": 0.6,
            "repetition_penalty": 1.05,
            "temperature": 0.7
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
            response_text = response_data.get("response", "").strip()
            return response_text
            
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to Ollama at {ollama_url}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from Ollama: {e}")


def ollama_translate_blocks(blocks: list[str], config: dict, target_language: str) -> list[str]:
    """
    Translate blocks using Ollama API.
    
    Chunks blocks to control prompt size and concatenates results.
    Uses stdlib urllib to avoid adding dependencies.
    """
    if not blocks:
        return []
    
    ollama_url = config.get("ollama_url", DEFAULT_CONFIG["ollama_url"])
    model = config.get("ollama_model", DEFAULT_CONFIG["ollama_model"])
    template = config.get("prompt_template", DEFAULT_CONFIG["prompt_template"])
    
    # Chunk size - translate in batches to avoid token limits
    CHUNK_SIZE = 20
    all_translations = []
    
    for i in range(0, len(blocks), CHUNK_SIZE):
        chunk = blocks[i:i + CHUNK_SIZE]
        prompt = build_prompt(chunk, target_language, template)
        
        # Call Ollama
        url = f"{ollama_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "top_k": 20,
                "top_p": 0.6,
                "repetition_penalty": 1.05,
                "temperature": 0.7
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
                response_text = response_data.get("response", "")
                
                # Parse the JSON array from response
                translations = extract_json_array(response_text)
                
                if translations is None:
                    raise ValueError(f"Could not parse JSON array from response: {response_text[:500]}")
                
                if len(translations) != len(chunk):
                    # Try to pad or trim to match
                    if len(translations) < len(chunk):
                        translations.extend(["[Translation missing]"] * (len(chunk) - len(translations)))
                    else:
                        translations = translations[:len(chunk)]
                
                all_translations.extend(translations)
                
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Ollama at {ollama_url}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from Ollama: {e}")
    
    return all_translations
