"""
EPUB Translation CLI - Translates an EPUB book using OpenAI or Ollama backends.
Outputs a new EPUB with translated content and TOC.
"""

import argparse
import copy
import json
import os
import re
import sys
import time
from typing import List, Dict, Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString


# --- Configuration ---

def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Load environment variables from a .env file (simple KEY=VALUE parsing)."""
    env_vars = {}
    if not os.path.exists(env_path):
        return env_vars
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value
    return env_vars


def load_config(env_path: str = ".env", json_path: str = "translation_config.json") -> Dict[str, Any]:
    """
    Load configuration with JSON as primary, .env as overrides.
    Returns a config dict with all settings.
    """
    config = {
        # Ollama defaults
        "ollama_url": "http://localhost:11434",
        "ollama_model": "",
        "ollama_prompt": "",
        "ollama_timeout": 120,
        # OpenAI defaults
        "openai_base_url": "https://api.openai.com",
        "openai_api_key": "",
        "openai_model": "gpt-4o-mini",
        "openai_prompt": "",
        "openai_temperature": 0.3,
        "openai_timeout": 120,
        # Common
        "target_language": "zh",
        "max_batch_chars": 2000,
        "prompt_template": "Translate the following text blocks into natural, idiomatic {{TARGET_LANGUAGE}} text.\n\nIMPORTANT INSTRUCTIONS:\n- Return ONLY a valid JSON array of translated strings\n- The array must have exactly the same number of elements as the input\n- Preserve paragraph structure and formatting\n- Do not add explanations or notes\n\nInput blocks:\n{{BLOCKS_JSON}}\n\nOutput (JSON array of translated strings):",
    }

    # Load JSON config (primary)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            json_config = json.load(f)
            if "ollama_url" in json_config:
                config["ollama_url"] = json_config["ollama_url"]
            if "ollama_model" in json_config:
                config["ollama_model"] = json_config["ollama_model"]
            if "default_target_language" in json_config:
                config["target_language"] = json_config["default_target_language"]
            if "prompt_template" in json_config:
                config["prompt_template"] = json_config["prompt_template"]

    # Load .env overrides
    env_vars = load_env_file(env_path)

    # OpenAI overrides
    if "OPENAI_API_KEY" in env_vars:
        config["openai_api_key"] = env_vars["OPENAI_API_KEY"]
    if "OPENAI_BASE_URL" in env_vars:
        config["openai_base_url"] = env_vars["OPENAI_BASE_URL"]
    if "OPENAI_MODEL" in env_vars:
        config["openai_model"] = env_vars["OPENAI_MODEL"]
    if "OPENAI_PROMPT" in env_vars:
        config["openai_prompt"] = env_vars["OPENAI_PROMPT"]
    if "OPENAI_TEMPERATURE" in env_vars:
        config["openai_temperature"] = float(env_vars["OPENAI_TEMPERATURE"])
    if "OPENAI_TIMEOUT" in env_vars:
        config["openai_timeout"] = int(env_vars["OPENAI_TIMEOUT"])

    # Ollama overrides
    if "OLLAMA_URL" in env_vars:
        config["ollama_url"] = env_vars["OLLAMA_URL"]
    if "OLLAMA_MODEL" in env_vars:
        config["ollama_model"] = env_vars["OLLAMA_MODEL"]
    if "OLLAMA_PROMPT" in env_vars:
        config["ollama_prompt"] = env_vars["OLLAMA_PROMPT"]
    if "OLLAMA_TIMEOUT" in env_vars:
        config["ollama_timeout"] = int(env_vars["OLLAMA_TIMEOUT"])

    # Common overrides
    if "TARGET_LANGUAGE" in env_vars:
        config["target_language"] = env_vars["TARGET_LANGUAGE"]
    if "MAX_BATCH_CHARS" in env_vars:
        config["max_batch_chars"] = int(env_vars["MAX_BATCH_CHARS"])

    return config


# --- Translation Backends ---

def extract_json_array(text: str) -> List[str]:
    """Extract a JSON array from potentially messy LLM output."""
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(item) for item in result]
    except json.JSONDecodeError:
        pass

    # Try to find array in the text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(item) for item in result]
        except json.JSONDecodeError:
            pass

    # Fallback: return original text as single-item list
    return [text]


def translate_with_openai(blocks: List[str], config: Dict[str, Any]) -> List[str]:
    """Translate blocks using OpenAI-compatible API."""
    prompt_template = config.get("openai_prompt") or config["prompt_template"]
    prompt = prompt_template.replace("{{BLOCKS_JSON}}", json.dumps(blocks, ensure_ascii=False))
    prompt = prompt.replace("{{TARGET_LANGUAGE}}", config["target_language"])
    if any(SEGMENT_BREAK in block for block in blocks):
        prompt += (
            "\n\nIMPORTANT:\n"
            f"- Preserve the token {SEGMENT_BREAK} exactly in the output strings.\n"
            "- Keep the same number of segments separated by this token.\n"
        )

    url = f"{config['openai_base_url'].rstrip('/')}/v1/chat/completions"
    payload = {
        "model": config["openai_model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config["openai_temperature"],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['openai_api_key']}",
    }

    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(req, timeout=config["openai_timeout"]) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return extract_json_array(content)
    except (URLError, HTTPError) as e:
        print(f"  [ERROR] OpenAI API error: {e}")
        return blocks  # Return original on error


def translate_with_ollama(blocks: List[str], config: Dict[str, Any]) -> List[str]:
    """Translate blocks using Ollama API."""
    prompt_template = config.get("ollama_prompt") or config["prompt_template"]
    prompt = prompt_template.replace("{{BLOCKS_JSON}}", json.dumps(blocks, ensure_ascii=False))
    prompt = prompt.replace("{{TARGET_LANGUAGE}}", config["target_language"])
    if any(SEGMENT_BREAK in block for block in blocks):
        prompt += (
            "\n\nIMPORTANT:\n"
            f"- Preserve the token {SEGMENT_BREAK} exactly in the output strings.\n"
            "- Keep the same number of segments separated by this token.\n"
        )

    url = f"{config['ollama_url'].rstrip('/')}/api/generate"
    payload = {
        "model": config["ollama_model"],
        "prompt": prompt,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(req, timeout=config["ollama_timeout"]) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("response", "")
            return extract_json_array(content)
    except (URLError, HTTPError) as e:
        print(f"  [ERROR] Ollama API error: {e}")
        return blocks  # Return original on error


def translate_blocks(blocks: List[str], backend: str, config: Dict[str, Any]) -> List[str]:
    """Translate a list of text blocks using the specified backend."""
    if not blocks:
        return []

    if backend == "openai":
        return translate_with_openai(blocks, config)
    elif backend == "ollama":
        return translate_with_ollama(blocks, config)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def batch_translate(blocks: List[str], backend: str, config: Dict[str, Any]) -> List[str]:
    """
    Translate blocks one-by-one (single request per block).
    Returns translated blocks in the same order.
    """
    if not blocks:
        return []

    results = []
    total = len(blocks)
    for idx, block in enumerate(blocks, start=1):
        print(f"    [Paragraph {idx}/{total}] Translating...")
        translated = translate_blocks([block], backend, config)
        if len(translated) == 1:
            results.append(translated[0])
        else:
            print("  [WARN] Single-block translation mismatch; using original text")
            results.append(block)
        if idx % 10 == 0 and idx < total:
            time.sleep(1)

    return results


# --- HTML Processing ---

BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "td", "th", "div"]
SEGMENT_BREAK = "<<<SEGMENT_BREAK>>>"


def get_leaf_block_elements(soup: BeautifulSoup):
    """Return leaf block elements in document order."""
    elements = []
    for elem in soup.find_all(BLOCK_TAGS):
        if elem.find(BLOCK_TAGS):
            continue
        if elem.get("class") and "translated" in elem.get("class", []):
            continue
        if not elem.get_text(strip=True):
            continue
        elements.append(elem)
    return elements


def get_node_infos(element):
    """Collect non-empty text nodes with leading/trailing whitespace preserved."""
    node_infos = []
    for node in element.descendants:
        if isinstance(node, NavigableString):
            raw = str(node)
            if raw.strip() == "":
                continue
            leading_len = len(raw) - len(raw.lstrip())
            trailing_len = len(raw) - len(raw.rstrip())
            leading = raw[:leading_len]
            trailing = raw[len(raw) - trailing_len:] if trailing_len > 0 else ""
            core = raw.strip()
            node_infos.append(
                {"node": node, "leading": leading, "trailing": trailing, "core": core}
            )
    return node_infos


def build_block(node_infos) -> str:
    """Join core text segments for translation using a stable delimiter."""
    return SEGMENT_BREAK.join(info["core"] for info in node_infos)


def apply_translated_segments(node_infos, segments) -> bool:
    """Apply translated segments to text nodes. Returns True if applied."""
    if len(node_infos) != len(segments):
        return False
    for info, seg in zip(node_infos, segments):
        new_text = f"{info['leading']}{seg}{info['trailing']}"
        info["node"].replace_with(NavigableString(new_text))
    return True


def clone_element(elem):
    """Deep-clone a tag to preserve inline markup for bilingual output."""
    return copy.deepcopy(elem)


def translate_html_content(html: str, mode: str, backend: str, config: Dict[str, Any]) -> str:
    """
    Translate HTML content.
    mode: 'translation-only' or 'bilingual'
    Returns translated HTML.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find leaf block elements in document order
    elements = get_leaf_block_elements(soup)
    element_infos = []

    for elem in elements:
        node_infos = get_node_infos(elem)
        if not node_infos:
            continue
        element_infos.append(
            {"elem": elem, "node_infos": node_infos, "block": build_block(node_infos)}
        )

    if not element_infos:
        # No translatable content; return original HTML to preserve SVG/XML attributes
        return html

    # Translate all blocks
    blocks = [info["block"] for info in element_infos]
    translated_blocks = batch_translate(blocks, backend, config)

    # Apply translations
    for idx, info in enumerate(element_infos):
        if idx >= len(translated_blocks):
            break
        translated_block = translated_blocks[idx]
        expected_parts = len(info["node_infos"])

        if expected_parts == 1:
            segments = [translated_block]
        else:
            segments = translated_block.split(SEGMENT_BREAK)
            if len(segments) != expected_parts:
                segments = []

        # Fallback: translate each text node individually
        if not segments:
            cores = [n["core"] for n in info["node_infos"]]
            node_translations = batch_translate(cores, backend, config)
            if len(node_translations) == len(cores):
                segments = node_translations
            else:
                segments = cores

        if mode == "translation-only":
            apply_translated_segments(info["node_infos"], segments)
        else:  # bilingual
            translated_elem = clone_element(info["elem"])
            translated_infos = get_node_infos(translated_elem)
            if len(translated_infos) != len(segments):
                clone_cores = [n["core"] for n in translated_infos]
                node_translations = batch_translate(clone_cores, backend, config)
                if len(node_translations) == len(clone_cores):
                    segments = node_translations
                else:
                    segments = clone_cores
            apply_translated_segments(translated_infos, segments)
            classes = list(translated_elem.get("class", []))
            if "translated" not in classes:
                classes.append("translated")
            if classes:
                translated_elem["class"] = classes
            info["elem"].insert_after(translated_elem)

    return str(soup)


# --- TOC Processing ---

def translate_toc_titles(book: epub.EpubBook, mode: str, backend: str, config: Dict[str, Any]):
    """
    Translate TOC entry titles in the book.
    Modifies the book's toc in place.
    """
    def collect_titles(toc_items) -> List[str]:
        """Recursively collect all TOC titles."""
        titles = []
        for item in toc_items:
            if isinstance(item, tuple):
                section, children = item
                if hasattr(section, 'title') and section.title:
                    titles.append(section.title)
                titles.extend(collect_titles(children))
            elif hasattr(item, 'title') and item.title:
                titles.append(item.title)
        return titles

    def apply_translations(toc_items, translations: Dict[str, str], mode: str):
        """Recursively apply translations to TOC items."""
        for item in toc_items:
            if isinstance(item, tuple):
                section, children = item
                if hasattr(section, 'title') and section.title:
                    original = section.title
                    if original in translations:
                        if mode == "translation-only":
                            section.title = translations[original]
                        else:  # bilingual
                            section.title = f"{original} / {translations[original]}"
                apply_translations(children, translations, mode)
            elif hasattr(item, 'title') and item.title:
                original = item.title
                if original in translations:
                    if mode == "translation-only":
                        item.title = translations[original]
                    else:  # bilingual
                        item.title = f"{original} / {translations[original]}"

    # Collect all titles (preserve duplicates but translate unique once)
    titles = collect_titles(book.toc)
    if not titles:
        return

    unique_titles = list(dict.fromkeys(titles))
    print(f"Translating {len(unique_titles)} unique TOC entries...")

    # Translate unique titles
    translated = batch_translate(unique_titles, backend, config)

    # Build translation map (reuse translations for duplicates)
    translation_map = {}
    for i, title in enumerate(unique_titles):
        if i < len(translated):
            translation_map[title] = translated[i]

    # Apply translations
    apply_translations(book.toc, translation_map, mode)


# --- Main Translation Pipeline ---

def translate_epub(epub_path: str, mode: str, backend: str, config: Dict[str, Any]) -> str:
    """
    Main translation pipeline.
    Returns the path to the translated EPUB.
    """
    print(f"Loading EPUB: {epub_path}")
    book = epub.read_epub(epub_path)

    # Get all document items
    doc_items = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
    total_docs = len(doc_items)

    print(f"Found {total_docs} document items to translate")
    print(f"Mode: {mode}, Backend: {backend}, Target: {config['target_language']}")
    print("-" * 50)

    # Translate each document
    for idx, item in enumerate(doc_items):
        name = item.get_name()
        print(f"[{idx + 1}/{total_docs}] Translating: {name}")

        try:
            content = item.get_content().decode("utf-8", errors="ignore")
            translated_content = translate_html_content(content, mode, backend, config)
            item.set_content(translated_content.encode("utf-8"))
        except Exception as e:
            print(f"  [ERROR] Failed to translate {name}: {e}")

    # Translate TOC
    print("-" * 50)
    translate_toc_titles(book, mode, backend, config)

    # Generate output path
    base, ext = os.path.splitext(epub_path)
    output_path = f"{base}-translated{ext}"

    # Write output
    print("-" * 50)
    print(f"Writing translated EPUB: {output_path}")
    epub.write_epub(output_path, book)

    return output_path


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Translate an EPUB book using OpenAI or Ollama backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run translate_epub.py book.epub --mode bilingual --backend ollama
  uv run translate_epub.py book.epub --mode translation-only --backend openai
        """
    )
    parser.add_argument("epub_path", help="Path to the EPUB file to translate")
    parser.add_argument(
        "--mode",
        choices=["translation-only", "bilingual"],
        default="bilingual",
        help="Translation mode: 'translation-only' replaces text, 'bilingual' adds translated text after original (default: bilingual)"
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "ollama"],
        default="ollama",
        help="Translation backend to use (default: ollama)"
    )
    parser.add_argument(
        "--target-language",
        help="Target language for translation (overrides config)"
    )
    parser.add_argument(
        "--config",
        default="translation_config.json",
        help="Path to JSON config file (default: translation_config.json)"
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file (default: .env)"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.epub_path):
        print(f"Error: File not found: {args.epub_path}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.env, args.config)

    # Override target language if specified
    if args.target_language:
        config["target_language"] = args.target_language

    # Validate backend requirements
    if args.backend == "openai" and not config["openai_api_key"]:
        print("Error: OpenAI API key not configured. Set OPENAI_API_KEY in .env")
        sys.exit(1)
    if args.backend == "ollama" and not config["ollama_model"]:
        print("Error: Ollama model not configured. Set ollama_model in config or OLLAMA_MODEL in .env")
        sys.exit(1)

    # Run translation
    try:
        output_path = translate_epub(args.epub_path, args.mode, args.backend, config)
        print("-" * 50)
        print(f"Translation complete! Output: {output_path}")
    except Exception as e:
        print(f"Error during translation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
