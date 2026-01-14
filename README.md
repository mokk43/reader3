# reader 3

![reader3](reader3.png)

A lightweight, self-hosted EPUB reader that lets you read through EPUB books one chapter at a time. This makes it very easy to copy paste the contents of a chapter to an LLM, to read along. Basically - get epub books (e.g. [Project Gutenberg](https://www.gutenberg.org/) has many), open them up in this reader, copy paste text around to your favorite LLM, and read together and along.

This project was 90% vibe coded just to illustrate how one can very easily [read books together with LLMs](https://x.com/karpathy/status/1990577951671509438). I'm not going to support it in any way, it's provided here as is for other people's inspiration and I don't intend to improve it. Code is ephemeral now and libraries are over, ask your LLM to change it in whatever way you like.

## Usage

The project uses [uv](https://docs.astral.sh/uv/). So for example, download [Dracula EPUB3](https://www.gutenberg.org/ebooks/345) to this directory as `dracula.epub`, then:

```bash
uv run reader3.py dracula.epub
```

This creates the directory `dracula_data`, which registers the book to your local library. We can then run the server:

```bash
uv run server.py
```

And visit [localhost:8123](http://localhost:8123/) to see your current Library. You can easily add more books, or delete them from your library by deleting the folder. It's not supposed to be complicated or complex.

## Chapter Translation (via Ollama)

reader3 supports chapter-by-chapter translation using [Ollama](https://ollama.ai). Translations are cached to disk, so they only need to be generated once per chapter.

### Setup

1. Install and run Ollama locally:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ```

2. Pull a model (default is `llama3.2`):
   ```bash
   ollama pull llama3.2
   ```

### Using Translation

In the reader interface, you'll find translation controls in the sidebar:

- **Translation Mode**: Choose between:
  - **None** - Show original text only
  - **Bilingual** - Show original paragraphs followed by translations
  - **Translated Only** - Show only translated text

- **Target Language**: Enter the language you want to translate to (e.g., "English", "Spanish", "日本語")

- **⚙️ Translation Settings**: Configure:
  - Ollama URL (default: `http://127.0.0.1:11434`)
  - Model name (default: `llama3.2`)
  - Custom prompt template

### How It Works

1. When you switch to Bilingual or Translated mode, reader3 checks for cached translations
2. If no cache exists, it sends the chapter content to Ollama for translation
3. Translations are cached in `{book}_data/translations/{lang}/{model}/chapter_{n}.json`
4. Subsequent views load instantly from cache

### Environment Variables

- `READER3_OLLAMA_URL` - Override default Ollama URL
- `READER3_OLLAMA_MODEL` - Override default model name

### Cache Storage

Translation caches are stored per-book in the `*_data` folders:
```
{book}_data/
├── book.pkl
├── images/
└── translations/
    └── {target_language}/
        └── {model}/
            └── chapter_{index}.json
```

Delete a translation cache to regenerate it. Delete the entire `translations/` folder to clear all translations for a book.

## License

MIT