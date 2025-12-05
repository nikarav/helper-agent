# Data Preparation

This document describes how we prepare documentation for the vector store.

## Sources

We use the official `llms-full.txt` files.

| Source | URL |
|--------|-----|
| LangGraph | https://langchain-ai.github.io/langgraph/llms-full.txt|
| LangChain | https://docs.langchain.com/llms-full.txt  |

Raw files are stored in `data/`:
- `data/langgraph-llms-full.txt`
- `data/langchain-llms-full.txt`

## Filtering

The LangChain file contains ~575 docs across multiple languages and platforms. We filter to keep only what's relevant.

**What we keep:**

| Category | URL Pattern | Reason |
|----------|-------------|--------|
| LangChain Python | `/oss/python/langchain/` | Core library docs |
| Integrations | `/oss/python/integrations/` | Model providers, vector stores, tools |
| Migration Guides | `/oss/python/migrate/` | Useful for version upgrades |
| Release Notes | `/oss/python/releases/` | New features and changes |

**What we exclude:**

| Category | URL Pattern | Reason |
|----------|-------------|--------|
| JavaScript/TypeScript | `/oss/javascript/` | Python-only scope |
| LangSmith | `/langsmith/` | Observability platform |
| LangGraph (from LangChain) | `/oss/python/langgraph/` | Already covered by dedicated LangGraph file |
| Deep Agents | `/oss/python/deepagents/` | Experimental |
| Contributing | `/oss/python/contributing/` | Not end-user docs |

Filtering rules are defined in `configs/data_processing.yaml`.

## Processing Pipeline

**Step 0: Perform Initial Analysis** (`notebooks/`)
- At first we ran an analysis to understand the datasets and their formats
- Using this analysis we identify which ones should be included in the final dataset.
- Afterwards, on the remaining filter dataset we try to analyze for possible chunking methods
- This is very useful, since it provided with insights related to how many `h2` and `h3` are contained, how many code blocks exist, etc.

**Step 1: Parse and Filter** (`scripts/data/process_docs.py`)
- Parses both file formats (LangGraph uses `---` delimiters, LangChain uses `Source:` headers)
- Applies category-based filtering to LangChain docs
- Outputs to `data/processed/filtered_docs_new.json`

**Step 2: Chunk and Embed** (`scripts/embed/build_vectordb.py`)
- Cleans documents (removes navigation artifacts, normalizes whitespace)
- Splits using three-stage markdown-aware chunking:
  - Stage 1: Split at `##` and `###` headers to preserve section boundaries
  - Stage 2: Greedy merge adjacent small sections within the same `h2` parent. (Optional step)
  - Stage 3: Apply size limits with code block protection
- Desired Chunk size (MAX): **1400 tokens**, overlap: **200 tokens**
- Embeds with Gemini's `gemini-embedding-001` (512 dimensions). We use 512 dims to keep the db small and according to MTEB the quality is still very high for 512. We could try other dims, maybe go even lower.
- Stores in ChromaDB at `data/vectordb/`

Final output: v2 has ~**732 chunks** from 116 documents, while v1 has ~**1070 chunks**. 

## Chunking Strategy

We use hierarchical markdown splitting with optional greedy merging:

1. **First pass** — split at `##` and `###` headers to preserve section boundaries
2. **Second pass** — (optional) greedy merge adjacent small sections *only if* they share the same `h2` parent and combined size ≤ chunk_size
3. **Third pass** — split large sections by size while protecting code blocks

Settings in `configs/embedding.yaml`:
```yaml
chunking:
  chunk_size: 1400      # tokens per chunk
  chunk_overlap: 200    # overlap between chunks
  greedy_merge: true    # merge adjacent small sections (set false for v1 behavior)
```

**Token counting:** We use `tiktoken` (OpenAI's tokenizer, `cl100k_base` encoding) for local token counting instead of calling the Gemini API. This avoids thousands of API calls during chunking — the recursive splitter calls the length function many times per document.

### Why these settings?

**Chunk size (1400 tokens):** We analyzed the corpus and found that only 47% of documents fit within 2K tokens — the rest need splitting. The 1400-token target keeps most how-to sections intact while ensuring large documents get properly chunked.

**Header-based splitting:** 94% of documents have H2 headers (average 4.1 per doc), making them reliable semantic boundaries. Splitting at headers preserves topical coherence better than arbitrary character splits.

**Greedy merge:** Header-only splitting creates many small chunks (1070 for our corpus). The greedy merge step combines adjacent sections that share the same `h2` parent header, reducing chunk count by ~32% (1070 → 732) while preserving topical coherence—sections about different topics are never merged together.

**Code block protection:** 70% of documents contain code (avg 23% code ratio). Some docs are >50% code. The markdown-aware splitter avoids breaking code blocks mid-function.

**Overlap (200 tokens):** Ensures questions spanning section boundaries still retrieve relevant context.

## Corpus Stats

| Source | Documents |
|--------|-----------|
| LangGraph | ~59 |
| LangChain (filtered) | ~57 |
| **Total** | ~116 |

After chunking: **732** chunks in the vector store.

### Database Versions

| Version | Chunks | Collection Name | Strategy | Status |
|---------|--------|-----------------|----------|--------|
| **v1 (default)** | 1070 | `langgraph_docs` | Header-only splitting | Shipping (better accuracy) |
| v2 | 732 | `documentation` | Header splitting + greedy merge | Experimental – needs further testing |

## Extending the Database

### Adding a new llms.txt source

1. Add the source to `configs/data_processing.yaml`:

```yaml
sources:
  - path: data/new-source-llms.txt
    format: langchain  # or langgraph
    name: new_source
    filter:
      include_categories:
        - relevant/category
```

2. Download the file:
```bash
curl -o data/new-source-llms.txt https://example.com/llms-full.txt
```

3. Re-run the pipeline:
```bash
python scripts/data/process_docs.py
python scripts/embed/build_vectordb.py --reset
```

### Adding custom documents

For non-llms.txt sources, you can:

1. Convert your docs to the JSON format used by `filtered_docs_new.json`:
```json
{
  "title": "Document Title",
  "content": "Full markdown content...",
  "source_file": "your-source",
  "category": "custom"
}
```

2. Append to the existing JSON file or create a separate one
3. Run the embedding script

### Supported formats

| Format | Delimiter | Example |
|--------|-----------|---------|
| `langgraph` | `---\npath.md\n---\n# Title\ncontent` | LangGraph docs |
| `langchain` | `# Title\nSource: url\n\ncontent` | LangChain docs |

New parsers can be added in `helper_agent/data/parsers.py`.

## Refreshing the Data

When upstream docs change:

```bash
# 1. Download fresh sources
curl -o data/langgraph-llms-full.txt https://langchain-ai.github.io/langgraph/llms-full.txt
curl -o data/langchain-llms-full.txt https://docs.langchain.com/llms-full.txt

# 2. Process
python scripts/data/process_docs.py

# 3. Rebuild (--reset clears existing vectors)
python scripts/embed/build_vectordb.py --reset
```

The `--reset` flag is important — it clears the existing collection before adding new vectors. Without it, you'd accumulate duplicates.
