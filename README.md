# LangGraph Helper Agent

A Python agent that answers questions about LangGraph and LangChain, built with LangGraph itself. The agent operates in two modes: **offline** (RAG with self-correction against a local vector store) and **online** (ReAct pattern with web search). Both modes use Google Gemini as the LLM backend.

The offline mode implements a retrieve-generate-evaluate loop that reformulates queries when answers aren't well-grounded in the retrieved context. The online mode extends this with Tavily web search for real-time information.

## Quick Start

First, you need to clone this repo:
```bash
git clone https://github.com/nikarav/helper-agent.git
```

### 1, Git LFS (Required for Vector Store)

The pre-built vector store (data/vectordb/) is essential for the Offline Mode and is tracked using Git LFS due to its size. This step is mandatory.
 After cloning, you need to fetch the necessary files:

```bash
git lfs install
git lfs fetch --all
git lfs pull
```

If you don't have Git LFS installed, follow the instructions [here](https://git-lfs.com/).

### 2. Configure Your Python Environment
We recommend creating a clean environment with Python 3.11+. The following instructions require `conda`, which can be installed following instructions [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). 
```bash
conda create -n helper-agent python=3.11
conda activate helper-agent
```

At this point, you should be in a shell with python installed. Then you need to install dependencies with:

```bash
pip install -e .
```

### 4. API Key Configuration
The agent requires two API keys to function in its full capacity

#### Google Gemini API Key
The agent uses Gemini as its backend. Get a free-tier key from [Google AI Studio](https://aistudio.google.com/app/api-keys) and set it as an environment variable:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

#### Tavily API Key (For Online Mode)
The Online Mode utilizes Tavily for web search. Sign up for a free key on their [website](https://app.tavily.com/home):

```bash
export TAVILY_API_KEY="your-tavily-api-key"
```

## How to Run the Agent
Once all environment variables are set, you can run the agent in several ways.

### Normal mode
You can start the agent and use the flag `--mode` to specify which mode the agent will run.

```bash
python scripts/run_agent.py --mode offline "How do I add persistence to a LangGraph agent?"
python scripts/run_agent.py --mode online "What are the latest LangGraph features?"
```

You can also set the mode via environment variable:
```bash
export AGENT_MODE=online
python scripts/run_agent.py "Your question here"
```


### Interactive mode
There is also the option to have an interactive shell, where you are asked a question and as soon as the answer is provided you remain in this mode and are prompted to provide with the next question. 
```bash
python scripts/run_agent.py --mode offline -i
```
>Note: This is a Q&A app. Each time you provide with a question, it is considered as a fresh start. The context of the previous iterations is not saved.

### Docker

Build the image:
```bash
docker build -t helper-agent .
```

Run the agent:
```bash
# Offline mode
docker run --rm -e GEMINI_API_KEY=$GEMINI_API_KEY helper-agent --mode offline "How do I add persistence?"

# Online mode
docker run --rm -e GEMINI_API_KEY=$GEMINI_API_KEY -e TAVILY_API_KEY=$TAVILY_API_KEY helper-agent --mode online "What's new in LangGraph?"

# Interactive mode (requires -it flags)
docker run --rm -it -e GEMINI_API_KEY=$GEMINI_API_KEY helper-agent --mode offline -i
```

## Operating Modes

### Offline Mode

Uses a local vector store built from LangGraph/LangChain documentation. No internet required (except for LLM API calls).

The agent implements a RAG pipeline with self-correction:
1. Retrieve relevant docs from vector store
2. Generate answer from retrieved context
3. Evaluate if the answer is grounded in the docs
4. If not confident, reformulate the query and retry (up to 2 times)

```
START → Retrieve → Generate → Evaluate → [Confident?]
                                         ├─ Yes/Max retries → END
                                         └─ No → Reformulate → Retrieve (loop)
```

**This architecture optimizes offline mode's constraint (limited data) through
intelligent retry with query reformulation.**

### Online Mode

Extends offline capabilities with web search via Tavily. Useful for recent updates or topics not covered in the local docs.

The agent uses a ReAct pattern: it reasons about the question, decides whether to search the local docs or the web, and iterates until it has enough information to answer.

```
START → Agent (LLM) → [Tool calls?]
                      ├─ No → END (final answer)
                      └─ Yes → Execute Tools → Agent (loop)

Tools: search_documentation (vector store), web_search (Tavily)
```

## Architecture

Built with LangGraph. Both modes are implemented as state graphs with different structures:

| Aspect | Offline | Online |
|--------|---------|--------|
| Pattern | RAG + self-correction | ReAct (reason + act) |
| Tools | None (retriever is a node) | `search_documentation`, `web_search` |
| Loop trigger | Failed evaluation | Tool calls in LLM response |
| Termination | Confident answer or max retries | No more tool calls |


## Data Strategy

### Sources

The vector store is built from the official `llms-full.txt` files:
- **LangGraph**: https://langchain-ai.github.io/langgraph/llms-full.txt (all docs)
- **LangChain**: https://docs.langchain.com/llms-full.txt (filtered to Python-only)

From LangChain, we keep: core library docs, integrations (model providers, vector stores), migration guides, and release notes. We exclude JavaScript docs, LangSmith, and experimental frameworks. We exclude also most langgraph realated docs, since we have the `llms-full.txt` of langgraph.

Why filter? The raw LangChain file has ~575 docs across multiple languages. We keep support for python for now. We can extend this in the future.

### Updating the Data

The vector store ships pre-built, but you can refresh it when the upstream documentation changes.

**Step 1 — Download fresh docs:**
```bash
curl -o data/langgraph-llms-full.txt https://langchain-ai.github.io/langgraph/llms-full.txt
curl -o data/langchain-llms-full.txt https://docs.langchain.com/llms-full.txt
```

**Step 2 — Process and filter:**
```bash
python scripts/data/process_docs.py
```
This script parses both `llms-full.txt` files, applies category-based filtering (configured in `configs/data_processing.yaml`), and outputs the cleaned corpus to `data/processed/filtered_docs_new.json`.

**Step 3 — Rebuild vector store:**
```bash
python scripts/embed/build_vectordb.py --reset
```
This script chunks the documents using markdown-aware splitting (~**1400** tokens per chunk with **200** token overlap), embeds them with Gemini's `gemini-embedding-001` model, and stores the vectors in ChromaDB.

**Why these settings?** We analyzed the corpus (~116 docs, ~440K tokens) and found that 94% of documents have H2 headers (avg 4.1 per doc), making them reliable split points. The 1400-token chunk size fits most how-to sections while keeping context coherent. Around 70% of docs contain code blocks, so we use markdown-aware splitting to avoid breaking code mid-block.

### Vector Database Versions

The repository includes two versions of the pre-built vector database:

| Version | Chunks | Collection Name | Chunking Strategy | Status |
|---------|--------|-----------------|-------------------|--------|
| **v1 (default)** | 1070 | `langgraph_docs` | Header-only splitting (split at every `##`/`###`) | Shipping — better grounded answers |
| v2 | 732 | `documentation` | Header splitting + greedy merge | Experimental — needs more regression tests |

> **Note:** v1 is selected by default. The newer version `v2` introduces a greedy merge step that combines adjacent small sections *only if* they share the same `h2` parent header and fit within the chunk size limit. This reduces chunk count by ~32% while preserving topical coherence—sections about different topics are never merged together.

Toggle via `configs/embedding.yaml`:
```yaml
chunking:
  greedy_merge: false  # v1 behavior (default). Set true to try the v2 build.
```

### Extending with New Sources

We can add new documentation sources by editing `configs/data_processing.yaml`. The pipeline supports two formats:

| Format | Structure | Example |
|--------|-----------|---------|
| `langgraph` | `---\npath.md\n---\n# Title\ncontent` | LangGraph docs |
| `langchain` | `# Title\nSource: url\n\ncontent` | LangChain docs |

To add a new source, append an entry to the `sources` list with filtering rules if needed. Then re-run the processing and embedding scripts.

See [docs/data_preparation.md](docs/data_preparation.md) for detailed filtering rationale and chunking analysis.

## Project Structure

```
helper_agent/
├── agent/
│   ├── offline/          # RAG agent with self-correction
│   │   ├── graph.py      # Graph builder and runner
│   │   ├── nodes.py      # retrieve, generate, evaluate, reformulate
│   │   └── state.py      # OfflineAgentState
│   └── online/           # ReAct agent with tools
│       ├── graph.py
│       ├── tools.py      # search_documentation, web_search
│       └── state.py      # OnlineAgentState
├── data/                 # Parsing, chunking, filtering
├── vectorstore/          # ChromaDB wrapper, embeddings
└── utilities/            # Config loading, logging, rate limiting

scripts/
├── run_agent.py          # Main entry point
├── data/process_docs.py  # Document processing
└── embed/build_vectordb.py

configs/
├── agent.yaml            # LLM settings, retrieval params
├── data_processing.yaml  # Source files, filtering rules
└── embedding.yaml        # Chunking, embedding model
```


## Requirements

- Python 3.11+
- Dependencies in `pyproject.toml`

## Future Work

### Observability
Integrate Langfuse first things first.

### Retrieval Enhancements
- Retrieval Evaluation. Add some metrics for evaluation.
- Different Chunking Methods. Currently, chunking takes place thanks to langchain. We need to verify that our chunking is working correctly and try other methods as well. Maybe try regex-based chunking, though this increases the complexity and potential bugs.
- Hybrid Search (Vector + BM25). Add BM25 keyword matching alongside vector similarity.
- Reranking. Add cross-encoder reranking of top-K results.

### Agent Enhancements
- Streaming Responses. Stream LLM output token-by-token to user
- Potential other architectures. Maybe one agent with subgraphs.
- Chat-based Q&A app. This is hard to implement, due to rate limits, but still consists an interesting future work of this project, i.e. be able to have back and fourth conversation with the llm. 


