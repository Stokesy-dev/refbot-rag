# PRD: RefBot — RAG-based Football Rules Q&A

**Status**: `ready-for-agent`
**Date**: 2026-07-02

---

## Problem Statement

Football (soccer) rules are codified in the IFAB Laws of the Game — a ~150-page PDF covering 17 laws with deeply nested sub-sections. Finding the answer to a specific rules question requires manually searching through the document, cross-referencing laws, and interpreting dense legal-style language. There is no fast, reliable way to ask a natural-language question and get a cited, trustworthy answer grounded in the official text.

## Solution

Build **RefBot**, a retrieval-augmented generation (RAG) application that:

1. Ingests the IFAB Laws of the Game PDF and indexes it into a searchable vector store
2. Retrieves the most relevant passages for a user's natural-language question
3. Generates a concise, accurate answer using an LLM, citing the specific Law number and section
4. Refuses to answer if the retrieved context doesn't cover the question (no hallucination)
5. Provides a Streamlit-based chat UI with an expandable "Sources" panel showing exactly which passages were used
6. Supports two chunking strategies (fixed-size and structure-aware) so the developer can compare and document retrieval quality differences

## User Stories

1. As a football enthusiast, I want to ask a natural-language question about football rules, so that I don't have to manually search through the 150-page Laws of the Game PDF.
2. As a referee student, I want answers that cite the specific Law number and section, so that I can look up the original text for deeper study.
3. As a user, I want to see the exact passages the answer was derived from, so that I can verify the answer's correctness myself.
4. As a user, I want the system to explicitly tell me when a question isn't covered in the Laws of the Game, so that I'm never misled by a fabricated answer.
5. As a developer, I want two chunking strategies (fixed-size and structure-aware) available in the same codebase, so that I can compare retrieval quality and document trade-offs.
6. As a developer, I want to switch between chunking strategies from the Streamlit sidebar at runtime, so that I can interactively compare answers without restarting the app.
7. As a developer, I want a CLI script that builds FAISS indexes for both strategies in one command, so that setup is a single step after placing the PDF.
8. As a developer, I want chunk metadata (Law number, section title, page number) stored as a human-readable JSON sidecar alongside the FAISS index, so that I can inspect and debug retrieval results easily.
9. As a developer, I want a 10-question test harness that runs both strategies and outputs a comparison to a markdown file, so that I can paste results directly into documentation.
10. As a user, I want graceful error messages when the API key is missing or the Groq API hits rate limits, so that the app never crashes with a raw traceback.
11. As a user, I want each question to be independent (single-turn), so that answers are always grounded in freshly retrieved context rather than drifting from conversation history.
12. As a developer, I want all configuration (model names, chunk sizes, top-k) centralized in one config module, so that I can tune parameters without hunting through multiple files.
13. As a developer, I want the README to document what broke during development and how it was fixed, so that I have an honest learning record.
14. As a user, I want similarity scores displayed alongside each source chunk, so that I can gauge how confident the retrieval was.
15. As a developer, I want the project to use a simple `requirements.txt` + `venv` setup, so that onboarding is straightforward with no extra tooling.

## Implementation Decisions

### Module Architecture

Nine modules with clean, testable interfaces:

| Module | Responsibility | Interface |
|---|---|---|
| **Config** | Central constants, paths, env loading | Importable constants, `load_dotenv()` on import |
| **PDF Loader** | Extract text from PDF page-by-page | `load_pdf(path) → list[{page_number, text}]` |
| **Chunking Engine** | Split text into retrievable chunks with metadata | `chunk_fixed_size(pages) → list[Chunk]` and `chunk_by_structure(pages) → list[Chunk]` — same return type, switchable |
| **Embedding & Indexing** | Encode chunks, build/persist FAISS index | `build_index(chunks, dir)` and `load_index(dir) → (index, metadata)` |
| **Retriever** | Search FAISS, return top-k with scores and metadata | `retrieve(query, index, metadata, model, k) → list[Result]` |
| **Generator** | Call Groq with no-hallucination prompt | `generate_answer(query, chunks) → str` |
| **Index Builder** | CLI orchestration: PDF → chunks → indexes | Script, no importable interface |
| **Streamlit UI** | Chat input, answer display, sources panel, strategy toggle | Streamlit app entry point |
| **Evaluation Harness** | 10-question comparison across strategies | Script, outputs to console + markdown |

### Chunking Strategies

**Strategy A (Fixed-size):**
- Concatenate all pages, tokenize with `tiktoken` (`cl100k_base`), slide a 500-token window with 50-token overlap
- Assign Law/section metadata by scanning backwards for the most recent regex match (`Law \d+`, section headers)
- Trade-off: simple, uniform chunk sizes, but may split mid-sentence or merge content across Law boundaries

**Strategy B (Structure-aware):**
- Split at Law boundaries and sub-section boundaries using regex patterns on extracted text
- Each section becomes a chunk with its Law number and section title as metadata
- Sections exceeding ~256 tokens (embedding model's effective window) are sub-chunked at paragraph boundaries, preserving parent metadata
- Trade-off: respects document structure, but chunk sizes vary and very short sections may lack context

### Retrieval Design

- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers), 384-dimensional normalized vectors
- FAISS index type: `IndexFlatIP` (inner product on normalized vectors = cosine similarity)
- Top-k = 5, no hard similarity threshold — the generation prompt handles irrelevant context by instructing the model to say "Not covered"
- Metadata stored in `chunks_metadata.json` sidecar (list of dicts indexed to match FAISS vector IDs)

### Generation Design

- Model: `llama-3.1-8b-instant` on Groq (free tier, fast inference)
- Temperature: 0 (deterministic)
- System prompt enforces: answer only from context, cite Law/section, say "Not covered in the Laws of the Game" if context is insufficient
- Single-turn: no conversation history sent to the model
- API errors wrapped in user-friendly strings, never raised to the UI

### Secrets Management

- `.env` file loaded via `python-dotenv` at config import time
- `.env.example` committed with placeholder value
- `.env` added to `.gitignore`

### Data Layout

```
data/
├── laws_of_the_game.pdf
└── index/
    ├── fixed/    (index.faiss + chunks_metadata.json)
    └── semantic/ (index.faiss + chunks_metadata.json)
```

Both `data/index/` directories are gitignored — indexes are built locally via the CLI script.

## Testing Decisions

### What Makes a Good Test Here

For a RAG application, the primary measure of quality is **retrieval accuracy** (did the right chunks come back?) and **answer faithfulness** (did the model answer from context and cite correctly?). Unit-testing individual functions in isolation adds limited value when the real signal comes from end-to-end question→answer evaluation against known-good answers.

### Testing Approach

**10-question evaluation harness** (`test_questions.py`):
- Runs each question through the full pipeline (retrieve → generate) for both chunking strategies
- Outputs: answer text, retrieved chunks with Law/section labels and similarity scores
- Saves results to `test_results.md` for inclusion in the README
- Question mix: 7 factual (simple + nuanced), 2 edge cases, 1 deliberately out-of-scope

This is the primary testing mechanism. The developer manually reviews `test_results.md` to compare retrieval quality between strategies and verify citation accuracy.

**Manual verification checklist:**
- Inspect `chunks_metadata.json` to confirm Law/section labels are correct
- Launch Streamlit, test strategy switching, error states (missing key, missing index)
- Verify the out-of-scope question returns "Not covered in the Laws of the Game"

### Modules Worth Unit Testing (Future)

If the project grows beyond MVP, the highest-value modules to unit test would be:
- **Chunking Engine** — test that regex patterns correctly identify Law/section boundaries across edge cases in the PDF text
- **Generator** — test that the prompt construction correctly formats chunks and that error wrapping works

These are not in scope for the initial build.

## Out of Scope

- **Authentication / user accounts** — no login, no user management
- **Multi-turn conversation** — each question is independent; no follow-up context
- **Deployment** — runs locally only; no Docker, no cloud deployment
- **Multiple PDF support** — single PDF (IFAB Laws of the Game) only
- **Unit tests** — testing is done via the 10-question evaluation harness, not pytest
- **UI polish** — functional Streamlit UI; citation correctness is prioritized over visual design
- **Caching / performance optimization** — no query caching, no embedding caching
- **Alternative embedding models or LLMs** — fixed to `all-MiniLM-L6-v2` and `llama-3.1-8b-instant`
- **Reranking** — no cross-encoder reranking step; raw FAISS similarity is used directly
- **Hybrid search** — no BM25 or keyword search; vector-only retrieval

## Further Notes

- **Regex calibration required**: The structure-aware chunking regex patterns will need tuning once the actual PDF text is extracted. The first development step should be extracting a sample page and calibrating patterns before building the full pipeline.
- **Embedding model token limit**: `all-MiniLM-L6-v2` has a 256-token effective window. Chunks exceeding this will have truncated representations. The structure-aware strategy explicitly handles this; the fixed-size strategy's 500-token chunks will be partially truncated by the model but this is an accepted trade-off for the baseline comparison.
- **Groq free-tier rate limits**: The test harness runs 20 Groq calls (10 questions × 2 strategies). Rate limiting may require adding a small delay between calls.
- **README honesty**: The README will include a "What Broke & How It Was Fixed" section documenting real issues encountered during development. This is a deliberate requirement from the developer.
