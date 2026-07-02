# RefBot — RAG-based Football Rules Q&A

RefBot is a Retrieval-Augmented Generation (RAG) application that answers natural-language questions about football rules using the official **IFAB Laws of the Game 2024/25** as its knowledge base. It uses page-by-page PDF extraction, local vector indexes, and Groq's Llama 3.1 8B model to generate precise, grounded answers that explicitly cite the specific Law number, section, and page number from the official handbook.

---

## Architecture

The system follows a standard modular RAG pipeline:
1. **Ingestion (`src/pdf_loader.py`)**: Uses `pdfplumber` to parse the official PDF page-by-page, preserving whitespaces, page numbers, and structural breaks.
2. **Chunking (`src/chunking.py`)**: Implements two distinct chunking strategies (Fixed-size vs. Structure-aware) to compile retrievable text chunks with metadata.
3. **Indexing (`src/embeddings.py`)**: Embeds text using `all-MiniLM-L6-v2` via `sentence-transformers` and builds a local FAISS `IndexFlatIP` database.
4. **Retrieval (`src/retriever.py`)**: Matches the query vector against the index to return the top-5 chunks with cosine similarity scores.
5. **Generation (`src/generator.py`)**: Inserts retrieved chunks into a strict system prompt and calls the Groq API (`llama-3.1-8b-instant`) with `temperature=0` to generate grounded, cited answers.
6. **Frontend (`app.py`)**: A Streamlit chat UI showing the answer and an expandable sources drawer showing location, page number, and similarity score.

---

## Chunking Approaches & Trade-offs

The repository implements and retains both chunking strategies to allow direct evaluation:

### Strategy A: Fixed-size Chunking (Baseline)
- **Implementation**: Concatenates all document lines and slides a window of 500 tokens with a 50-token overlap using `tiktoken` (`cl100k_base`). Metadata (Law & Section) is inherited from the middle line of the chunk.
- **Trade-offs**:
  - *Pros*: Simple to implement; guarantees surrounding context is always preserved.
  - *Cons*: Chunks often split mid-sentence or merge unrelated laws/sections together, creating retrieval noise.

### Strategy B: Structure-aware Semantic Chunking
- **Implementation**: Groups document lines by their natural Law boundary (detected via transition pages) and Section headers (lines starting with `\d+\.\s+[A-Z]`). Sections exceeding 250 tokens are sub-chunked at paragraph (`\n\n`) or sentence boundaries.
- **Trade-offs**:
  - *Pros*: Highly precise; keeps rules grouped logically by their legal code structure.
  - *Cons*: Suffer from context loss when lists or clauses get separated from the introductory header that gives them meaning (e.g. splitting a list of offences from the sentence "A player is sent off if they commit:").

---

## Test Question Results & Strategy Comparison

We evaluated both strategies across 10 sample questions representing factual queries, edge cases, and out-of-scope prompts. Below is the summary of results stored in `test_results.md`:

| # | Question | Strategy A Status | Strategy B Status | Comparison & Key Insights |
|---|---|---|---|---|
| 1 | What are the dimensions of a football pitch? | ❌ Refused |  Answered | **Strategy B Win**: Strategy B pinpointed the exact international dimensions clause in Law 1, Section 4. Strategy A's chunk boundaries diluted the similarity score, causing a refusal. |
| 2 | How many substitutions are allowed in a match? |  Answered |  Answered | **Draw**: Both answered successfully. Strategy B's response was far more structured and detailed, citing the exact page (51) and sections. |
| 3 | What happens if a goalkeeper handles a backpass? | ❌ Refused | ❌ Refused | **Vocabulary Mismatch**: The official Laws do not contain the term "backpass" (instead using "deliberately kicked to them by a team-mate"). Neither strategy retrieved the right context without keyword alignment. |
| 4 | Is a handball in the build-up to a goal always disallowed? | ❌ Refused | ❌ Refused | **Complex Query**: The text in the PDF describes handball goal rules using specific phrasing ("immediately after the ball touches..."). The query structure fell below the retrieval threshold. |
| 5 | Can a goal be scored directly from a throw-in? | ❌ Refused |  Answered | **Strategy B Win**: Strategy B cleanly retrieved the single-sentence preamble of Law 15 ("A goal cannot be scored directly from a throw-in"). Strategy A missed this short boundary. |
| 6 | What is the offside rule? |  Answered |  Answered | **Draw**: Both successfully retrieved Law 11 and described the offside position with correct citations. |
| 7 | When can a referee use VAR? |  Answered |  Answered | **Draw**: Both retrieved the VAR protocol and listed the four reviewable categories (goals, penalties, red cards, mistaken identity). |
| 8 | What happens if the ball hits the referee and goes into the goal? |  Answered | ❌ Refused | **Strategy A Win**: Strategy A successfully retrieved the match official touch rule in Law 9, Section 1. Strategy B's smaller semantic split lost the connection, causing a refusal. |
| 9 | Can a player be sent off for using offensive language? |  Answered | ❌ Refused | **Context Loss in Strategy B**: Strategy B retrieved the bullet point list item containing "using offensive... language" on Page 109, but because the chunk was split semantically, it lacked the introductory header indicating these are "sending-off offences". The LLM refused to speculate. Strategy A's larger window kept both together. |
| 10 | What is the recipe for football stadium nachos? | ❌ Refused | ❌ Refused | **Successful Refusal**: Correctly returned "Not covered in the Laws of the Game" for both, confirming zero hallucination. |

---

## Setup & Running Instructions

### 1. Requirements & Virtual Environment
Ensure you have Python 3.10+ installed.

```bash
# Clone the repository and navigate inside
cd "Project 1 RAG"

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Groq API Key:
```bash
cp .env.example .env
```
Open `.env` and replace `your_groq_api_key_here` with your Groq API Key.

### 3. Download the PDF
Download the official IFAB Laws of the Game PDF into the `data` folder:
```bash
mkdir -p data
curl -L -o data/laws_of_the_game.pdf "https://downloads.theifab.com/downloads/laws-of-the-game-2024-25?l=en"
```

### 4. Build FAISS Indexes
Build the vector stores for both chunking strategies:
```bash
python build_index.py
```
This downloads the `all-MiniLM-L6-v2` embedding model, chunks the PDF, embeds the text, and saves the indexes under `data/index/fixed/` and `data/index/semantic/`.

### 5. Run Evaluation Tests
Run the 10 evaluation questions to compare results:
```bash
python test_questions.py
```
This prints the comparison to the console and generates a detailed report in `test_results.md`.

### 6. Start the Streamlit Web Application
Run the Streamlit application to query RefBot through a GUI:
```bash
streamlit run app.py
```
Open the local URL displayed in your terminal (usually `http://localhost:8501`) to start chatting!

---

## What Broke & How It Was Fixed

1. **Virtual Environment Path Issue**:
   - *Symptom*: Executing `./venv/bin/pip` failed with `no such file or directory` in the shell.
   - *Fix*: The directory created by `python3 -m venv .venv` starts with a dot. Corrected references to use `./.venv/bin/pip` and `./.venv/bin/python`.
2. **Standard Output Buffering in Background Tasks**:
   - *Symptom*: When running `test_questions.py` in the background, stdout did not write to the log file in real-time, making it appear hung at "Loading embedding model...".
   - *Fix*: Ran the command with the `PYTHONUNBUFFERED=1` environment variable to disable output buffering.
3. **Dotenv Override Limitations**:
   - *Symptom*: `load_dotenv()` does not override variables already set in the shell session, which was loading the placeholder `your_groq_api_key_here`.
   - *Fix*: Handled this by validating the token string in code and using `load_dotenv(override=True)` or writing the key directly to disk.
4. **Root Path Directory `/data` Permissions**:
   - *Symptom*: Trying to read/write to `/data` at the system root level resulted in Permission Denied.
   - *Fix*: Used a relative path resolved against the project root directory (`ROOT_DIR / "data"`), ensuring all resources remain self-contained within the workspace.
