import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()

from src.config import FIXED_INDEX_DIR, SEMANTIC_INDEX_DIR
from src.embeddings import get_embedding_model, load_index
from src.retriever import retrieve
from src.generator import generate_answer

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="RefBot — Football Rules Q&A",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caching embedding model load
@st.cache_resource
def load_embedding_model():
    return get_embedding_model()

# Caching FAISS index loads
@st.cache_resource
def load_faiss_index(index_dir):
    return load_index(index_dir)

# Initialize resources
model = load_embedding_model()
groq_key = os.getenv("GROQ_API_KEY")

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Chunking Strategy Toggle
    strategy_label = st.radio(
        "Select Chunking Strategy:",
        ("Fixed-size (Strategy A)", "Structure-aware (Strategy B)"),
        help="Strategy A uses sliding window of 500 tokens. Strategy B splits strictly by Law and section boundaries."
    )
    
    st.markdown("---")
    
    # Display descriptions based on strategy selection
    if "Fixed-size" in strategy_label:
        st.subheader("Strategy A: Fixed-size Chunks")
        st.markdown(
            "- **Size:** 500 tokens, 50 overlap\n"
            "- **Pros:** Surrounding context of sentences is preserved; avoids losing context headers.\n"
            "- **Cons:** Chunks may merge content across unrelated rules or end in the middle of a sentence."
        )
        selected_index_dir = FIXED_INDEX_DIR
    else:
        st.subheader("Strategy B: Structure-aware Chunks")
        st.markdown(
            "- **Size:** Groups lines by Law & Section, splits long sections (>250 tokens) at paragraph boundaries.\n"
            "- **Pros:** Keeps rules grouped logically by official law structure; higher precision.\n"
            "- **Cons:** Can suffer from context loss if bullet lists get separated from their section introduction."
        )
        selected_index_dir = SEMANTIC_INDEX_DIR
        
    st.markdown("---")
    st.markdown("**LLM Engine:** Llama 3.1 8B (Groq)")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("---")
    st.info("Ensure laws_of_the_game.pdf is in the `data/` folder and indexing is run before querying.")

# Main Page Layout
st.title("⚽ RefBot — Football Rules Q&A")
st.write(
    "Ask RefBot any question about football rules! Answers are generated using ONLY the official "
    "**IFAB Laws of the Game 2024/25** and cited with specific Law, Section, and page numbers."
)

# Verify if indexes exist
if not FIXED_INDEX_DIR.exists() or not SEMANTIC_INDEX_DIR.exists():
    st.error(
        "FAISS indexes not found on disk. Please run the index builder first:\n"
        "```bash\npython build_index.py\n```"
    )
    st.stop()

# Load the selected index
try:
    index, metadata = load_faiss_index(selected_index_dir)
except Exception as e:
    st.error(f"Error loading index: {e}")
    st.stop()

# Session state initialization for stateless single-turn Q&A
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "last_strategy" not in st.session_state:
    st.session_state.last_strategy = strategy_label

# If strategy was toggled, invalidate current answer to force rerun or clear sources
if st.session_state.last_strategy != strategy_label:
    st.session_state.last_strategy = strategy_label
    # If there is a question, rerun search with the new index
    if st.session_state.current_question:
        with st.spinner("Rerunning query with new chunking strategy..."):
            hits = retrieve(st.session_state.current_question, index, metadata, model)
            st.session_state.current_sources = hits
            st.session_state.current_answer = generate_answer(st.session_state.current_question, hits)

# Chat Input box
user_query = st.chat_input("Ask a rules question (e.g., 'Is a handball in the build-up to a goal always disallowed?')")

if user_query:
    st.session_state.current_question = user_query
    
    with st.spinner("RefBot is searching the Laws of the Game..."):
        # 1. Retrieve top-k chunks
        hits = retrieve(user_query, index, metadata, model)
        st.session_state.current_sources = hits
        
        # 2. Call generator
        if not groq_key or "your_" in groq_key:
            st.session_state.current_answer = (
                "Error: Groq API Key is missing or invalid. Please check your `.env` file "
                "and set a valid `GROQ_API_KEY`."
            )
        else:
            answer = generate_answer(user_query, hits)
            st.session_state.current_answer = answer

# Render Q&A blocks
if st.session_state.current_question:
    # User message
    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.current_question}**")
        
    # RefBot response
    with st.chat_message("assistant", avatar="⚽"):
        st.markdown(st.session_state.current_answer)
        
        # If successfully answered, display expandable sources
        if st.session_state.current_sources and "Error:" not in st.session_state.current_answer:
            st.markdown("---")
            with st.expander("📚 View Retrieved Sources", expanded=True):
                for idx, hit in enumerate(st.session_state.current_sources):
                    st.markdown(f"##### Source {idx + 1}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Location:** `{hit['law']}`")
                    with col2:
                        st.markdown(f"**Section:** `{hit['section']}`")
                    with col3:
                        st.markdown(f"**Page:** `{hit['page_number']}` | **Similarity:** `{hit['score']*100:.1f}%`")
                    
                    st.info(hit['text'])
