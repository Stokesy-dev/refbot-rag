import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PDF_PATH = DATA_DIR / "laws_of_the_game.pdf"
INDEX_DIR = DATA_DIR / "index"
FIXED_INDEX_DIR = INDEX_DIR / "fixed"
SEMANTIC_INDEX_DIR = INDEX_DIR / "semantic"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
