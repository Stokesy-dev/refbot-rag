import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

def get_embedding_model() -> SentenceTransformer:
    """Loads and returns the SentenceTransformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)

def build_index(chunks: list[dict], index_dir: Path):
    """
    Encodes chunk texts, builds a FAISS IndexFlatIP (inner product/cosine similarity),
    and persists index and metadata JSON to the given index_dir.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    
    print(f"Encoding {len(texts)} chunks...")
    # Normalize embeddings so that inner product equals cosine similarity
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save FAISS index
    index_path = index_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    
    # Save metadata JSON sidecar
    metadata_path = index_dir / "chunks_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
        
    print(f"Index built successfully and saved to {index_dir}")

def load_index(index_dir: Path):
    """
    Loads FAISS index and metadata sidecar from index_dir.
    Returns (faiss_index, metadata_list)
    """
    index_dir = Path(index_dir)
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "chunks_metadata.json"
    
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Index files not found in: {index_dir}")
        
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    return index, metadata
