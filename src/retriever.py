import numpy as np
from src.config import TOP_K

def retrieve(query: str, index, metadata: list, model, k: int = TOP_K) -> list[dict]:
    """
    Encodes the query, searches the FAISS index, and maps result indices to metadata.
    Returns a list of dicts, each representing a retrieved chunk with its text,
    law/section metadata, page number, and similarity score.
    """
    # Encode query and normalize vector
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")
    
    # Search FAISS index
    scores, indices = index.search(query_vec, k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        # idx can be -1 if FAISS returns fewer results than requested
        if idx == -1 or idx >= len(metadata):
            continue
            
        chunk_data = metadata[idx]
        results.append({
            "text": chunk_data["text"],
            "law": chunk_data["law"],
            "section": chunk_data["section"],
            "page_number": chunk_data["page_number"],
            "score": float(score)  # Since vectors are normalized, score is Cosine Similarity
        })
        
    return results
