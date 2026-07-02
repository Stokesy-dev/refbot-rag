import sys
from pathlib import Path
from src.config import PDF_PATH, FIXED_INDEX_DIR, SEMANTIC_INDEX_DIR
from src.pdf_loader import load_pdf
from src.chunking import chunk_fixed_size, chunk_by_structure
from src.embeddings import build_index

def main():
    print("=== RefBot Indexing CLI ===")
    
    if not PDF_PATH.exists():
        print(f"Error: PDF file not found at: {PDF_PATH}")
        print("Please ensure the IFAB Laws of the Game PDF is placed inside the 'data/' directory.")
        sys.exit(1)
        
    print(f"Parsing PDF: {PDF_PATH}")
    try:
        pages = load_pdf(PDF_PATH)
        print(f"Successfully extracted {len(pages)} pages.")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        sys.exit(1)
        
    # --- Strategy A: Fixed-size chunking ---
    print("\n--- Strategy A: Fixed-size chunking (500 tokens, 50 overlap) ---")
    try:
        fixed_chunks = chunk_fixed_size(pages)
        print(f"Generated {len(fixed_chunks)} chunks.")
        print(f"Building and saving FAISS index to: {FIXED_INDEX_DIR}")
        build_index(fixed_chunks, FIXED_INDEX_DIR)
    except Exception as e:
        print(f"Error building fixed-size index: {e}")
        
    # --- Strategy B: Structure-aware chunking ---
    print("\n--- Strategy B: Structure-aware chunking (semantic boundaries) ---")
    try:
        semantic_chunks = chunk_by_structure(pages)
        print(f"Generated {len(semantic_chunks)} chunks.")
        print(f"Building and saving FAISS index to: {SEMANTIC_INDEX_DIR}")
        build_index(semantic_chunks, SEMANTIC_INDEX_DIR)
    except Exception as e:
        print(f"Error building structure-aware index: {e}")
        
    print("\n=== Indexing Complete ===")

if __name__ == "__main__":
    main()
