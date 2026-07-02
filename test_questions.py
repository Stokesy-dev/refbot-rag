import time
import os
from pathlib import Path
from src.config import FIXED_INDEX_DIR, SEMANTIC_INDEX_DIR
from src.embeddings import get_embedding_model, load_index
from src.retriever import retrieve
from src.generator import generate_answer

QUESTIONS = [
    "What are the dimensions of a football pitch?",
    "How many substitutions are allowed in a match?",
    "What happens if a goalkeeper handles a backpass?",
    "Is a handball in the build-up to a goal always disallowed?",
    "Can a goal be scored directly from a throw-in?",
    "What is the offside rule?",
    "When can a referee use VAR?",
    "What happens if the ball hits the referee and goes into the goal?",
    "Can a player be sent off for using offensive language?",
    "What is the recipe for football stadium nachos?"
]

def main():
    print("=== RefBot Test Evaluation Harness ===")
    
    # Check if Groq API key is set
    # Load environment variables if not already done
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or "your_" in groq_key:
        print("Error: GROQ_API_KEY is not set or has placeholder value in your .env file.")
        print("Please check your .env file and paste a valid Groq API key.")
        return
        
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = get_embedding_model()
    
    print("Loading FAISS indexes...")
    try:
        fixed_index, fixed_metadata = load_index(FIXED_INDEX_DIR)
        print("Strategy A (Fixed-size) loaded successfully.")
    except Exception as e:
        print(f"Error loading Strategy A index: {e}")
        return
        
    try:
        semantic_index, semantic_metadata = load_index(SEMANTIC_INDEX_DIR)
        print("Strategy B (Structure-aware) loaded successfully.")
    except Exception as e:
        print(f"Error loading Strategy B index: {e}")
        return
        
    results_md = []
    results_md.append("# RefBot Evaluation Test Results\n")
    results_md.append(f"Date of Evaluation: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_md.append("This document contains the evaluation of RefBot's answers using Chunking Strategy A (Fixed-size) vs Strategy B (Structure-aware) across 10 test questions.\n")
    results_md.append("| # | Question | Strategy A Status | Strategy B Status | Comparison Summary |\n")
    results_md.append("|---|---|---|---|---|\n")
    
    # We will build two parts: a summary table and a detailed question comparison
    summary_rows = []
    detailed_comparisons = []
    
    for idx, question in enumerate(QUESTIONS):
        print(f"\nProcessing Question {idx+1}/{len(QUESTIONS)}: {question}")
        detailed_comparisons.append(f"\n## Question {idx+1}: {question}\n")
        
        # --- Strategy A: Fixed-size ---
        print("  Running Strategy A (Fixed-size)...")
        fixed_hits = retrieve(question, fixed_index, fixed_metadata, model)
        fixed_answer = generate_answer(question, fixed_hits)
        time.sleep(2)  # Pause to avoid rate limits
        
        # --- Strategy B: Structure-aware ---
        print("  Running Strategy B (Structure-aware)...")
        semantic_hits = retrieve(question, semantic_index, semantic_metadata, model)
        semantic_answer = generate_answer(question, semantic_hits)
        time.sleep(2)  # Pause to avoid rate limits
        
        # Print comparison to console
        print("-" * 60)
        print(f"QUESTION {idx+1}: {question}")
        print("STRATEGY A (Fixed-size) ANSWER:")
        print(fixed_answer)
        print("STRATEGY B (Structure-aware) ANSWER:")
        print(semantic_answer)
        print("-" * 60)
        
        # Determine status
        a_status = "Answered" if "Not covered" not in fixed_answer else "Refused"
        b_status = "Answered" if "Not covered" not in semantic_answer else "Refused"
        
        summary_rows.append(f"| {idx+1} | {question} | {a_status} | {b_status} | Compare details below |\n")
        
        # Add to detailed comparisons list
        detailed_comparisons.append("### Chunking Strategy A: Fixed-size\n")
        detailed_comparisons.append(f"**Answer:**\n{fixed_answer}\n\n")
        detailed_comparisons.append("**Retrieved Sources:**\n")
        for h_idx, hit in enumerate(fixed_hits):
            detailed_comparisons.append(f"- **Source {h_idx+1}:** {hit['law']} - {hit['section']} (Page {hit['page_number']}, Similarity Score: {hit['score']:.4f})\n")
            truncated_text = hit['text'][:180].replace('\n', ' ') + '...'
            detailed_comparisons.append(f"  *\"{truncated_text}\"* \n")
        detailed_comparisons.append("\n")
        
        detailed_comparisons.append("### Chunking Strategy B: Structure-aware\n")
        detailed_comparisons.append(f"**Answer:**\n{semantic_answer}\n\n")
        detailed_comparisons.append("**Retrieved Sources:**\n")
        for h_idx, hit in enumerate(semantic_hits):
            detailed_comparisons.append(f"- **Source {h_idx+1}:** {hit['law']} - {hit['section']} (Page {hit['page_number']}, Similarity Score: {hit['score']:.4f})\n")
            truncated_text = hit['text'][:180].replace('\n', ' ') + '...'
            detailed_comparisons.append(f"  *\"{truncated_text}\"* \n")
        detailed_comparisons.append("\n" + "---" + "\n")
        
    # Write summary table
    results_md.extend(summary_rows)
    results_md.append("\n---\n")
    # Write detailed comparison
    results_md.extend(detailed_comparisons)
    
    # Save file
    output_path = Path("test_results.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(results_md))
        
    print(f"\nTest evaluation complete. Results written to: {output_path.absolute()}")

if __name__ == "__main__":
    main()
