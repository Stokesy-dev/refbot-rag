import pdfplumber
from pathlib import Path

def load_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Extracts text page-by-page from the PDF using pdfplumber.
    Returns a list of dicts: [{"page_number": int, "text": str}]
    """
    pages_data = []
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages_data.append({
                "page_number": idx + 1,
                "text": text
            })
            
    return pages_data
