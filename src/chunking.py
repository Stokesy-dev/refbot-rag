import re
import tiktoken
from collections import defaultdict
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

LAW_TITLES = {
    1: "The Field of Play",
    2: "The Ball",
    3: "The Players",
    4: "The Players' Equipment",
    5: "The Referee",
    6: "The Other Match Officials",
    7: "The Duration of the Match",
    8: "The Start and Restart of Play",
    9: "The Ball in and out of Play",
    10: "Determining the Outcome of a Match",
    11: "Offside",
    12: "Fouls and Misconduct",
    13: "Free Kicks",
    14: "The Penalty Kick",
    15: "The Throw-in",
    16: "The Goal Kick",
    17: "The Corner Kick",
}

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a text string using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def parse_document_to_lines(pages: list[dict]) -> list[dict]:
    """
    Parses the pages list into structured lines with metadata.
    Detects Law transitions and numbered sub-section headers.
    """
    lines_metadata = []
    current_law = None
    current_section = "Intro"
    
    for page in pages:
        page_num = page["page_number"]
        raw_text = page["text"] or ""
        
        # Check if the page is a Law transition page
        stripped_text = raw_text.strip()
        # Law transition page text is usually like "Law\n1" or "Law\n12"
        law_match = re.match(r'^Law\s*\n\s*(\d+)$', stripped_text, re.IGNORECASE)
        if law_match:
            current_law = int(law_match.group(1))
            current_section = "Preamble"
            lines_metadata.append({
                "text": stripped_text.replace("\n", " "),
                "page_number": page_num,
                "law": f"Law {current_law}: {LAW_TITLES.get(current_law, '')}",
                "section": "Transition Page"
            })
            continue
            
        # Handle outside standard Law pages
        if page_num < 36:
            current_law = "Introductory Material"
            if 1 <= page_num <= 5:
                current_section = "Preface"
            elif 6 <= page_num <= 8:
                current_section = "Contents"
            elif 9 <= page_num <= 15:
                current_section = "About the Laws"
            elif 16 <= page_num <= 19:
                current_section = "Notes and modifications"
            elif 20 <= page_num <= 23:
                current_section = "General modifications"
            elif 24 <= page_num <= 27:
                current_section = "Guidelines for temporary dismissals (sin bins)"
            elif 28 <= page_num <= 29:
                current_section = "Guidelines for return substitutes"
            else:
                current_section = "Additional permanent concussion substitutions protocol"
        elif page_num > 141:
            if 142 <= page_num <= 155:
                current_law = "VAR Protocol"
                current_section = "Video Assistant Referee (VAR) Protocol"
            elif 156 <= page_num <= 170:
                current_law = "Law Changes"
                current_section = "Law Changes 2024/25"
            elif 171 <= page_num <= 183:
                current_law = "Glossary"
                current_section = "Glossary of Football Terms"
            else:
                current_law = "Practical Guidelines"
                current_section = "Practical Guidelines for Match Officials"
                
        # Split text into lines
        lines = raw_text.split("\n")
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # If we are inside a Law, check for numbered section headers
            if isinstance(current_law, int):
                # E.g. "1. Field surface" or "12. Commercial advertising"
                section_match = re.match(r'^(\d+)\.\s+([A-Z].*)$', line_stripped)
                if section_match:
                    current_section = line_stripped
                    
            lines_metadata.append({
                "text": line_stripped,
                "page_number": page_num,
                "law": f"Law {current_law}: {LAW_TITLES.get(current_law, '')}" if isinstance(current_law, int) else str(current_law),
                "section": current_section
            })
            
    return lines_metadata

def chunk_fixed_size(pages: list[dict]) -> list[dict]:
    """
    Implements Chunking Strategy A (Fixed-size sliding window on lines).
    Chunk size = 500 tokens, overlap = 50 tokens.
    Metadata is assigned from the middle line of the chunk.
    """
    lines = parse_document_to_lines(pages)
    if not lines:
        return []
        
    chunks = []
    current_lines = []
    current_tokens = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_tokens = count_tokens(line["text"] + "\n")
        
        if line_tokens > CHUNK_SIZE:
            if current_lines:
                mid_idx = len(current_lines) // 2
                mid_line = current_lines[mid_idx]
                chunks.append({
                    "text": "\n".join([l["text"] for l in current_lines]),
                    "law": mid_line["law"],
                    "section": mid_line["section"],
                    "page_number": mid_line["page_number"],
                    "strategy": "fixed"
                })
                current_lines = []
                current_tokens = 0
            chunks.append({
                "text": line["text"],
                "law": line["law"],
                "section": line["section"],
                "page_number": line["page_number"],
                "strategy": "fixed"
            })
            i += 1
            continue
            
        if current_tokens + line_tokens <= CHUNK_SIZE:
            current_lines.append(line)
            current_tokens += line_tokens
            i += 1
        else:
            mid_idx = len(current_lines) // 2
            mid_line = current_lines[mid_idx]
            chunks.append({
                "text": "\n".join([l["text"] for l in current_lines]),
                "law": mid_line["law"],
                "section": mid_line["section"],
                "page_number": mid_line["page_number"],
                "strategy": "fixed"
            })
            
            # Slide back for overlap
            overlap_lines = []
            overlap_tokens = 0
            for l in reversed(current_lines):
                l_tokens = count_tokens(l["text"] + "\n")
                if overlap_tokens + l_tokens <= CHUNK_OVERLAP:
                    overlap_lines.insert(0, l)
                    overlap_tokens += l_tokens
                else:
                    break
            
            current_lines = overlap_lines
            current_tokens = overlap_tokens
            
    if current_lines:
        mid_idx = len(current_lines) // 2
        mid_line = current_lines[mid_idx]
        chunks.append({
            "text": "\n".join([l["text"] for l in current_lines]),
            "law": mid_line["law"],
            "section": mid_line["section"],
            "page_number": mid_line["page_number"],
            "strategy": "fixed"
        })
        
    return chunks

def split_section_into_subchunks(text: str, law: str, section: str, page_number: int) -> list[dict]:
    """Splits a section text into subchunks at paragraph/sentence boundaries of max 250 tokens."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        
    subchunks = []
    current_para_group = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        if para_tokens > 250:
            if current_para_group:
                subchunks.append({
                    "text": "\n\n".join(current_para_group),
                    "law": law,
                    "section": section,
                    "page_number": page_number,
                    "strategy": "semantic"
                })
                current_para_group = []
                current_tokens = 0
                
            # Sentence split
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_sent_group = []
            current_sent_tokens = 0
            
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if sent_tokens > 250:
                    if current_sent_group:
                        subchunks.append({
                            "text": " ".join(current_sent_group),
                            "law": law,
                            "section": section,
                            "page_number": page_number,
                            "strategy": "semantic"
                        })
                        current_sent_group = []
                        current_sent_tokens = 0
                    subchunks.append({
                        "text": sent,
                        "law": law,
                        "section": section,
                        "page_number": page_number,
                        "strategy": "semantic"
                    })
                elif current_sent_tokens + sent_tokens <= 250:
                    current_sent_group.append(sent)
                    current_sent_tokens += sent_tokens
                else:
                    subchunks.append({
                        "text": " ".join(current_sent_group),
                        "law": law,
                        "section": section,
                        "page_number": page_number,
                        "strategy": "semantic"
                    })
                    current_sent_group = [sent]
                    current_sent_tokens = sent_tokens
            if current_sent_group:
                subchunks.append({
                    "text": " ".join(current_sent_group),
                    "law": law,
                    "section": section,
                    "page_number": page_number,
                    "strategy": "semantic"
                })
            continue
            
        if current_tokens + para_tokens <= 250:
            current_para_group.append(para)
            current_tokens += para_tokens
        else:
            subchunks.append({
                "text": "\n\n".join(current_para_group),
                "law": law,
                "section": section,
                "page_number": page_number,
                "strategy": "semantic"
            })
            current_para_group = [para]
            current_tokens = para_tokens
            
    if current_para_group:
        subchunks.append({
            "text": "\n\n".join(current_para_group),
            "law": law,
            "section": section,
            "page_number": page_number,
            "strategy": "semantic"
        })
        
    return subchunks

def chunk_by_structure(pages: list[dict]) -> list[dict]:
    """
    Implements Chunking Strategy B (Structure-aware semantic chunking).
    Groups lines by Law/section, and splits long sections (>250 tokens) at paragraph boundaries.
    """
    lines = parse_document_to_lines(pages)
    if not lines:
        return []
        
    grouped_sections = defaultdict(list)
    section_first_page = {}
    ordered_keys = []
    
    for line in lines:
        key = (line["law"], line["section"])
        if key not in grouped_sections:
            ordered_keys.append(key)
            section_first_page[key] = line["page_number"]
        grouped_sections[key].append(line)
        
    chunks = []
    for key in ordered_keys:
        law, section = key
        section_lines = grouped_sections[key]
        page_num = section_first_page[key]
        
        section_text = "\n".join([l["text"] for l in section_lines])
        
        if count_tokens(section_text) <= 250:
            chunks.append({
                "text": section_text,
                "law": law,
                "section": section,
                "page_number": page_num,
                "strategy": "semantic"
            })
        else:
            sub_chunks = split_section_into_subchunks(section_text, law, section, page_num)
            chunks.extend(sub_chunks)
            
    return chunks
