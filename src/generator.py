from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Formats the context, constructs the system prompt to enforce no-hallucination
    and citation rules, calls the Groq API, and handles errors gracefully.
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY is not set. Please add your Groq API key to the .env file in the root directory."
        
    # Format retrieved chunks into context block
    formatted_contexts = []
    for idx, chunk in enumerate(retrieved_chunks):
        formatted_contexts.append(
            f"Source [{idx + 1}]:\n"
            f"Location: {chunk['law']} - {chunk['section']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Text: {chunk['text']}\n"
        )
    context_str = "\n".join(formatted_contexts)
    
    system_prompt = (
        "You are RefBot, an expert football referee assistant. Your task is to answer the user's question "
        "using ONLY the provided context blocks. Each context block is labeled with a Law, section, and page number.\n\n"
        "Follow these rules strictly:\n"
        "1. Base your answer ONLY on the provided context. Do NOT use any external or general knowledge.\n"
        "2. If the provided context does not contain the answer to the question, you must respond EXACTLY: "
        "'Not covered in the Laws of the Game.' Do not add any other words, greetings, or explanations.\n"
        "3. Cite the specific Law number and section (e.g., 'According to Law 12, Section 1...') for every claim or rule you state.\n"
        "4. Be extremely precise, concise, and professional. Do not speculate or extrapolate.\n"
        "5. If there are contradictions or multiple points in the context, mention them with their respective citations."
    )
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.0  # Set to 0 for deterministic output
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg.lower():
            return "Groq API Error: Rate limit exceeded. Please try again in a moment."
        elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            return "Groq API Error: Authentication failed. Please check your GROQ_API_KEY in the .env file."
        return f"Groq API Error occurred: {error_msg}"
