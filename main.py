"""
rag_chatbot.py
A minimal, from-scratch RAG (Retrieval-Augmented Generation) chatbot.

The 4 steps, matching what we walked through by hand:
  1. EMBED    - turn text into vectors using a real neural embedding model
  2. STORE    - save those vectors in a vector database (Chroma)
  3. RETRIEVE - find the closest-matching chunks to a user's question
  4. GENERATE - hand those chunks + the question to an LLM to write the answer

Run this with:  python3 rag_chatbot.py
"""

import os
import chromadb
import voyageai
from anthropic import Anthropic

# -----------------------------------------------------------------------
# STEP 0: Load documents from a .txt file
# load them from .txt files (see load_documents_from_folder() below).
# -----------------------------------------------------------------------
print("Loading knowledge base...")

# 1. Open the text file in "read" mode ("r")
with open("documents.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# 2. Split the massive block of text into a list of separate chunks
# We split by "\n\n" (which represents a blank line between paragraphs)
documents = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]

print(f"Successfully loaded {len(documents)} chunks of information.")

# -----------------------------------------------------------------------
# STEP 1: EMBEDDING

# Voyage AI uses a neural network to turn text into vectors that capture
# MEANING. The voyage-4-lite model returns 1024-dimensional vectors.
# Requires VOYAGE_API_KEY environment variable.
# -----------------------------------------------------------------------
print("Initializing Voyage AI client (voyage-4-lite)...")
voyage_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
 
 
def embed(texts):
    """Turn a list of strings into a list of 1024-dimensional vectors."""
    result = voyage_client.embed(
        texts,
        model="voyage-4-lite"
    )
    return [emb for emb in result.embeddings]

# -----------------------------------------------------------------------
# STEP 2: STORAGE
# Chroma is a lightweight vector database. We create a "collection"
# (like a table) and add our documents + their vectors to it.
# -----------------------------------------------------------------------
print("Setting up vector database...")
chroma_client = chromadb.Client()  # in-memory; use PersistentClient() to save to disk
collection = chroma_client.create_collection(name="my_docs")

collection.add(
    documents=documents,
    embeddings=embed(documents),
    ids=[f"doc_{i}" for i in range(len(documents))],
)


# -----------------------------------------------------------------------
# STEP 3: RETRIEVAL
# Embed the user's question the SAME way, then ask Chroma for the
# n closest document vectors (by distance, same idea as our by-hand demo).
# -----------------------------------------------------------------------
def retrieve(question, n_results=2):
    results = collection.query(
        query_embeddings=embed([question]),
        n_results=n_results,
    )
    return results["documents"][0]  # list of the closest matching chunks


# -----------------------------------------------------------------------
# STEP 4: GENERATION
# Hand the retrieved chunks + the original question to an LLM, with
# instructions to answer USING that context. This requires an API key.
# -----------------------------------------------------------------------
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Use the following context to answer the question. "
        f"Suggest next steps or ask clarifying questions when appropriate."
        f"If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

       # System prompt: tells Claude HOW to behave (tone, format, style)
    system_prompt = (
        "Tell the user that you are a helpful assistant answering questions based on provided context. "
        "Tell the user that they can request to loop in one of our team members via email if they want to, but the response time will be slower. "
        "Keep responses concise and clear. "
        "Use bullet points for lists. "
        "If information is missing, be honest about it. "
        "Maintain a friendly but professional tone."
        
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# -----------------------------------------------------------------------
# Put it all together
# -----------------------------------------------------------------------
def chat(question):
    chunks = retrieve(question)
    print("\n--- Retrieved context ---")
    for c in chunks:
        print(f"  - {c}")
    answer = generate_answer(question, chunks)
    print("\n--- Answer ---")
    print(answer)
    return answer


if __name__ == "__main__":
    print("\nRAG chatbot ready. Type a question (or 'quit' to exit).\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("quit", "exit"):
            break
        chat(q)