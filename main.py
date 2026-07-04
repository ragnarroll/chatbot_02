"""
main.py — your RAG chatbot, exposed as a web API.

Same 4 steps as rag_chatbot.py (embed / store / retrieve / generate),
just wrapped so a website can call it over HTTP instead of you typing
into a terminal.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic


# -----------------------------------------------------------------------
# STEP 0: Load documents from a .txt file
# Your documents — replace with your own content, or later load from files.
# -----------------------------------------------------------------------

print("Loading knowledge base...")

# 1. Open the text file in "read" mode ("r")
with open("camp_data.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# 2. Split the massive block of text into a list of separate chunks
# We split by "\n\n" (which represents a blank line between paragraphs)
documents = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]

print(f"Successfully loaded {len(documents)} chunks of information.")

# -----------------------------------------------------------------------
# Lazy initialization: these don't load until the first request, avoiding
# memory spikes during Render's build phase (free tier = 512MB limit).
# -----------------------------------------------------------------------
embedder = None
collection = None
anthropic_client = None


def init():
    global embedder, collection, anthropic_client
    if embedder is not None:
        return  # already initialized
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Building vector database...")
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_docs")
    collection.add(
        documents=documents,
        embeddings=embedder.encode(documents).tolist(),
        ids=[f"doc_{i}" for i in range(len(documents))],
    )
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def retrieve(question, n_results=2):
    results = collection.query(
        query_embeddings=embedder.encode([question]).tolist(),
        n_results=n_results,
    )
    return results["documents"][0]


def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Use the following context to answer the question. "
        f"If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# -----------------------------------------------------------------------
# The web API itself
# -----------------------------------------------------------------------
app = FastAPI()


# Initialize models on server startup (after build, when running)
@app.on_event("startup")
def startup():
    init()

# CORS: only allow requests from YOUR website. Replace the placeholder
# below with your actual site's domain before deploying for real.
ALLOWED_ORIGINS = [
    "https://your-website.com",       # <-- CHANGE THIS
    "http://localhost:3000",          # handy for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    chunks = retrieve(req.question)
    answer = generate_answer(req.question, chunks)
    return {"answer": answer}


@app.get("/")
def health_check():
    return {"status": "ok"}