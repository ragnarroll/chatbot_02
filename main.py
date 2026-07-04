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
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# -----------------------------------------------------------------------
# Load documents from a text file
# -----------------------------------------------------------------------
def load_documents(filepath="documents.txt", chunk_size=500):
    """
    Load documents from a text file and split into chunks.
    
    Args:
        filepath: path to the .txt file (default: documents.txt in same folder)
        chunk_size: approximate words per chunk (default: 500)
    
    Returns:
        list of document chunks
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using placeholder documents.")
        return [
            "The Bethesda summer tennis camp offers structured, high-energy sessions focusing on tennis, fitness, and friendships. The camp is open to children and teens ages 6 to 18 of all skill levels, from beginners to advanced, with groups organized by ability.",    
            "The summer tennis camp runs Monday through Friday. Half-day sessions operate from 9 AM to 12 PM and cost $330 per week. Full-day sessions operate from 9 AM to 3 PM and cost $550 per week.",
            "For the first part of the summer, specifically the weeks of June 8th through June 18th, the tennis camp is located at Meadowbrook Park Tennis Courts, 6321 Meadowbrook Lane, Chevy Chase, MD 20815.",
            "For the remainder of the summer, from June 22nd through August 14th, the tennis camp is located at Westland Middle School Tennis Courts, 5511 Massachusetts Ave, Bethesda, MD 20816.",
        ]
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split by double newlines (paragraphs) first, then further split if needed
    paragraphs = text.split("\n\n")
    documents = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If paragraph is too long, split by sentences
        words = para.split()
        if len(words) > chunk_size:
            sentences = para.split(". ")
            current_chunk = []
            current_word_count = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                if current_word_count + sentence_words > chunk_size and current_chunk:
                    documents.append(". ".join(current_chunk) + ".")
                    current_chunk = [sentence]
                    current_word_count = sentence_words
                else:
                    current_chunk.append(sentence)
                    current_word_count += sentence_words
            
            if current_chunk:
                documents.append(". ".join(current_chunk) + ".")
        else:
            documents.append(para)
    
    print(f"Loaded {len(documents)} document chunks from {filepath}")
    return documents


documents = load_documents()

# -----------------------------------------------------------------------
# Lazy initialization: these don't load until the first request, avoiding
# memory spikes during Render's build phase (free tier = 512MB limit).
# -----------------------------------------------------------------------
embedder = None
pinecone_index = None
anthropic_client = None


def init():
    global embedder, pinecone_index, anthropic_client
    if embedder is not None:
        return  # already initialized
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.environ.get("PINECONE_INDEX_NAME"))
    
    print("Uploading documents to Pinecone...")
    vectors_to_upsert = []
    for i, doc in enumerate(documents):
        embedding = embedder.encode(doc).tolist()
        vectors_to_upsert.append({
            "id": f"doc_{i}",
            "values": embedding,
            "metadata": {"text": doc}
        })
    pinecone_index.upsert(vectors=vectors_to_upsert)
    
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def retrieve(question, n_results=2):
    question_embedding = embedder.encode(question).tolist()
    results = pinecone_index.query(
        vector=question_embedding,
        top_k=n_results,
        include_metadata=True
    )
    # Extract the actual text from metadata
    return [match["metadata"]["text"] for match in results["matches"]]


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
