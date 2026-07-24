"""
main.py — RAG chatbot as a web API for Starr Mark Tennis
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from anthropic import Anthropic
import voyageai

# ============================================================================
# LOAD DOCUMENTS AT STARTUP (only read the file, don't call APIs yet)
# ============================================================================

def load_documents(filepath="documents.txt"):
    """Load documents from a text file."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using placeholder documents.")
        return [
            "Contact Starr Mark Tennis for camp information.",
            "Summer camp runs Monday through Friday.",
            "Ages 6-18 welcome for all skill levels.",
        ]
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split by double newlines (paragraph breaks)
    paragraphs = text.split("\n\n")
    documents = [p.strip() for p in paragraphs if p.strip()]
    print(f"Loaded {len(documents)} document chunks from {filepath}")
    return documents


documents = load_documents()

# ============================================================================
# INITIALIZE FASTAPI APP (must happen before decorators)
# ============================================================================

app = FastAPI()

# Add CORS middleware — allow all origins (Claude artifacts, any website, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from anywhere
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ============================================================================
# LAZY-LOADED GLOBALS (initialized on first request, not at startup)
# ============================================================================

voyage_client = None
pinecone_index = None
anthropic_client = None
docs_uploaded = False


def get_voyage_client():
    """Get or create Voyage AI client."""
    global voyage_client
    if voyage_client is None:
        voyage_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
    return voyage_client


def embed_text(text):
    """Call Voyage AI to embed text."""
    client = get_voyage_client()
    result = client.embed([text], model="voyage-4-lite")
    return result.embeddings[0]


def init_pinecone_and_anthropic():
    """Initialize Pinecone and Anthropic on first use."""
    global pinecone_index, anthropic_client
    
    if pinecone_index is not None:
        return  # already initialized
    
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.environ.get("PINECONE_INDEX_NAME"))
    
    print("Connecting to Anthropic...")
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def upload_docs():
    """Upload documents to Pinecone on first chat request."""
    global docs_uploaded
    
    if docs_uploaded:
        return  # already done
    
    init_pinecone_and_anthropic()
    
    print("Uploading documents to Pinecone...")
    vectors_to_upsert = []
    
    for i, doc in enumerate(documents):
        try:
            embedding = embed_text(doc)
            vectors_to_upsert.append({
                "id": f"doc_{i}",
                "values": embedding,
                "metadata": {"text": doc}
            })
        except Exception as e:
            print(f"Warning: failed to embed doc {i}: {e}")
    
    if vectors_to_upsert:
        pinecone_index.upsert(vectors=vectors_to_upsert)
        print(f"Uploaded {len(vectors_to_upsert)} chunks to Pinecone")
    
    docs_uploaded = True


def retrieve(question, n_results=2):
    """Search Pinecone for relevant documents."""
    init_pinecone_and_anthropic()
    question_embedding = embed_text(question)
    results = pinecone_index.query(
        vector=question_embedding,
        top_k=n_results,
        include_metadata=True
    )
    return [match["metadata"]["text"] for match in results["matches"]]


def generate_answer(question, context_chunks):
    """Use Claude to generate an answer based on retrieved context."""
    init_pinecone_and_anthropic()
    
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Use the following context to answer the question. "
        f"If the context doesn't contain the answer, say so."
        f"Do not use emojis. Produce only plain text.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    
    system_prompt = (
        "You are a helpful assistant answering questions about Starr Mark Tennis camp. "
        "Be friendly, informative, and concise. "
        "If you don't have the information, offer to have someone contact them."
    )
    
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ============================================================================
# API ENDPOINTS
# ============================================================================

class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Handle chat requests."""
    upload_docs()  # Initialize everything on first request
    chunks = retrieve(req.question)
    answer = generate_answer(req.question, chunks)
    return {"answer": answer}


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}