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
from anthropic import Anthropic
import voyageai

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
            "Dunking requires explosive leg power, which is built through plyometric exercises.",
            "Georgetown University's Office of Advancement manages alumni engagement.",
            "The Mazda3 hatchback is popular for city driving in Washington DC.",
            "Learning Design and Technology focuses on how people learn.",
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
# Voyage AI for embeddings (cheap, high quality: voyage-4-lite)
# -----------------------------------------------------------------------
voyage_client = None


def get_voyage_client():
    global voyage_client
    if voyage_client is None:
        voyage_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
    return voyage_client


def embed_text(text):
    """
    Call Voyage AI's embedding API to get embeddings.
    Returns 1024-dimensional vectors (voyage-4-lite model).
    Cheaper and faster than OpenAI, excellent quality.
    """
    client = get_voyage_client()
    result = client.embed(
        [text],
        model="voyage-4-lite"
    )
    return result.embeddings[0]


# -----------------------------------------------------------------------
# Lazy initialization: connections happen on first chat request
# -----------------------------------------------------------------------
pinecone_index = None
anthropic_client = None
docs_uploaded = False


def init():
    """Initialize Pinecone and Anthropic connections (safe to do on first request)."""
    global pinecone_index, anthropic_client
    
    if pinecone_index is not None:
        return  # already initialized
    
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.environ.get("PINECONE_INDEX_NAME"))
    
    print("Connecting to Anthropic...")
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def upload_documents_to_pinecone():
    """Upload documents to Pinecone on first chat request (when server is running)."""
    global docs_uploaded
    
    if docs_uploaded:
        return  # already done
    
    init()  # ensure Pinecone is connected
    
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
        print(f"Uploaded {len(vectors_to_upsert)} document chunks to Pinecone")
    
    docs_uploaded = True


def retrieve(question, n_results=2):
    init()  # ensure Pinecone is connected
    question_embedding = embed_text(question)
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


# Startup hook — just validate environment, don't call APIs yet
@app.on_event("startup")
def startup():
    # Check that required env vars exist (fail fast if missing)
    required = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    print(f"✓ All environment variables set")

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
    upload_documents_to_pinecone()  # upload on first request (safe when server is running)
    chunks = retrieve(req.question)
    answer = generate_answer(req.question, chunks)
    return {"answer": answer}


@app.get("/")
def health_check():
    return {"status": "ok"}
