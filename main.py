import os
import hashlib
import pdfplumber
import requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import uuid

# Load environment variables - Loads your .env file so os.getenv() can access the API keys and URLs.
load_dotenv()

# Initialize FastAPI

app = FastAPI() # This line creates the API server instance. All your API routes (/upload, /ask) will be defined on this app

# CORS = Cross-Origin Resource Sharing. Required when your frontend is calling your backend from a different domain (e.g., localhost:8501 → localhost:8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, etc.
    allow_headers=["*"],  # Allow all headers
)


# Config
COLLECTION_NAME = "rag_documents"
VECTOR_SIZE = 768  
EMBED_MODEL = "all-mpnet-base-v2"

# Initialize embedding model - This loads a Hugging Face model (all-mpnet-base-v2) that turns chunks of text into vectors.
embedder = SentenceTransformer(EMBED_MODEL)

# Initialize Qdrant - This connects your app to Qdrant — a database specialized in storing and searching vectors.
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Checks if a collection exists in Qdrant. If not, it creates one to store vectors using cosine similarity.
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
    )
    print(f"Created collection `{COLLECTION_NAME}` with vector size {VECTOR_SIZE}")
else:
    print(f"Collection `{COLLECTION_NAME}` already exists")

# Utility functions

def compute_hash(content: str) -> str:
    """Creates an MD5 hash for the entire document content. 
       This helps you check if the same file was already uploaded earlier"""
    return hashlib.md5(content.encode()).hexdigest()

def extract_text(file: UploadFile) -> str:
    """Handles file uploads:
    Supports .txt and .pdf.
    Uses pdfplumber to extract text from PDFs.
    Returns the full plain text."""
    if file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    else:
        raise HTTPException(status_code=400, detail="Only PDF or TXT supported")

def chunk_text(text: str, max_len: int = 1500) -> list:
    """Breaks the text into smaller pieces (~1500 characters).
    This is needed because LLMs work better on smaller inputs.
    Splits by sentence (.) for cleaner context."""
    sentences = text.split(".")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += s + "."
        else:
            chunks.append(current.strip())
            current = s + "."
    if current:
        chunks.append(current.strip())
    return chunks

def query_huggingface_chat(question: str, context: str) -> str:
    """Sends the context + user's question to Hugging Face LLM."""

    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

    system_prompt = (
        "You are a helpful assistant answering questions using the provided context.\n"
        "If the answer is not contained in the context, say: 'I don't know.'\n"
        "Do not make up information. Be concise and clear."
    )

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }

    resp = requests.post(api_url, headers=headers, json=payload)
    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {resp.status_code} - {resp.text}"


## API Endpoints
# In FastAPI, you define these APIs as functions that run when someone hits a specific URL (called a route) 
# with a specific HTTP method (like GET, POST, etc.).



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """@app.post("/upload"): Defines a POST API at /upload
    file: UploadFile = File(...): Tells FastAPI to expect a file (PDF or TXT)
    async def ...: async helps with performance when handling files or I/O"""
    text = extract_text(file)
    file_hash = compute_hash(text)

    # Avoid duplicates
    points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, with_payload=True)
    if any(pt.payload.get("file_hash") == file_hash for pt in points):
        return {"message": "File already processed"}

    # Chunk + embed
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks).tolist()

    # Store in Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            rest.PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={"text": chunks[i], "file_hash": file_hash}
            ) for i in range(len(chunks))
        ]
    )

    return {"message": f"File processed and {len(chunks)} chunks stored."}

@app.post("/ask")
async def ask(question: str = Form(...)):
    """Defines a POST endpoint at /ask
    Accepts a form field question
    Embeds the question, fetches similar content from Qdrant, sends it to LLM, and returns the answer"""
    q_emb = embedder.encode(question).tolist()
    search = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_emb, limit=3)

    if not search:
        return {"answer": "No relevant context found.", "context": ""}

    # Merge top chunks into a single context
    context = "\n".join([hit.payload.get("text", "") for hit in search])

    # Query LLM
    answer = query_huggingface_chat(question, context)

    return {"answer": answer, "context": context}
