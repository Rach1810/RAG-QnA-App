# RAG-based QnA System

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline that allows you to:

* Upload PDF/TXT documents
* Embed and store them in a **Qdrant** vector database
* Ask questions related to the uploaded content
* Use **Hugging Face models** via API (Model used: `Mistral-7B-Instruct`)
* View responses through a simple **Streamlit frontend**

---

## 📁 Project Structure

```
├── app.py              # Streamlit frontend for uploading files and QnA
├── main.py             # FastAPI backend for handling uploads, embeddings, and LLM interaction
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (HF_API_KEY, QDRANT_URL, etc.)
├── test_model.py       # File to test the Hugging face API
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Rach1810/RAG-QnA-App.git
cd RAG-QnA-App
```

---

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv ragvenv
source ragvenv/bin/activate  # macOS/Linux
# or
ragvenv\Scripts\activate     # Windows
```

---

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Environment Variables

Create a `.env` file in the root directory with the following content:

```env
HF_API_KEY=your_huggingface_api_key
QDRANT_URL=https://your-qdrant-instance-url
QDRANT_API_KEY=your_qdrant_api_key
```
---
### Step-by-Step: Qdrant Cloud Setup

### 1. Sign Up (Free)

Go to [https://cloud.qdrant.io](https://cloud.qdrant.io) and sign up.


### 2. Create a Cluster

* Click **"Create Cluster"**
* Choose:

  * **Free Tier**
  * Any region (e.g., `AWS us-east-1`)
* Give it a name like `rag-demo`
* Wait for it to be provisioned

### 3. Get API Key and URL

Once the cluster is created:

* Click on the cluster
* Go to the **API Keys** tab

  * Create a key (e.g., "dev-key")
  * Copy it

Also copy your **Cluster URL** — it’ll look like:

```
https://abc-xyz.qdrant.xyz
```
---

## 🚀 Running the Application

### Step 1: Start the FastAPI Backend

```bash
uvicorn main:app --reload
```

This will run the backend at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### Step 2: Start the Streamlit Frontend

In another terminal (in the same environment):

```bash
streamlit run app.py
```

This will launch the frontend at: [http://localhost:8501](http://localhost:8501)

---

## 💡 Features

- Upload `.pdf` or `.txt` files
- Avoids duplicate storage (via content hashing)
- Embeds document chunks using `sentence-transformers`
- Stores and queries with Qdrant vector DB
- Answers questions using Hugging Face inference API
- Ask general or document-based questions anytime

---

## 🗂️ File Descriptions

### `main.py`

* FastAPI backend
* Handles file uploads, embeddings, vector storage
* Fetches relevant chunks and uses Hugging Face API for answers

### `app.py`

* Streamlit interface
* File uploader and chat interface
* Maintains user session state, displays chat and history

---
## 📬 Contact / Contributions

Open an issue or pull request for improvements.

---