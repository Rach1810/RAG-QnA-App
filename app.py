import streamlit as st
import requests

# ------------------- Config -------------------
API_BASE = "http://localhost:8000"  # Change if your FastAPI is running elsewhere

# ------------------- Session State Init -------------------
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# ------------------- Sidebar Reset Option -------------------
st.sidebar.title("⚙️ Settings")
if st.sidebar.button("🔄 Reset Upload"):
    st.session_state.uploaded = False
    st.session_state.qa_pairs = []
    st.session_state.last_uploaded = ""
    st.session_state.last_answer = None
    st.sidebar.success("Upload reset. Ready for a new file!")

# ------------------- Title -------------------
st.title("📄 RAG-based Q&A Bot")

# ------------------- Upload Section -------------------
st.header("1️⃣ Upload a Document (Optional)")
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file and uploaded_file.name != st.session_state.last_uploaded:
    with st.spinner("Processing and uploading the document..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        try:
            response = requests.post(f"{API_BASE}/upload", files=files)
            if response.ok:
                msg = response.json().get("message", "File uploaded successfully.")
                st.success(f"✅ {msg}")
                st.session_state.uploaded = True
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.qa_pairs = []
                st.session_state.last_answer = None
            else:
                st.error(f"❌ Upload failed: {response.text}")
        except Exception as e:
            st.error(f"❌ API connection error: {e}")

# ------------------- Ask Section -------------------
st.header("2️⃣ Ask a Question")

question = st.text_input("Ask your question here:")
if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        try:
            payload = {"question": question}
            response = requests.post(f"{API_BASE}/ask", data=payload)
            if response.ok:
                data = response.json()
                answer = data.get("answer", "No answer received.")
                context = data.get("context", "")
                st.markdown(answer)
                st.session_state.last_answer = (question, answer, context)
                st.session_state.qa_pairs.append((question, answer, context))
            else:
                st.error("❌ Error: " + response.text)
        except Exception as e:
            st.error(f"❌ API connection error: {e}")


# ------------------- Q&A History -------------------
if st.session_state.qa_pairs:
    st.markdown("---")
    st.header("📚 Q&A History")
    for idx, (q, a, ctx) in enumerate(reversed(st.session_state.qa_pairs), 1):
        with st.expander(f"❓ Q{idx}: {q}"):
            st.markdown(f"**Answer:** {a}")
            if ctx:
                with st.expander("🔍 View context used"):
                    st.text(ctx)
