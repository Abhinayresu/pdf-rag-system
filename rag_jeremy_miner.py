import os
import pickle
import requests
import faiss
import numpy as np

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer



# CONFIG

PDF_DIR = "./pdfs"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBED_CACHE_FILE = "chunk_embeddings.pkl"
FAISS_INDEX_FILE = "faiss.index"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 4

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"



# SYSTEM PROMPT (STRICT – DO NOT CHANGE)

SYSTEM_PROMPT = """
You are Jeremy Miner.

Rules you MUST follow:
- Answer ONLY using the provided context.
- If the answer is not explicitly stated in the context, say exactly:
  "Jeremy Miner does not explicitly address this."
- Do NOT use outside knowledge.
- Do NOT generalize.
- Do NOT invent frameworks or terminology.
- Tone must be calm, consultative, and psychology-driven.
- Use short, declarative sentences.
- Explain psychology ONLY if it appears in the context.
- End every answer with:
  Source: <PDF or video title>
"""



# LOAD + CHUNK PDFs (LangChain ONLY)

def load_and_chunk_pdfs():
    documents = []

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, filename))
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = filename
                documents.append(page)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    return chunks



# BUILD OR LOAD FAISS INDEX
def build_or_load_index(chunks):
    if os.path.exists(EMBED_CACHE_FILE) and os.path.exists(FAISS_INDEX_FILE):
        with open(EMBED_CACHE_FILE, "rb") as f:
            metadata = pickle.load(f)
        index = faiss.read_index(FAISS_INDEX_FILE)
        return index, metadata

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    metadata = []
    for chunk in chunks:
        metadata.append({
            "text": chunk.page_content,
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page")
        })

    with open(EMBED_CACHE_FILE, "wb") as f:
        pickle.dump(metadata, f)

    faiss.write_index(index, FAISS_INDEX_FILE)

    return index, metadata


 
# RETRIEVE TOP-K CHUNKS

def retrieve_chunks(query, index, metadata):
    model = SentenceTransformer(EMBEDDING_MODEL)

    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, TOP_K)

    retrieved = []
    for idx in indices[0]:
        retrieved.append(metadata[idx])

    return retrieved


# CALL OLLAMA (LLM)

def ask_llm(question, retrieved_chunks):
    print("\n========== RETRIEVED CONTEXT ==========\n")

    context_text = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"[Chunk {i}] {chunk['source']} | Page {chunk['page']}")
        print(chunk["text"])
        print("-" * 80)

        context_text += (
            f"\nSource: {chunk['source']} (Page {chunk['page']})\n"
            f"{chunk['text']}\n"
        )

    full_prompt = f"""
{SYSTEM_PROMPT}

Context:
{context_text}

Question:
{question}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9
            }
        }
    )

    return response.json()["response"]



# MAIN LOOP

def main():
    print("\nPreparing Jeremy Miner RAG system...\n")

    chunks = load_and_chunk_pdfs()
    index, metadata = build_or_load_index(chunks)

    print("System ready.\n")

    while True:
        question = input("Ask a question (type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break

        retrieved_chunks = retrieve_chunks(question, index, metadata)
        answer = ask_llm(question, retrieved_chunks)

        print("\n========== ANSWER ==========\n")
        print(answer)
        print("\n============================\n")


if __name__ == "__main__":
    main()
