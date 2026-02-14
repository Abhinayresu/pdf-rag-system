# Professional-Grade RAG Pipeline: Sales-Centric Consultative AI

A high-performance Retrieval-Augmented Generation (RAG) system engineered to emulate complex consultative sales methodologies. This project implements a local-first, vertically integrated RAG pipeline optimized for high-precision knowledge retrieval and strict behavioral grounding.

## 🎯 Problem Statement

Traditional LLMs often struggle with "hallucination" and lack the specific psychological nuance required for high-stakes consultative sales. Standard RAG implementations frequently fail to maintain a specific "voice" or methodology (e.g., Jeremy Miner’s NEPQ logic) and often generalize beyond the provided corpus. This project aims to solve these challenges by implementing a zero-generalization, strictly grounded retrieval system that adheres to a specific psychological framework.

## 🏗️ System Architecture

The pipeline follows a modular architecture designed for low-latency retrieval and precise response generation:

1.  **Ingestion Layer**: Asynchronous PDF parsing using `PyPDFLoader`.
2.  **Processing Layer**: Recursive character splitting with token-level awareness (`tiktoken`) to preserve semantic boundaries.
3.  **Embedding Layer**: Dense vector representation using the `BAAI/bge-m3` model.
4.  **Retrieval Layer**: Vector similarity search via a `FAISS` FlatIP index.
5.  **Inference Layer**: Orchestrated via LangChain, utilizing a quantized `Llama 3.2` model via Ollama with strict system-level prompting.

## 🧠 Design Decisions & Rationale

### 1. Vector Embedding: `BAAI/bge-m3`
*   **Why**: Unlike standard models, BGE-M3 supports multi-functionality (dense/sparse retrieval) and is optimized for multilingual/multi-length sequences. It provides superior retrieval recall (R@K) for technical corpora.
*   **Implementation**: Utilizes $L_2$ normalization on vectors to allow for efficient Inner Product (FlatIP) similarity search, effectively computing cosine similarity in Euclidean space.

### 2. Indexing: FAISS (Facebook AI Similarity Search)
*   **Why**: FAISS allows for highly optimized search performance. While this implementation uses a linear `FlatIP` index for maximum precision, the architecture is designed to scale to `IVFFlat` or `HNSW` should the corpus grow beyond $10^5$ chunks.

### 3. Local LLM: Llama 3.2 via Ollama
*   **Why**: Data privacy and latency control. Running `Llama 3.2` locally ensures that sensitive training materials never leave the infrastructure while maintaining high-quality reasoning capabilities.
*   **Prompt Engineering**: Implemented a "Strict-Context" system prompt to prevent model drift and enforce the specific consultative tone.

## ⚡ Performance & Optimization

*   **Chunk Strategy**: $400$ token chunks with $20\%$ ($80$ tokens) overlap. This overlap ensures semantic continuity across the vector space.
*   **Vector Caching**: Embeddings and FAISS indices are serialized via `pickle`, reducing cold-boot time from minutes to milliseconds.
*   **Grounding Enforcement**: Low temperature ($0.3$) and specific $p$-sampling ($0.9$) settings were selected to minimize creative variance and maximize factual recall.

## 🚧 Limitations

*   **Static Context**: The current implementation does not support real-time corpus updates without a full re-index.
*   **Linear Search**: `IndexFlatIP` is $O(N)$ complexity. While acceptable for current datasets, it will require partitioning for massive scale.
*   **Memory Overhead**: Sentence-transformers require significant VRAM/RAM relative to lightweight API-based solutions.

## 🚀 Future Improvements

*   **Hybrid Search**: Implementing BM25 keyword search alongside dense vector retrieval to capture specific sales terminology.
*   **Reranking**: Adding a Cross-Encoder (e.g., `BGE-Reranker`) stage to post-process the top-K retrieved chunks for higher precision.
*   **Graph-RAG**: Transitioning to a knowledge-graph-based retrieval to better understand the relationships between different sales modules.

## 🎙️ Interview Talking Points

*   **Chunking Logic**: "I used `RecursiveCharacterTextSplitter` with `tiktoken` to ensure chunks don't cut off mid-thought, while staying within the model's optimal context window."
*   **Similarity Choice**: "FAISS FlatIP with $L_2$ normalized embeddings was chosen because it's the gold standard for precision when retrieval accuracy is more critical than search speed."
*   **Grounding**: "I enforced a 'No Knowledge Outside Context' rule in the system prompt to ensure the AI remains a specialist rather than a generalist."

---

### 📋 Technical Requirements
```text
langchain, sentence-transformers (BGE-M3), faiss-cpu, pypdf, tiktoken, numpy, Ollama (Llama 3.2)
```
