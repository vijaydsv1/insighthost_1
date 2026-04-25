# InsightHost Pinecone Retrieval (LangChain JS)

This project implements a **Retrieval Augmented Generation (RAG) pipeline** using **LangChain (JavaScript), Google Gemini embeddings, and Pinecone vector database**.

The system ingests documents (including **PDFs with OCR support**), stores embeddings in Pinecone, and retrieves relevant information using natural language queries.

---

# Features

- Document ingestion into Pinecone
- PDF text extraction
- OCR support for image-based PDFs
- Text chunking using LangChain
- Gemini embeddings for vector generation
- Semantic search using Pinecone
- Natural language question answering
- Source citations for retrieved content
- Duplicate-safe ingestion
- Support for multiple documents

---

# Project Structure

```
insighthost_1
│
├── ingestGoogle.js        # Document ingestion pipeline
├── extractGoogle.js       # Query and answer generation
│
├── package.json
├── .env                   # API keys (not committed)
│
└── documents
    ├── Accion Labs Innovation Summit 2026 Synopsis.pdf
    └── other_documents.pdf
```

---

# Technologies Used

- LangChain (JavaScript)
- Google Gemini API
- Pinecone Vector Database
- Node.js
- PDF.js (PDF parsing)
- Tesseract.js (OCR for image PDFs)
- Sharp (image preprocessing for OCR)

---

# Setup

## 1 Install Dependencies

```
npm install
```

If OCR support is required:

```
npm install pdfjs-dist tesseract.js sharp
```

---

# Environment Variables

Create a `.env` file in the root directory.

```
GOOGLE_API_KEY=your_google_ai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
```

---

# Document Ingestion

Run the ingestion script:

```
node ingestGoogle.js
```

This process will:

1. Read documents from the `documents/` folder
2. Extract text from PDFs
3. Apply OCR for image-based pages
4. Split text into chunks
5. Generate Gemini embeddings
6. Upload vectors to Pinecone

Example output:

```
📂 Found 1 files
📄 Reading Accion Labs Innovation Summit 2026 Synopsis.pdf
📑 PDF has 37 pages
🚀 Uploading 125 chunks in batches...
```

---

# Query the System

Run:

```
node extractGoogle.js "What are the four risks discussed in the summit?"
```

Example Output:

```
Answer:
The summit discusses four key risks including strategic misalignment,
technology adoption challenges, talent shortages, and market volatility.

Sources:
[Source 1] Accion Labs Innovation Summit 2026 Synopsis.pdf (chunk 28)
```

---

# How the System Works

```
Documents (PDF/TXT)
        ↓
Text Extraction + OCR
        ↓
Chunking (LangChain)
        ↓
Gemini Embeddings
        ↓
Pinecone Vector Database
        ↓
Semantic Search
        ↓
Gemini LLM Answer Generation
        ↓
Citations Returned
```

---

# Chunking Strategy

Documents are split using:

```
chunkSize: 800
chunkOverlap: 150
```

This improves retrieval quality for long PDF documents by preserving contextual information between chunks.

---

# Duplicate Handling

The ingestion pipeline generates deterministic vector IDs:

```
filename-chunkNumber
```

This ensures:

- Re-ingesting the same document **updates vectors**
- Duplicate vectors are **not created**
- Updated documents **replace old vectors**

---

# Future Improvements

Possible enhancements:

- Gemini reranking for improved retrieval accuracy
- Streaming responses
- Web interface for querying
- Multi-document filtering
- Hybrid search (vector + keyword)

---

# Author

Vijay Kumar

AI / ML Engineer  
LangChain • RAG • Vector Databases • LLM Systems