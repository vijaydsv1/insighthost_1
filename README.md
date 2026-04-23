# InsightHost Pinecone Retrieval (LangChain JS)

This project demonstrates **document ingestion and retrieval using LangChain, OpenAI embeddings, and Pinecone vector database**.

## Features

- Ingest documents into Pinecone
- Chunk text using LangChain
- Store filename as metadata
- Query Pinecone using natural language
- Return relevant results with citations

## Project Structure
insighthost_1
│
├── ingest.js # Script to ingest documents into Pinecone
├── extract.js # Script to query Pinecone
├── package.json
├── .env # API keys (not committed)
└── documents
└── accion_services.txt


## Setup

Install dependencies:


npm install


Create `.env` file:


OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=your_index_name


## Ingest Documents


node ingest.js


This will:
- Read files from `documents/`
- Split text into chunks
- Generate embeddings
- Upload vectors to Pinecone

## Query Pinecone


node extract.js "What services does Accion provide?"


Example Output:

Result 1

Accion Labs provides services in digital engineering, cloud transformation...

Citation: accion_services.txt


## Technologies Used

- LangChain (JavaScript)
- OpenAI Embeddings
- Pinecone Vector Database
- Node.js