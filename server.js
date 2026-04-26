import fs from "fs";
import express from "express";
import dotenv from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";

dotenv.config();

// Load system prompt
const systemMessage = fs.readFileSync("./system_prompt.md", "utf8");

const app = express();
app.use(express.json());

/* ------------------------------
   Pinecone Setup
------------------------------ */

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});

const pineconeIndex = pinecone.index(process.env.PINECONE_INDEX_NAME);

/* ------------------------------
   Embeddings
------------------------------ */

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "models/gemini-embedding-2"
});

/* ------------------------------
   Vector Store (Initialized Once)
------------------------------ */

let vectorStore;

async function initVectorStore() {
  vectorStore = await PineconeStore.fromExistingIndex(
    embeddings,
    { pineconeIndex }
  );
  console.log("✅ Vector store initialized");
}

/* ------------------------------
   Gemini LLM
------------------------------ */

const llm = new ChatGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "gemini-2.5-flash",
  temperature: 0
});

/* ------------------------------
   Hybrid Keyword Scoring
------------------------------ */

function keywordScore(text, question) {
  const qWords = question.toLowerCase().split(/\s+/);
  const tWords = text.toLowerCase();

  let score = 0;
  qWords.forEach(word => {
    if (tWords.includes(word)) score++;
  });

  return score;
}

/* ------------------------------
   RAG Endpoint
------------------------------ */

app.post("/extractRAG", async (req, res) => {

  try {

    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Question required" });
    }

    /* Step 1: Semantic Retrieval (MMR) */

    const docs = await vectorStore.maxMarginalRelevanceSearch(
      question,
      {
        k: 12,
        fetchK: 25,
        lambda: 0.7
      }
    );

    /* Step 2: Hybrid Keyword Reranking */

    const rankedDocs = docs
      .map(doc => ({
        doc,
        score: keywordScore(doc.pageContent, question)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 6)
      .map(d => d.doc);

    /* Step 3: Build Context */

    const context = rankedDocs
      .map(d => d.pageContent)
      .join("\n\n");

    /* Step 4: Build Prompt */

    const prompt = `
${systemMessage}

Context:
${context}

Question:
${question}

Answer:
`;

    /* Step 5: Generate Answer */

    const response = await llm.invoke(prompt);

    /* Step 6: Unique Sources */

    const sources = [...new Set(
      rankedDocs.map(d => d.metadata?.filename || "Unknown")
    )];

    res.json({
      answer: response.content,
      sources
    });

  } catch (error) {

    res.status(500).json({
      error: error.message
    });

  }

});

/* ------------------------------
   Start Server
------------------------------ */

async function startServer() {

  await initVectorStore();

  app.listen(3000, () => {
    console.log("🚀 RAG API running on port 3000");
  });

}

startServer();