import fs from "fs";
import dotenv from "dotenv";

import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

dotenv.config();

// Initialize Pinecone
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});

const index = pc.Index(process.env.PINECONE_INDEX);

// Embedding model
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY
});

// Text splitter (IMPORTANT for good retrieval)
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100
});

async function ingest() {

  const documents = [];

  const files = fs.readdirSync("./documents");

  for (const file of files) {

    const content = fs.readFileSync(`./documents/${file}`, "utf8");

    const chunks = await splitter.splitText(content);

    chunks.forEach((chunk, index) => {

      documents.push({
        pageContent: chunk,
        metadata: {
          filename: file,
          chunk: index
        }
      });

    });

  }

  await PineconeStore.fromDocuments(
    documents,
    embeddings,
    { pineconeIndex: index }
  );

  console.log("Documents successfully uploaded to Pinecone");

}

ingest();