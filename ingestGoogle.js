import fs from "fs";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import {
  getDocument,
  GlobalWorkerOptions,
} from "pdfjs-dist/legacy/build/pdf.mjs";
import Tesseract from "tesseract.js";
import sharp from "sharp";

import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

GlobalWorkerOptions.workerSrc = path.join(
  __dirname,
  "node_modules/pdfjs-dist/legacy/build/pdf.worker.mjs",
);

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 800,
  chunkOverlap: 150,
});

// ✅ Helper to delay between batches
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ✅ Upload documents in small batches with delay
async function uploadInBatches(
  documents,
  embeddings,
  index,
  batchSize = 50,
  delayMs = 2000,
) {
  for (let i = 0; i < documents.length; i += batchSize) {
    const batch = documents.slice(i, i + batchSize);
    console.log(
      `   📤 Uploading batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(documents.length / batchSize)} (${batch.length} chunks)...`,
    );

    for (const doc of batch) {

      const vector = await embeddings.embedQuery(doc.pageContent)

      if (!vector || vector.length !== 3072) {
        console.log("⚠️ Skipping invalid embedding chunk")
        continue
      }

      await index.upsert([
        {
          id: `${doc.metadata.filename}-${doc.metadata.chunk}`,
          values: vector,
          metadata: doc.metadata
        }
      ])

    }

    if (i + batchSize < documents.length) {
      console.log(`   ⏳ Waiting ${delayMs / 1000}s to avoid rate limiting...`);
      await sleep(delayMs);
    }
  }
}

async function renderPageToImage(page) {
  const viewport = page.getViewport({ scale: 2.0 });
  const width = Math.floor(viewport.width);
  const height = Math.floor(viewport.height);

  const pixelBuffer = new Uint8ClampedArray(width * height * 4);

  const canvasContext = {
    canvas: { width, height },
    drawImage: () => {},
    fillRect: () => {},
    putImageData: () => {},
    getImageData: () => ({ data: pixelBuffer }),
    restore: () => {},
    save: () => {},
    transform: () => {},
    resetTransform: () => {},
    scale: () => {},
    translate: () => {},
    beginPath: () => {},
    stroke: () => {},
    fill: () => {},
    moveTo: () => {},
    lineTo: () => {},
    closePath: () => {},
    clip: () => {},
    rect: () => {},
    setTransform: () => {},
    createImageData: (w, h) => ({
      data: new Uint8ClampedArray(w * h * 4),
      width: w,
      height: h,
    }),
  };

  await page.render({ canvasContext, viewport }).promise;

  const pngBuffer = await sharp(Buffer.from(pixelBuffer), {
    raw: { width, height, channels: 4 },
  })
    .png()
    .toBuffer();

  return pngBuffer;
}

async function ocrImage(imageBuffer) {

  const processed = await sharp(imageBuffer)
    .grayscale()
    .normalize()
    .resize({ width: 2000 })
    .toBuffer()

  const { data } = await Tesseract.recognize(processed, "eng", {
    logger: () => {},
  })

  return data.text
}

async function extractText(filepath, filename) {
  if (filename.endsWith(".pdf")) {
    const buffer = fs.readFileSync(filepath);
    const doc = await getDocument({
      data: new Uint8Array(buffer),
      standardFontDataUrl: path.join(
        __dirname,
        "node_modules/pdfjs-dist/standard_fonts/",
      ),
    }).promise;

    console.log(`   📑 PDF has ${doc.numPages} pages`);

    const pageTexts = [];

    for (let i = 0; i < doc.numPages; i++) {
      const page = await doc.getPage(i + 1);
      const content = await page.getTextContent();
      const text = content.items
        .map((item) => item.str)
        .join(" ")
        .trim();

      if (text.length > 50) {
        console.log(
          `   ✅ Page ${i + 1}: extracted ${text.length} chars (text-based)`,
        );
        pageTexts.push(text);
      } else {
        console.log(`   🔍 Page ${i + 1}: no embedded text, running OCR...`);
        try {
          const imageBuffer = await renderPageToImage(page);
          const ocrText = await ocrImage(imageBuffer);
          if (ocrText.trim().length > 0) {
            console.log(
              `   ✅ Page ${i + 1}: OCR extracted ${ocrText.length} chars`,
            );
            pageTexts.push(ocrText);
          } else {
            console.log(`   ⚠️  Page ${i + 1}: OCR returned no text`);
          }
        } catch (err) {
          console.log(`   ⚠️  Page ${i + 1}: OCR failed — ${err.message}`);
        }
      }
    }

    const fullText = pageTexts.join("\n");

    if (!fullText || fullText.trim().length === 0) {
      throw new Error(
        "PDF extracted no text after both text extraction and OCR",
      );
    }

    return fullText;
  }

  return fs.readFileSync(filepath, "utf8");
}

async function ingest() {
  try {
    const indexName = process.env.PINECONE_INDEX_NAME;

    if (!indexName) {
      throw new Error("PINECONE_INDEX_NAME missing in .env");
    }

    const index = pinecone.index(indexName);

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      model: "models/gemini-embedding-2",
    });

    // Sanity check
    const testVector = await embeddings.embedQuery("test");
    console.log("🔍 Embedding dimension:", testVector.length);

    if (testVector.length === 0) {
      throw new Error(
        "Embedding sanity check failed — dimension is 0. Check your GOOGLE_API_KEY.",
      );
    }

    const documents = [];
    const files = fs.readdirSync("./documents");

    console.log(`📂 Found ${files.length} files\n`);

    for (const file of files) {
      if (file.startsWith(".")) {
        console.log(`⏭️  Skipping system file: ${file}`);
        continue;
      }

      console.log(`📄 Reading ${file}`);

      const filepath = `./documents/${file}`;
      let content = "";
      const stats = fs.statSync(filepath);
      const fileModifiedTime = stats.mtime.getTime();

      try {
        content = await extractText(filepath, file);
      } catch (err) {
        console.log(`⚠️  Failed to extract text from ${file}: ${err.message}`);
        continue;
      }

      if (!content || content.trim().length === 0) {
        console.log(`⚠️  Skipping empty file: ${file}`);
        continue;
      }

      console.log(
        `   ✅ Extracted ${content.length} total characters from ${file}\n`,
      );

      const chunks = await splitter.splitText(content);

      chunks.forEach((chunk, i) => {

        const cleanChunk = chunk.trim()

        // Skip empty or very small chunks
        if (!cleanChunk || cleanChunk.length < 20) {
            return
        }

        documents.push({
        pageContent: cleanChunk,
        metadata: {
            filename: file,
            chunk: i,
            lastModified: fileModifiedTime
            }
        })

    });
    }

    if (documents.length === 0) {
      throw new Error("No valid documents to upload.");
    }

    console.log(`\n🚀 Uploading ${documents.length} chunks in batches...\n`);

    // ✅ Batched upload with rate limit protection
    await uploadInBatches(documents, embeddings, index);

    console.log("\n✅ Ingestion successful");
  } catch (error) {
    console.error("❌ Gemini ingest error:", error.message);
  }
}

ingest();