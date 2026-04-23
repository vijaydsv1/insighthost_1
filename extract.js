import dotenv from "dotenv"
import { Pinecone } from "@pinecone-database/pinecone"
import { OpenAIEmbeddings } from "@langchain/openai"
import { PineconeStore } from "@langchain/pinecone"

dotenv.config()

// Get question from command line
const question = process.argv.slice(2).join(" ").trim()

if (!question) {
  console.log("❌ Please provide a question")
  console.log('Example: node extract.js "What services does Accion provide?"')
  process.exit(1)
}

console.log(`\n💬 Question: ${question}\n`)

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
})

const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX)

// Initialize embeddings
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY
})

async function search() {

  try {

    console.log("🔎 Searching Pinecone...\n")

    const vectorStore = await PineconeStore.fromExistingIndex(
      embeddings,
      { pineconeIndex }
    )

    const results = await vectorStore.similaritySearch(question, 3)

    if (!results || results.length === 0) {
      console.log("❌ No results found.")
      return
    }

    console.log("📄 Top Results:\n")

    results.forEach((doc, i) => {

      console.log(`Result ${i + 1}`)
      console.log("------------------------------------")

      const text = doc.pageContent?.substring(0, 250) || "No content available"
      const source = doc.metadata?.filename || "Unknown source"

      console.log(text)
      console.log(`\n📚 Citation: ${source}\n`)

    })

  } catch (error) {

    console.error("❌ Error during search:", error.message)

  }

}

search()