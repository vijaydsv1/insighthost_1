import dotenv from "dotenv"
import { Pinecone } from "@pinecone-database/pinecone"
import { OpenAIEmbeddings } from "@langchain/openai"
import { PineconeStore } from "@langchain/pinecone"

dotenv.config()

// Get question from command line
const question = process.argv.slice(2).join(" ")

if (!question) {
  console.log("❌ Please provide a question")
  console.log('Example: node extract.js "What services does Accion provide?"')
  process.exit()
}

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
})

const index = pinecone.Index(process.env.PINECONE_INDEX)

// Initialize embeddings
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY
})

async function search() {

  console.log("\n🔎 Searching Pinecone...\n")

  const vectorStore = await PineconeStore.fromExistingIndex(
    embeddings,
    { pineconeIndex: index }
  )

  const results = await vectorStore.similaritySearch(question, 3)

  if (results.length === 0) {
    console.log("No results found.")
    return
  }

  console.log("📄 Top Results:\n")

  results.forEach((doc, i) => {

    console.log(`Result ${i + 1}`)
    console.log("------------------------------------")

    console.log(doc.pageContent.substring(0, 250))

    console.log(`\n📚 Citation: ${doc.metadata?.filename || "Unknown source"}\n`)

  })

}

search()