import os
import requests
from load_and_split import load_all_pdfs_from_folder, split_text
from vector_store import SimpleVectorStore


OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"
CHAT_MODEL = "llama3"


def generate_answer(context, question):
    prompt = f"""
You are a question-answering assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in the documents."

Context:
{context}

Question:
{question}

Answer:
"""
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"].strip()


print("\n📂 RAG Chatbot – PDF Folder Selection\n")

# ---- Folder selection ----
while True:
    pdf_folder = input("Enter the FULL folder path containing PDFs:\n> ").strip()
    if os.path.isdir(pdf_folder):
        break
    print("❌ Invalid folder path. Try again.\n")

# ---- Ingestion ----
print("\n📄 Loading PDFs...")
raw_text = load_all_pdfs_from_folder(pdf_folder)

print("✂️ Splitting text into chunks...")
chunks = split_text(raw_text)

print(f"🧠 Building vector store ({len(chunks)} chunks)...")
vector_store = SimpleVectorStore()

for i, chunk in enumerate(chunks, 1):
    vector_store.add_text(chunk)
    if i % 20 == 0:
        print(f"   Embedded {i}/{len(chunks)} chunks")

print("\n✅ RAG system ready. Ask questions!")
print("Type 'exit' to quit.\n")

# ---- Chat loop ----
while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("👋 Exiting RAG chatbot.")
        break

    if not question:
        print("❗ Please enter a valid question.\n")
        continue

    try:
        context_chunks = vector_store.search(question)
        context = "\n\n".join(context_chunks)

        answer = generate_answer(context, question)
        print("\n🤖 Answer:\n", answer, "\n")

    except Exception as e:
        print(f"⚠️ Error: {e}\n")
