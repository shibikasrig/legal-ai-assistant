import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# Environment Setup
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ----------------------------
# Load RAG Embeddings
# ----------------------------
# Dynamically locate the 'vectorstore' folder relative to this script
BASE_DIR = os.path.dirname(__file__)
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")

# Auto-create if missing
if not os.path.exists(VECTOR_DIR):
    os.makedirs(VECTOR_DIR)
    print(f"⚠️ Created missing folder: {VECTOR_DIR}")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load all FAISS indexes from vectorstore
vectorstores = []
for folder in os.listdir(VECTOR_DIR):
    folder_path = os.path.join(VECTOR_DIR, folder)
    if os.path.isdir(folder_path):
        try:
            store = FAISS.load_local(folder_path, embedding_model, allow_dangerous_deserialization=True)
            vectorstores.append(store)
            print(f"✅ Loaded RAG index: {folder_path}")
        except Exception as e:
            print(f"⚠️ Skipping {folder_path}: {str(e)}")

def search_knowledge_base(query, top_k=3):
    """Retrieve top chunks from all FAISS indexes."""
    results = []
    for store in vectorstores:
        docs = store.similarity_search(query, k=top_k)
        results.extend(docs)
    return results

# ----------------------------
# RAG-Powered Legal Assistant
# ----------------------------
def legal_rag_assistant(query):
    if not query.strip():
        return "⚠️ Please enter a legal or policy question."

    docs = search_knowledge_base(query)
    if not docs:
        context = "No matching legal data found."
    else:
        context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a professional Indian legal research assistant.
    Use the following legal context to answer concisely and cite relevant Acts or Sections if possible.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert legal AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"🚨 Error generating answer: {str(e)}"

# ----------------------------
# Gradio UI
# ----------------------------
app = gr.Interface(
    fn=legal_rag_assistant,
    inputs=gr.Textbox(lines=5, placeholder="Ask about IT Act, Gambling Act, etc.", label="💬 Ask a Legal Question"),
    outputs=gr.Markdown(label="🧠 AI Legal Answer"),
    title="⚖️ Legal RAG Assistant",
    description="AI-powered assistant that answers from real Indian legal documents using Retrieval-Augmented Generation (RAG)."
)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7861)
