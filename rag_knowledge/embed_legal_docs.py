import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------
# Base directory setup (use absolute path)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")

os.makedirs(VECTOR_DIR, exist_ok=True)

# ----------------------------
# Initialize Embedding Model
# ----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Process PDFs
# ----------------------------
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ Data folder not found at: {DATA_DIR}")

for file_name in os.listdir(DATA_DIR):
    if not file_name.endswith(".pdf"):
        continue

    file_path = os.path.join(DATA_DIR, file_name)
    print(f"✅ Loading: {file_name}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    print(f"📄 Total text chunks created: {len(docs)}")

    # Create and save FAISS index
    vectorstore = FAISS.from_documents(docs, embedding_model)
    save_path = os.path.join(VECTOR_DIR, f"{os.path.splitext(file_name)[0]}_index")
    vectorstore.save_local(save_path)
    print(f"💾 Saved FAISS vectorstore to {save_path}")

print("🎯 Embedding process complete!")
