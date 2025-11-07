import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import bcrypt
import json
import os
import fitz
import requests
import boto3
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# Paths & Constants
# ---------------------------
USERS_FILE = "users.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "rag_knowledge", "vectorstore")

# ---------------------------
# User management helpers
# ---------------------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def register_user(username, email, password, contact, role):
    if not all([username, email, password, contact, role]):
        return "⚠️ Please fill all fields."

    if not contact.isdigit() or len(contact) != 10:
        return "⚠️ Enter a valid 10-digit contact number."

    users = load_users()
    if username in users:
        return "❌ Username already exists."

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = {"email": email, "password": hashed_pw, "contact": contact, "role": role}
    save_users(users)
    return "✅ Registration successful! Please go to Login tab."

def login_user(email, password, role):
    users = load_users()
    for username, data in users.items():
        if data["email"] == email and data["role"].lower() == role.lower():
            if bcrypt.checkpw(password.encode(), data["password"].encode()):
                return True, f"✅ Welcome {username} ({role.title()})!"
            else:
                return False, "❌ Incorrect password."
    return False, "❌ Invalid email or role."

# ---------------------------
# AI Functions
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "facebook/bart-large-cnn")

def summarize_with_huggingface(text):
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {"inputs": text, "parameters": {"max_length": 300}}
        response = requests.post(f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}", headers=headers, json=payload)
        data = response.json()
        return {"summary": data[0]["summary_text"], "model_used": HUGGINGFACE_MODEL}
    except Exception as e:
        return {"error": str(e)}

def legal_ai_assistant(query):
    if not query.strip():
        return "⚠️ Please enter a legal question."
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": "You are a legal assistant."},
                      {"role": "user", "content": query}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"🚨 Error: {str(e)}"

def analyze_pdf(file):
    try:
        with fitz.open(file.name) as doc:
            text = "\n".join(p.get_text("text") for p in doc)
        result = summarize_with_huggingface(text)
        return f"### Summary\n{result['summary']}\n\n🧠 **Model:** {result['model_used']}"
    except Exception as e:
        return f"⚠️ PDF Error: {str(e)}"

def rag_query(question):
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if not os.path.exists(VECTORSTORE_PATH):
            return "⚠️ No RAG data found."
        stores = []
        for folder in os.listdir(VECTORSTORE_PATH):
            fp = os.path.join(VECTORSTORE_PATH, folder)
            if os.path.isdir(fp):
                store = FAISS.load_local(fp, embedding_model, allow_dangerous_deserialization=True)
                stores.append(store)
        docs = []
        for s in stores:
            docs.extend(s.similarity_search(question, k=3))
        if not docs:
            return "⚠️ No relevant info."
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Based on this legal context:\n{context}\n\nQuestion: {question}"
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": "Legal assistant using Indian legal data."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        return f"### ⚖️ Legal Answer\n{answer}\n\n📚 Sources:\n{context[:1000]}..."
    except Exception as e:
        return f"🚨 RAG Error: {str(e)}"

# ---------------------------
# Legal Assistant Tabs
# ---------------------------
def launch_legal_app():
    qa_tab = gr.Interface(fn=legal_ai_assistant, inputs=gr.Textbox(lines=6, label="💬 Ask Legal Question"),
                          outputs=gr.Textbox(lines=10, label="🧠 Response"), title="Legal Q&A")

    pdf_tab = gr.Interface(fn=analyze_pdf,
                           inputs=gr.File(label="📄 Upload Legal PDF"),
                           outputs=gr.Markdown(), title="PDF Analyzer")

    rag_tab = gr.Interface(fn=rag_query,
                           inputs=gr.Textbox(lines=6, label="⚖️ Ask based on Indian Acts"),
                           outputs=gr.Markdown(), title="Legal RAG Search")

    return gr.TabbedInterface(
        [qa_tab, pdf_tab, rag_tab],
        tab_names=["💬 Q&A", "📁 PDF", "📚 RAG"],
        title="⚖️ Legal AI Assistant"
    )

# ---------------------------
# Login + Registration Interface
# ---------------------------
def login_interface():
    legal_ui = launch_legal_app()
    with gr.Blocks(theme="soft", css=".gradio-container {max-width: 750px; margin:auto;}") as app:
        gr.Markdown("<h1 style='text-align:center;'>🔐 Legal AI Assistant Login</h1>")

        login_section = gr.Group(visible=True)
        legal_section = gr.Group(visible=False)

        with login_section:
            with gr.Tab("Login"):
                email = gr.Textbox(label="📧 Email", placeholder="Enter your email")
                password = gr.Textbox(label="🔑 Password", placeholder="Enter password", type="password")
                show_pw = gr.Checkbox(label="👁 Show Password", value=False)
                remember = gr.Checkbox(label="💾 Remember Me")
                role = gr.Dropdown(["Lawyer", "Student", "Judge", "Legal Researcher", "Other"], label="👔 Professional Role")
                login_btn = gr.Button("🚪 Login", variant="primary")
                login_msg = gr.Markdown()

                # Toggle password visibility
                def toggle_pw(show):
                    return gr.update(type="text" if show else "password")
                show_pw.change(toggle_pw, show_pw, password)

            with gr.Tab("Register"):
                reg_user = gr.Textbox(label="👤 Username", placeholder="Choose username")
                reg_mail = gr.Textbox(label="📧 Email", placeholder="Enter your email")
                reg_pass = gr.Textbox(label="🔑 Password", placeholder="Create password", type="password")
                reg_contact = gr.Textbox(label="📞 Contact Number", placeholder="Enter 10-digit mobile number")
                reg_role = gr.Dropdown(["Lawyer", "Student", "Judge", "Legal Researcher", "Other"], label="👔 Professional Role")
                reg_btn = gr.Button("📝 Register", variant="primary")
                reg_msg = gr.Markdown()

        with legal_section:
            with gr.Row():
                gr.Markdown("### ⚖️ Welcome to Legal Assistant Dashboard")
                logout_btn = gr.Button("🔒 Logout", variant="secondary")
            legal_ui.render()

        # Event handlers
        def handle_login(email, password, role, remember):
            success, msg = login_user(email, password, role)
            if success:
                if remember:
                    with open("remembered_user.json", "w") as f:
                        json.dump({"email": email, "role": role}, f)
                return gr.update(visible=False), gr.update(visible=True), msg
            return gr.update(), gr.update(), msg

        def handle_register(u, e, p, c, r):
            return register_user(u, e, p, c, r)

        def handle_logout():
            if os.path.exists("remembered_user.json"):
                os.remove("remembered_user.json")
            return gr.update(visible=True), gr.update(visible=False)

        login_btn.click(handle_login, [email, password, role, remember], [login_section, legal_section, login_msg])
        reg_btn.click(handle_register, [reg_user, reg_mail, reg_pass, reg_contact, reg_role], reg_msg)
        logout_btn.click(handle_logout, None, [login_section, legal_section])

    return app

# ---------------------------
# Launch App
# ---------------------------
if __name__ == "__main__":
    app = login_interface()
    app.launch(server_name="127.0.0.1", server_port=7860)
