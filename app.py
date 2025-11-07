import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import requests
import fitz  # PyMuPDF for PDF extraction
import os

# ----------------------------------
# Load environment variables
# ----------------------------------
load_dotenv()

# ---------------------------
# 1️⃣ OpenAI (Groq-compatible)
# ---------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ---------------------------
# 2️⃣ Backend API connection
# ---------------------------
API_URL = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000")

# ---------------------------
# 3️⃣ Legal AI Q&A (Groq/OpenAI)
# ---------------------------
def legal_ai_assistant(query):
    """AI-powered legal Q&A assistant using Groq-compatible model."""
    if not query.strip():
        return "⚠️ Please enter a legal or policy question."

    try:
        prompt = f"""
        You are a professional legal assistant with expertise in law, compliance, and policy.
        Provide accurate, concise, and neutral guidance.
        If the question is not about law or policy, politely decline to answer.

        Question: {query}
        """

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert legal AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"🚨 Error: {str(e)}"

# ---------------------------
# 4️⃣ Text Analyzer (connects to FastAPI)
# ---------------------------
def analyze_text_input(text):
    """Send text to backend FastAPI (Bedrock mock or Hugging Face)."""
    if not text.strip():
        return "⚠️ Please enter or upload some legal text."

    try:
        response = requests.post(f"{API_URL}/analyze_text", json={"text": text, "summarize": True})
        if response.status_code == 200:
            data = response.json()
            summary = data.get("summary", "No summary available.")
            classification = data.get("classification", "Unclassified Document")
            return f"### 🧾 Summary\n{summary}\n\n### 🏷️ Classification\n**{classification}**"
        else:
            return f"❌ Backend Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"🚨 Connection Error: Could not connect to backend.\n\n**Details:** {str(e)}"

# ---------------------------
# 5️⃣ PDF Upload Analyzer
# ---------------------------
def analyze_pdf(file):
    """Extract text from uploaded PDF and analyze."""
    try:
        with fitz.open(file.name) as doc:
            text = "\n".join(page.get_text("text") for page in doc)
        if not text.strip():
            return "⚠️ No readable text found in PDF."
        return analyze_text_input(text)
    except Exception as e:
        return f"⚠️ Failed to process PDF: {str(e)}"

# ---------------------------
# 6️⃣ Gradio Tabs Setup
# ---------------------------

qa_tab = gr.Interface(
    fn=legal_ai_assistant,
    inputs=gr.Textbox(
        lines=4,
        label="💬 Ask your legal question",
        placeholder="e.g. Can my employer terminate me without notice?"
    ),
    outputs=gr.Textbox(label="🧠 AI Response"),
    title="⚖️ Legal AI Assistant (Groq Model)",
    description="Ask any legal or compliance question — powered by Llama 3.1 via Groq-compatible API.",
    theme="soft",
)

text_tab = gr.Interface(
    fn=analyze_text_input,
    inputs=gr.Textbox(
        lines=12,
        placeholder="📜 Paste your legal or policy text here...",
        label="✏️ Enter Legal or Policy Text"
    ),
    outputs=gr.Markdown(label="AI Summary & Classification"),
    theme="soft",
)

pdf_tab = gr.Interface(
    fn=analyze_pdf,
    inputs=gr.File(label="📄 Upload Legal Document (PDF)"),
    outputs=gr.Markdown(label="AI Summary & Classification"),
    theme="soft",
)

# Combine all tabs
app = gr.TabbedInterface(
    [qa_tab, text_tab, pdf_tab],
    tab_names=["💬 Legal Q&A", "📝 Text Input", "📁 PDF Upload"],
    title="⚖️ Legal AI Assistant Dashboard"
)

# ---------------------------
# 7️⃣ Launch
# ---------------------------
if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
