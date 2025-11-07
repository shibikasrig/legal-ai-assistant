from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF
import boto3
import json
import os
from transformers import pipeline

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-v2")
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "legal-text-classifier")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------
# Initialize clients
# ------------------------------
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
# ✅ AWS clients stay here for structure (won’t run unless invoked)
sagemaker = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ✅ Initialize Hugging Face summarizer (safe & free)
hf_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(
    title="⚖️ Legal AI Assistant",
    description="AI-powered legal document analyzer using AWS Bedrock, SageMaker, Hugging Face & OpenAI fallback.",
    version="2.3"
)

# Allow Gradio or Web Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Utility: Extract text from PDF
# ------------------------------
def extract_text_from_pdf(file: UploadFile) -> str:
    """Extract readable text from PDF."""
    try:
        pdf = fitz.open(stream=file.file.read(), filetype="pdf")
        text = "\n".join(page.get_text("text") for page in pdf)
        pdf.close()
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# ------------------------------
# SageMaker: Classification
# ------------------------------
def classify_document(text: str) -> str:
    """Classify document using a SageMaker endpoint."""
    try:
        response = sagemaker.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"text": text})
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        return result.get("label", "Unclassified")
    except Exception as e:
        return f"Error (SageMaker): {str(e)}"

# ------------------------------
# Bedrock: Legal Summarization (disabled for safety)
# ------------------------------
def summarize_with_bedrock(text: str) -> str:
    """Placeholder Bedrock summarization — won't actually invoke AWS."""
    return (
        "🟡 Bedrock summarization placeholder: Integration configured but not executed "
        "(safe mode enabled — no AWS billing)."
    )

# ------------------------------
# Hugging Face: Safe Summarization
# ------------------------------
def summarize_with_huggingface(text: str) -> str:
    """Summarize legal/compliance content using Hugging Face (local/free)."""
    try:
        summary = hf_summarizer(text[:2000], max_length=150, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error (Hugging Face): {str(e)}"

# ------------------------------
# OpenAI: Fallback Summarization
# ------------------------------
def summarize_with_openai(text: str) -> str:
    """Summarize using OpenAI (Groq-compatible)."""
    if not client:
        return "OpenAI API key not set."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional legal AI assistant."},
                {"role": "user", "content": f"Summarize and extract legal or compliance clauses:\n\n{text[:4000]}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error (OpenAI): {str(e)}"

# ------------------------------
# Request Model
# ------------------------------
class TextRequest(BaseModel):
    text: str
    summarize: bool = False
    use_openai: bool = False
    use_huggingface: bool = True  # ✅ Added default safe option

# ------------------------------
# Routes
# ------------------------------
@app.get("/")
def root():
    return {"message": "✅ Legal AI Assistant backend running safely with Hugging Face integration."}

@app.post("/analyze_text")
async def analyze_text(payload: TextRequest):
    """Analyze text input — classify and summarize."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text provided.")

    classification = classify_document(text)

    # Safe summarization logic
    if payload.summarize:
        if payload.use_huggingface:
            summary = summarize_with_huggingface(text)
        elif payload.use_openai:
            summary = summarize_with_openai(text)
        else:
            summary = summarize_with_bedrock(text)
    else:
        summary = "Summarization skipped."

    return {"classification": classification, "summary": summary}

@app.post("/analyze_pdf")
async def analyze_document(file: UploadFile, summarize: bool = Form(False), use_openai: bool = Form(False), use_huggingface: bool = Form(True)):
    """Analyze uploaded PDF — classify and summarize."""
    text = extract_text_from_pdf(file)
    if "Error" in text:
        return JSONResponse(content={"error": text}, status_code=400)

    classification = classify_document(text)

    # Safe summarization logic
    if summarize:
        if use_huggingface:
            summary = summarize_with_huggingface(text)
        elif use_openai:
            summary = summarize_with_openai(text)
        else:
            summary = summarize_with_bedrock(text)
    else:
        summary = "Summarization skipped."

    return {"classification": classification, "summary": summary}
