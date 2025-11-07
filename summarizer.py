# summarizer.py
from transformers import pipeline

def summarize_with_huggingface(text: str):
    """
    Summarize and highlight key compliance risks using Hugging Face models.
    Safe, free, and local (no AWS billing).
    """
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        result = summarizer(
            text,
            max_length=130,
            min_length=30,
            do_sample=False
        )

        summary_text = result[0]["summary_text"]

        return {
            "summary": summary_text,
            "model_used": "facebook/bart-large-cnn (Hugging Face)"
        }

    except Exception as e:
        return {
            "error": f"Hugging Face summarization failed: {str(e)}",
            "note": "Ensure transformers and torch are installed."
        }
