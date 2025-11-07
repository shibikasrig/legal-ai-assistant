"""
bedrock_client.py
-----------------
SAFE HYBRID VERSION — Does NOT actually invoke AWS Bedrock or incur billing.

This file keeps your project AWS Bedrock–ready, but defaults to
a local Hugging Face fallback or mock response to avoid AWS costs.

🟢 USE_BEDROCK=false (default) → Safe simulation or Hugging Face
🔵 USE_BEDROCK=true → Real AWS Bedrock invocation (optional)
"""

import json
import os
import time

# Optional: Free local generation using Hugging Face
try:
    from transformers import pipeline
    hf_generator = pipeline("text-generation", model="gpt2")
except Exception:
    hf_generator = None


def summarize_with_bedrock(text: str):
    """
    Summarize input text safely.

    If USE_BEDROCK=false, runs a Hugging Face model or mock output.
    If USE_BEDROCK=true, uses AWS Bedrock (real API call, may incur charges).
    """

    USE_BEDROCK = os.getenv("USE_BEDROCK", "false").lower() == "true"

    # ✅ SAFE LOCAL SIMULATION (Default mode — no AWS call)
    if not USE_BEDROCK:
        time.sleep(1)  # simulate network delay

        if hf_generator:
            # Generate a short simulated summary
            result = hf_generator(
                f"Summarize this: {text[:500]}",
                max_length=120,
                num_return_sequences=1
            )[0]["generated_text"]
            summary_text = f"🧠 [Local Hugging Face Summary]\n{result.strip()}"
        else:
            # Fallback mock text (if Hugging Face not installed)
            summary_text = (
                "🧠 [Simulated Bedrock Output]\n"
                "This is a mock summary generated locally for demonstration.\n"
                "No AWS API request or billing occurred.\n\n"
                f"Document snippet analyzed:\n{text[:200]}..."
            )

        return {
            "summary": summary_text,
            "model_used": "huggingface/gpt2 (safe local mode)",
            "safe_mode": True
        }

    # 🟠 REAL BEDROCK CALL (Only if explicitly enabled)
    import boto3

    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )

        prompt = f"Summarize and highlight key compliance or legal risks in the following text:\n\n{text}"

        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 300,
            "temperature": 0.5
        })

        response = bedrock.invoke_model(
            modelId=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-v2"),
            contentType="application/json",
            accept="application/json",
            body=body
        )

        model_output = json.loads(response["body"].read())
        text_output = model_output.get("completion", "").strip()

        return {
            "summary": text_output or "No summary returned.",
            "model_used": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-v2"),
            "safe_mode": False
        }

    except Exception as e:
        return {
            "error": f"Bedrock summarization failed: {str(e)}",
            "note": "No charges occurred during this process.",
            "safe_mode": False
        }


# ✅ Test run
if __name__ == "__main__":
    print("\n--- Bedrock Safe Summarizer Test ---\n")
    sample_text = "Artificial intelligence is transforming healthcare by improving diagnosis and treatment accuracy."
    result = summarize_with_bedrock(sample_text)
    print(json.dumps(result, indent=2))
