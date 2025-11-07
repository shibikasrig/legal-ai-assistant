import os
import json
import time
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

def summarize_with_bedrock(text: str):
    """
    SAFE MOCK: Summarize and highlight key compliance risks using Amazon Bedrock (Claude-v2 or Titan).

    ⚠️ No actual Bedrock API call is made when USE_BEDROCK=false (default).
    Set USE_BEDROCK=true in your .env if you ever want to enable real AWS calls.
    """

    USE_BEDROCK = os.getenv("USE_BEDROCK", "false").lower() == "true"

    if not USE_BEDROCK:
        # ✅ Simulated response — No AWS cost or network call
        time.sleep(1)  # mimic API latency
        return {
            "summary": (
                "🧠 [Simulated Bedrock Summary]\n"
                "This is a locally generated mock response to demonstrate AWS Bedrock integration.\n"
                "No actual AWS API or billing occurred.\n\n"
                f"Analyzed text snippet:\n{text[:250]}..."
            ),
            "model_used": "anthropic.claude-v2 (mock)",
            "safe_mode": True
        }

    # -----------------------------------------------------------
    # 🔥 Real Bedrock invocation (only runs if USE_BEDROCK=true)
    # -----------------------------------------------------------
    import boto3

    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        prompt = f"Summarize and highlight key compliance risks in the following document:\n\n{text}"

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
            "note": "No billing occurred — Bedrock integration attempted safely.",
            "safe_mode": False
        }

# ✅ Local test
if __name__ == "__main__":
    sample_text = (
        "The company migrated its data to the cloud but did not enable encryption. "
        "Manual access permissions increase risk of human error. No breach response plan exists."
    )

    result = summarize_with_bedrock(sample_text)
    print(json.dumps(result, indent=2))
