import boto3
import json
import os

def analyze_with_sagemaker(text: str):
    """
    Sends text to a SageMaker endpoint for classification.
    Replace 'legal-text-classifier' with your real SageMaker endpoint name.
    """

    try:
        # Create SageMaker runtime client
        sagemaker = boto3.client('sagemaker-runtime', region_name='us-east-1')

        # Invoke your trained endpoint
        response = sagemaker.invoke_endpoint(
            EndpointName=os.getenv("SAGEMAKER_ENDPOINT", "legal-text-classifier"),
            ContentType="application/json",
            Body=json.dumps({"text": text})
        )

        result = json.loads(response['Body'].read().decode())

        # You can customize based on your model output
        return {
            "classification": result.get("label", "Unknown"),
            "confidence": result.get("confidence", None)
        }

    except Exception as e:
        # If SageMaker isn’t configured yet, return a fallback message
        return {
            "error": f"SageMaker call failed: {str(e)}",
            "note": "Make sure AWS credentials and endpoint name are correctly configured."
        }
