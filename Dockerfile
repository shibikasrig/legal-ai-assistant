# -------- Base Image --------
FROM python:3.10-slim

# -------- Set Working Directory --------
WORKDIR /app

# -------- Copy Project Files --------
COPY . .

# -------- Install System Dependencies --------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------- Upgrade pip --------
RUN pip install --no-cache-dir --upgrade pip

# -------- Install CPU-only Torch --------
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# -------- Install Remaining Dependencies --------
RUN pip install --no-cache-dir \
    gradio boto3 openai python-dotenv transformers pypdf PyMuPDF

# -------- Expose Port --------
EXPOSE 7860

# -------- Start App --------
CMD ["python", "gradio_app.py"]
