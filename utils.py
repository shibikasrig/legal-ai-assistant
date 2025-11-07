import fitz  # PyMuPDF for PDF

async def parse_document(file):
    if file.filename.endswith(".pdf"):
        pdf = fitz.open(stream=await file.read(), filetype="pdf")
        text = " ".join(page.get_text() for page in pdf)
    else:
        text = (await file.read()).decode("utf-8")
    return text
