import os
from fastapi.responses import RedirectResponse
import pdf2image
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  # Replace with the address of your React app
    # Add any other allowed origins here
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict HTTP methods if needed (e.g., ["GET", "POST"])
    allow_headers=["*"],  # You can restrict headers if needed
)
def pdf_to_text(pdf):
    pages = pdf2image.convert_from_bytes(pdf, poppler_path="poppler/bin")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    text_results = []
    for page_image in pages:
        img_np = np.array(page_image)
        result = ocr.ocr(img_np, cls=True)
        result = result[0]  # Assuming the first result is sufficient
        text = '\n'.join([line[1][0] for line in result])  # Join text lines into a single string
        text_results.append(text)
    return text_results

@app.get("/")
async def root():
    # redirect to /docs
    redirect_url = "/docs"
    return RedirectResponse(url=redirect_url)

@app.post("/upload/")
async def upload_file(pdf_file: UploadFile):
    if not os.path.exists("output"):
        os.makedirs("output", exist_ok=True)
    
    # Read PDF file content
    pdf_content = await pdf_file.read()
    # Convert PDF to images
    text_results = pdf_to_text(pdf_content)
    print(text_results)
    # Perform OCR on images
    return {"message": text_results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
