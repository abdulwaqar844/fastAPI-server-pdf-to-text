import os
from fastapi.responses import RedirectResponse
import pdf2image
from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

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
def pdf_to_image(pdf):
    if not os.path.exists('trial'):
        os.makedirs('output', exist_ok=True)
    #convert pdf to pages as images and name then in order
    pages = pdf2image.convert_from_bytes(pdf,poppler_path="poppler/bin")
    for i in range(len(pages)):
        pages[i].save('output\page'+str(i)+'.jpg', 'JPEG')

def ocr_images(images, ocr):
    ocr_results = []
    folder = "output"

    result = ocr.ocr(f"{folder}/{images}", cls=True)
    result = result[0]
    txts = [line[1][0] for line in result]

    return txts

@app.get("/")
async def root():
    # redirect to /docs
    redirect_url = "/docs"
    return RedirectResponse(url=redirect_url)

@app.post("/upload/")
async def upload_file(pdf_file: UploadFile):
    if not os.path.exists("output"):
        os.makedirs("output", exist_ok=True)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Read PDF file content
    pdf_content = await pdf_file.read()

    # Convert PDF to images
    pdf_to_image(pdf_content)

    # Perform OCR on images
    results = []
    for img in os.listdir("output"):
        ocr_results = ocr_images(img, ocr)
        print(ocr_results)
        # text_file_path = "output/" + img[:-4] + ".txt"
        # with open(text_file_path, 'w') as f:
        results.append(ocr_results)
        # for item in ocr_results:
            # f.write("%s\n" % item)
        os.remove(os.path.join("output", img))
    return {"message": results}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
