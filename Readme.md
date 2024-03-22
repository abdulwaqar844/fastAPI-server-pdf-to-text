# PdfToDocx

Convert PDF documents to DOCX format while preserving layout and formatting. This project utilizes layout analysis, table detection, and OCR techniques to ensure accurate conversion.

## System Design

### Step 1: Layout Analysis
- Utilize layout-parser with the Paddle PubLayNet dataset for layout analysis on each page.

### Step 2: Region Identification of Each page
- Identify regions as table, text, list, title, or figure.

### Step 3a: Table Conversion
- If the region is identified as a table, use table detection and table cell recognition transformers to convert the table to CSV while preserving the original formatting.

### Step 3b: Add Table to DOCX
- Incorporate the table into the DOCX document using a grid style.

### Step 4: Figure Handling
- If the region is identified as a figure, crop the image and place it on the DOCX.
- Perform OCR using angle classification to catch rotations, placing the detected text below the figure.

### Step 5: Text Block Processing
- If the region is identified as text, preprocess the block to detect text efficiently and add it to the DOCX.

### Step 6: List Handling
- If the region is identified as a list, count characters. If characters are less than 25, add a new line and append it to the DOCX.

## Usage

1. **Clone the Repo:**
   ```bash
   git clone <repository-url>
   ```

2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure PaddlePaddle is Installed:**
   Make sure PaddlePaddle is installed in your path.
   using
   ```bash
   wget https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
   pip3 install -U layoutparser-0.0.0-py3-none-any.whl
   ````
4. ***Install poppler and Tesseract:***
   ```bash
   sudo apt-get poppler-utils
   sudo apt-get tesseract-ocr
   ```
6. **Run the Script:**
   ```bash
   python app.py
   ```
   This starts a FastAPI endpoint listening at `/upload/`.

7. **Upload PDF for Conversion:**
   Send your PDF as an `UploadFile` object to the `/upload/` endpoint. The response will be a Word DOCX file.

8. **Response Format:**
   - Successful Response (Status 200): Word DOCX file.
   - Error Response (Status 500): Check for detailed error information. Added try-catch for HTTP exceptions.

## CI/CD Pipeline (GitHub Actions)

- The CI/CD pipeline is integrated using GitHub Actions.
- To set up:
  - Change the secrets: SSH key, user ID (`root`), and IP address if you want to modify the instance.

