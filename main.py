from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import fitz  # PyMuPDF
import io
import gdown
import os
import tempfile
import boto3
from botocore.exceptions import ClientError
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# AWS Textract configuration from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Default to Mumbai

# Validate that credentials are provided
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError("AWS credentials not found in environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")

# Initialize AWS Textract client
try:
    textract_client = boto3.client(
        'textract',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
except Exception as e:
    raise ValueError(f"Failed to initialize AWS Textract client: {str(e)}")

def pdf_to_images(pdf_document):
    """Convert PDF pages to images for Textract"""
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # Convert page to image with high DPI for better OCR
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        images.append(img_data)
        pix = None
    return images

def textract_extract_text(image_bytes):
    """Extract text using AWS Textract"""
    try:
        response = textract_client.detect_document_text(
            Document={'Bytes': image_bytes}
        )
        
        text_lines = []
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                text_lines.append(block['Text'])
        
        return "\n".join(text_lines)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"AWS Textract ClientError [{error_code}]: {e}")
        
        # Handle specific error cases
        if error_code == 'UnrecognizedClientException':
            raise HTTPException(status_code=401, detail="Invalid AWS credentials. Please check your access key and secret key.")
        elif error_code == 'AccessDeniedException':
            raise HTTPException(status_code=403, detail="AWS credentials don't have permission to access Textract.")
        elif error_code == 'InvalidParameterException':
            raise HTTPException(status_code=400, detail="Invalid parameters sent to Textract.")
        else:
            raise HTTPException(status_code=500, detail=f"AWS Textract error: {error_code}")
    except Exception as e:
        print(f"Textract extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

def textract_extract_text_with_structure(image_bytes):
    """Extract text with structure information using AWS Textract"""
    try:
        response = textract_client.detect_document_text(
            Document={'Bytes': image_bytes}
        )
        
        # Organize text by structure
        lines = []
        words = []
        
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                lines.append({
                    'text': block['Text'],
                    'confidence': block['Confidence'],
                    'geometry': block['Geometry']
                })
            elif block['BlockType'] == 'WORD':
                words.append({
                    'text': block['Text'],
                    'confidence': block['Confidence'],
                    'geometry': block['Geometry']
                })
        
        return {
            'raw_text': '\n'.join([line['text'] for line in lines]),
            'lines': lines,
            'words': words
        }
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"AWS Textract ClientError [{error_code}]: {e}")
        
        if error_code == 'UnrecognizedClientException':
            raise HTTPException(status_code=401, detail="Invalid AWS credentials.")
        elif error_code == 'AccessDeniedException':
            raise HTTPException(status_code=403, detail="Insufficient AWS permissions for Textract.")
        else:
            raise HTTPException(status_code=500, detail=f"AWS Textract error: {error_code}")
    except Exception as e:
        print(f"Textract extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

@app.post("/extract-text/", response_class=PlainTextResponse)
async def extract_text(file: UploadFile = File(...)):
    """Extract text from PDF using AWS Textract"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        
        all_text = []
        
        # Convert PDF pages to images and process with Textract
        page_images = pdf_to_images(doc)
        for page_num, page_img_bytes in enumerate(page_images, 1):
            textract_text = textract_extract_text(page_img_bytes)
            if textract_text.strip():
                all_text.append(f"--- Page {page_num} ---\n{textract_text.strip()}")
        
        doc.close()
        return "\n\n".join(all_text)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

@app.post("/extract-text-with-fallback/", response_class=PlainTextResponse)
async def extract_text_with_fallback(file: UploadFile = File(...)):
    """Extract text using Textract with PyMuPDF fallback for text-heavy PDFs"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        
        all_text = []
        
        # First try to extract regular text using PyMuPDF (faster for text-based PDFs)
        pymupdf_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            regular_text = page.get_text()
            if regular_text.strip():
                pymupdf_text.append(regular_text.strip())
        
        # If PyMuPDF found substantial text, use it; otherwise use Textract
        if pymupdf_text and any(len(text) > 100 for text in pymupdf_text):
            # Text-based PDF - use PyMuPDF results
            for page_num, text in enumerate(pymupdf_text, 1):
                all_text.append(f"--- Page {page_num} (Text Layer) ---\n{text}")
        else:
            # Image-based or scanned PDF - use Textract
            page_images = pdf_to_images(doc)
            for page_num, page_img_bytes in enumerate(page_images, 1):
                textract_text = textract_extract_text(page_img_bytes)
                if textract_text.strip():
                    all_text.append(f"--- Page {page_num} (OCR) ---\n{textract_text.strip()}")
        
        doc.close()
        return "\n\n".join(all_text)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

@app.post("/extract-text-from-drive/", response_class=PlainTextResponse)
async def extract_text_from_drive(drive_url: str = Form(...)):
    """Download PDF from Google Drive and extract text using AWS Textract"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "downloaded.pdf")
            gdown.download(url=drive_url, output=output_path, quiet=True, fuzzy=True)
            
            doc = fitz.open(output_path)
            all_text = []
            
            # Convert PDF pages to images and process with Textract
            page_images = pdf_to_images(doc)
            for page_num, page_img_bytes in enumerate(page_images, 1):
                textract_text = textract_extract_text(page_img_bytes)
                if textract_text.strip():
                    all_text.append(f"--- Page {page_num} ---\n{textract_text.strip()}")
            
            doc.close()
            return "\n\n".join(all_text)
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download or extract text: {str(e)}")

@app.post("/extract-text-json/", response_class=JSONResponse)
async def extract_text_json(file: UploadFile = File(...)):
    """Extract text from PDF using AWS Textract and return detailed JSON response"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        
        pages_data = []
        all_text_blocks = []
        
        # Process each page with Textract
        page_images = pdf_to_images(doc)
        for page_num, page_img_bytes in enumerate(page_images, 1):
            page_result = textract_extract_text_with_structure(page_img_bytes)
            
            if page_result['raw_text'].strip():
                pages_data.append({
                    'page_number': page_num,
                    'text': page_result['raw_text'].strip(),
                    'lines_count': len(page_result['lines']),
                    'words_count': len(page_result['words']),
                    'average_confidence': sum(line['confidence'] for line in page_result['lines']) / len(page_result['lines']) if page_result['lines'] else 0
                })
                all_text_blocks.append(page_result['raw_text'].strip())
        
        doc.close()
        
        return {
            "extracted_text": "\n\n".join(all_text_blocks),
            "pages": pages_data,
            "total_pages": len(pages_data),
            "extraction_method": "AWS Textract OCR",
            "text_blocks": all_text_blocks
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

@app.post("/extract-text-detailed/", response_class=JSONResponse)
async def extract_text_detailed(file: UploadFile = File(...)):
    """Extract text with detailed structure and confidence information"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        
        detailed_results = []
        
        # Process each page with detailed Textract analysis
        page_images = pdf_to_images(doc)
        for page_num, page_img_bytes in enumerate(page_images, 1):
            page_result = textract_extract_text_with_structure(page_img_bytes)
            
            detailed_results.append({
                'page_number': page_num,
                'raw_text': page_result['raw_text'],
                'lines': page_result['lines'],
                'words': page_result['words']
            })
        
        doc.close()
        
        return {
            "pages": detailed_results,
            "total_pages": len(detailed_results),
            "extraction_method": "AWS Textract with structure analysis"
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract detailed text: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "PDF Text Extraction API with AWS Textract", 
        "status": "running",
        "aws_region": AWS_REGION,
        "credentials_configured": bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY),
        "available_endpoints": {
            "/extract-text/": "Extract text using AWS Textract OCR",
            "/extract-text-with-fallback/": "Smart extraction (PyMuPDF for text PDFs, Textract for scanned PDFs)",
            "/extract-text-from-drive/": "Extract from Google Drive PDF using Textract",
            "/extract-text-json/": "JSON response with page-by-page analysis",
            "/extract-text-detailed/": "Detailed JSON with confidence scores and structure",
            "/health-textract/": "Check AWS Textract connection"
        }
    }

@app.get("/health-textract/")
async def health_textract():
    """Health check for AWS Textract connection"""
    try:
        # Create a small test image (1x1 pixel PNG)
        test_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01U\r1(\x00\x00\x00\x00IEND\xaeB`\x82'
        
        response = textract_client.detect_document_text(
            Document={'Bytes': test_image}
        )
        return {"status": "healthy", "message": "AWS Textract connection successful"}
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'UnrecognizedClientException':
            return {"status": "error", "message": "Invalid AWS credentials"}
        elif error_code == 'AccessDeniedException':
            return {"status": "error", "message": "AWS credentials lack Textract permissions"}
        elif error_code in ['InvalidParameterException', 'UnsupportedDocumentException']:
            return {"status": "healthy", "message": "AWS Textract connection working (expected parameter error with test data)"}
        else:
            return {"status": "error", "message": f"Textract connection issue: {error_code}"}
    except Exception as e:
        return {"status": "error", "message": f"Connection error: {str(e)}"}

# Uncomment below to run with: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)