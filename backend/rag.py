import numpy as np
from vectorstore import build_index, get_embedding
from mistral_client import get_answer
from PyPDF2 import PdfReader
from docx import Document
import ollama
import io
import cv2
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import words as nltk_words, stopwords
from nltk import download as nltk_download
import pytesseract
import tempfile
import os
import re
import string
from collections import Counter
import langdetect
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\tarun.p\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk_download('punkt')
    
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk_download('words')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk_download('stopwords')

# Create a set of English words for checking
ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())
STOPWORDS = set(stopwords.words('english'))
COMMON_WORDS = STOPWORDS.union(set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'was', 'for']))

# Initialize OCR with optimized settings
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_db_thresh=0.3,
    det_db_box_thresh=0.3,
    det_db_unclip_ratio=1.6,
    rec_batch_num=6,
)

# Create cache directory if it doesn't exist
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file_bytes):
    """Generate a hash for file bytes to use as a cache key"""
    return hashlib.md5(file_bytes).hexdigest()

def cache_result(cache_key, result):
    """Cache the OCR result for future use"""
    try:
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return True
    except Exception as e:
        print(f"Error caching result: {str(e)}")
        return False

def get_cached_result(cache_key):
    """Try to get cached OCR result"""
    try:
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error reading cache: {str(e)}")
    return None

def is_text_structured(text):
    # Use NLTK to check if text contains valid sentences
    sentences = sent_tokenize(text)
    return len(sentences) > 0 and any(len(s.split()) > 2 for s in sentences)

def preprocess_image_advanced(img):
    """Improved preprocessing specifically for text images - optimized for speed"""
    # Use a smaller set of preprocessing methods for faster processing
    preprocessed_images = []
    
    # Original image first (always try this)
    preprocessed_images.append(img.copy())
    
    # Check if image is very large and resize if needed
    h, w = img.shape[:2]
    if max(h, w) > 3000:
        scale = 3000 / max(h, w)
        img_resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        preprocessed_images.append(img_resized)
    
    # Convert to grayscale - often helps with text
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessed_images.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    
    # Only use the most effective preprocessing methods based on performance analysis
    
    # Simple binarization - good for clean scans
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    preprocessed_images.append(binary_bgr)
    
    # Adaptive thresholding - good for varying lighting
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
    adaptive_bgr = cv2.cvtColor(adaptive_threshold, cv2.COLOR_GRAY2BGR)
    preprocessed_images.append(adaptive_bgr)
    
    # Otsu's thresholding - good for bimodal images
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    preprocessed_images.append(otsu_bgr)
    
    return preprocessed_images

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF with better handling for image-only PDFs"""
    # Check cache first
    cache_key = get_file_hash(file_bytes)
    cached_result = get_cached_result(cache_key)
    if cached_result:
        print("Using cached PDF OCR result")
        return cached_result
    
    start_time = time.time()
    # Open PDF using PyMuPDF
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    reader = PdfReader(io.BytesIO(file_bytes))
    
    all_page_texts = []
    
    # Use ThreadPoolExecutor for parallel processing of pages
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all page processing tasks
        future_to_page = {
            executor.submit(process_pdf_page, i, page, reader.pages[i]): i 
            for i, page in enumerate(pdf)
        }
        
        # Process results as they come in
        for future in as_completed(future_to_page):
            page_idx = future_to_page[future]
            try:
                page_text, extraction_method = future.result()
                if page_text:
                    all_page_texts.append(f"[Page {page_idx+1}] {page_text}")
                    print(f"Page {page_idx+1}: Extracted via {extraction_method}")
                else:
                    print(f"Page {page_idx+1}: No text extracted")
            except Exception as e:
                print(f"Error processing page {page_idx+1}: {str(e)}")
    
    result = "No readable text could be extracted from this PDF." if not all_page_texts else "\n\n".join(all_page_texts)
    
    # Cache the result
    cache_result(cache_key, result)
    
    print(f"PDF processing time: {time.time() - start_time:.2f} seconds")
    return result

def process_pdf_page(page_idx, pdf_page, reader_page):
    """Process a single PDF page - extract text directly or via OCR"""
    # Try to extract text directly first
    page_text = reader_page.extract_text()
    
    if page_text and len(page_text.strip()) > 100:  # Only use if substantial text
        return page_text, "direct extraction"
    else:
        # Extract as image if direct text extraction yields little/no text
        img_text = extract_page_as_text(pdf_page)
        if img_text and len(img_text.strip()) > 10:
            return img_text, "OCR"
        else:
            return None, "failed"

def extract_page_as_text(page):
    try:
        # Determine appropriate DPI based on page size for optimal OCR
        page_width, page_height = page.rect.width, page.rect.height
        if page_width > 1000 or page_height > 1000:
            dpi = 150  # Lower DPI for large pages to save memory
        else:
            dpi = 300  # Higher DPI for smaller pages for better OCR
            
        # Get pixmap with appropriate DPI
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Handle different color formats
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unsupported channel format: {pix.n}")
        
        # Check if image is too large and resize if needed
        max_side = 4000
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
        # Use optimized OCR approach
        return extract_text_from_image_optimized(img)
        
    except Exception as e:
        print(f"Error processing page: {str(e)}")
        return ""

# Other functions remain the same
def extract_text_from_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_csv(file_bytes):
    # Read CSV from bytes
    df = pd.read_csv(io.BytesIO(file_bytes))

    # Convert DataFrame to text
    text = df.to_string(index=False)
    return text

@lru_cache(maxsize=100)
def detect_language(text_hash):
    """Cached language detection to avoid repeated processing"""
    text = text_hash  # In actual use, this would be the text retrieved by hash
    if not text or len(text.strip()) < 20:
        return 'unknown'
    
    try:
        return langdetect.detect(text)
    except:
        return 'unknown'

def is_gibberish(text, threshold=0.25):
    """Check if text appears to be gibberish - optimized version"""
    if not text or not isinstance(text, str) or len(text.strip()) < 5:
        return True
        
    # Clean the text
    text = text.strip().lower()
    words = re.findall(r'\b[a-z]{2,}\b', text)
    
    if not words:
        return True
    
    # Quick check for real English words (sample a subset for speed)
    sample_size = min(len(words), 20)  # Check at most 20 words for speed
    sample_words = words[:sample_size]
    real_word_count = sum(1 for w in sample_words if w in ENGLISH_WORDS or w in COMMON_WORDS)
    word_ratio = real_word_count / max(1, len(sample_words))
    
    # If sample has enough real words, it's probably not gibberish
    if word_ratio > 0.5:
        return False
    
    # For questionable text, do a more thorough check
    # Check for repetitive patterns
    char_counts = Counter(text[:100])  # Only examine beginning of text for speed
    most_common_char, most_common_count = char_counts.most_common(1)[0]
    char_repetition_ratio = most_common_count / max(1, len(text[:100]))
    
    # Calculate gibberish score more efficiently
    gibberish_score = (1 - word_ratio) * 0.7 + char_repetition_ratio * 0.3
    
    return gibberish_score > threshold

def clean_ocr_text(text):
    """Clean common OCR artifacts - optimized version"""
    if not text:
        return text
        
    # Most important replacements for speed
    text = re.sub(r'\b[A-Z]{5,}\b', ' ', text)  # Remove all-caps gibberish words
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[\r\n]+', '\n', text)       # Normalize line breaks
    text = re.sub(r'\s+', ' ', text)           # Normalize whitespace
    
    return text.strip()

def process_tesseract_image(img, config=''):
    """Apply Tesseract OCR to an image with the given configuration"""
    try:
        # Convert to RGB as Tesseract expects RGB
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply Tesseract
        text = pytesseract.image_to_string(img_rgb, config=config)
        
        # Clean and return the text
        return clean_ocr_text(text)
    except Exception as e:
        print(f"Tesseract error with config '{config}': {str(e)}")
        return ""

def extract_text_from_image_optimized(img_or_bytes):
    """Optimized OCR function with smart preprocessing selection"""
    start_time = time.time()
    
    # Convert bytes to image if needed
    if isinstance(img_or_bytes, bytes):
        # Check cache first for image bytes
        cache_key = get_file_hash(img_or_bytes)
        cached_result = get_cached_result(cache_key)
        if cached_result:
            print("Using cached image OCR result")
            return cached_result
            
        img_array = np.frombuffer(img_or_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return "Error: Unable to decode image."
    else:
        img = img_or_bytes
        # Generate cache key for image data
        cache_key = hashlib.md5(img.tobytes()).hexdigest()
        cached_result = get_cached_result(cache_key)
        if cached_result:
            print("Using cached image OCR result")
            return cached_result
    
    # Get reduced set of preprocessed versions - focus on ones that work best
    preprocessed_images = preprocess_image_advanced(img)
    
    # Store all results for comparison
    all_results = []
    
    # Define most effective configs based on analysis
    # Use fewer configs for speed
    tesseract_configs = [
        '--oem 3 --psm 6 -l eng',  # Default - Assume a single block of text
        '--oem 3 --psm 3 -l eng',  # Auto page segmentation
        '--oem 1 --psm 6 -l eng',  # Legacy engine for some cases
    ]
    
    # For speed, we'll process multiple image+config combinations in parallel
    ocr_tasks = []
    
    # Create a flat list of (image_idx, config) pairs to process
    for idx, img_version in enumerate(preprocessed_images):
        for config in tesseract_configs:
            ocr_tasks.append((img_version, config, idx))
    
    # Process OCR tasks in parallel
    with ThreadPoolExecutor(max_workers=min(len(ocr_tasks), os.cpu_count() * 2)) as executor:
        # Map each task to a future
        future_to_task = {
            executor.submit(process_ocr_task, img_version, config, idx): (idx, config) 
            for img_version, config, idx in ocr_tasks
        }
        
        # Process results as they complete
        for future in as_completed(future_to_task):
            idx, config = future_to_task[future]
            try:
                result = future.result()
                if result and result['score'] > 0:
                    all_results.append(result)
            except Exception as e:
                print(f"Error in OCR task {idx}, {config[:10]}: {str(e)}")
    
    # Try PaddleOCR on the original image as a fallback
    # Only if we haven't found good results yet
    if not all_results:
        try:
            paddle_result = ocr.ocr(img)
            if paddle_result and paddle_result[0]:
                extracted_text = []
                for line in paddle_result:
                    for word_info in line:
                        if (word_info and len(word_info) >= 2 and 
                            isinstance(word_info[1], tuple) and len(word_info[1]) >= 2):
                            text = word_info[1][0]
                            confidence = word_info[1][1]
                            if confidence > 0.4:  # Lower confidence threshold
                                extracted_text.append(text)
                
                text = " ".join(extracted_text)
                text = clean_ocr_text(text)
                
                if text and len(text) >= 10 and not is_gibberish(text):
                    score = calculate_text_quality(text)
                    if score > 0:
                        all_results.append({
                            'text': text,
                            'score': score,
                            'method': 'paddle'
                        })
        except Exception as e:
            print(f"PaddleOCR error: {str(e)}")
    
    # Early return if no good results found
    if not all_results:
        result = "No readable text detected in image."
    else:
        # Sort by score and pick the best
        all_results.sort(key=lambda x: x['score'], reverse=True)
        result = all_results[0]['text']
    
    # Cache the result
    cache_result(cache_key, result)
    
    print(f"Image OCR processing time: {time.time() - start_time:.2f} seconds")
    return result

def process_ocr_task(img, config, idx):
    """Process a single OCR task for parallel execution"""
    text = process_tesseract_image(img, config)
    
    # Skip if text is too short
    if not text or len(text) < 10:
        return None
        
    # Skip if text appears to be gibberish
    if is_gibberish(text):
        return None
    
    # Calculate quality score
    score = calculate_text_quality(text)
    
    if score > 0:
        return {
            'text': text,
            'score': score,
            'method': f'tesseract-{idx}-{config[:10]}'
        }
    return None

def calculate_text_quality(text):
    """Calculate a quality score for the extracted text - optimized version"""
    if not text or not isinstance(text, str):
        return 0
    
    # Break down into components
    words = text.split()  # Faster than word_tokenize for our purposes
    word_count = len(words)
    
    if word_count == 0:
        return 0
        
    # Count real words (use sampling for speed)
    sample_size = min(word_count, 30)  # Check up to 30 words
    sample = words[:sample_size]
    real_word_count = sum(1 for w in sample if w.lower() in ENGLISH_WORDS or w.lower() in COMMON_WORDS)
    real_word_ratio = real_word_count / sample_size
    
    # Simplified calculation for speed
    score = (
        word_count * 1.0 +                # More words is good
        real_word_ratio * 100.0           # Higher ratio of real words is important
    )
    
    return max(0, score)

def extract_text_from_image(file_bytes):
    """Use our improved OCR function"""
    return extract_text_from_image_optimized(file_bytes)

def chunk_text(text, page_number=None):
    chunk_size = 1000
    chunk_overlap = 200
    chunks = []

    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = {
            "text": text[i:i + chunk_size],
            "page_number": page_number
        }
        chunks.append(chunk)

    return chunks

def process_documents(documents, filenames):
    start_time = time.time()
    all_chunks = []

    # Process documents in parallel where possible
    with ThreadPoolExecutor(max_workers=min(len(documents), os.cpu_count())) as executor:
        # Map each document to a future
        future_to_doc = {
            executor.submit(process_single_document, content, filename): (content, filename) 
            for content, filename in zip(documents, filenames)
        }
        
        # Process results as they complete
        for future in as_completed(future_to_doc):
            content, filename = future_to_doc[future]
            try:
                chunks = future.result()
                if chunks:
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Ensure only string texts go to build_index
    chunk_texts = [chunk["text"] for chunk in all_chunks if isinstance(chunk["text"], str)]

    # Sanity check
    assert all(isinstance(text, str) for text in chunk_texts), "Non-string found in chunk_texts"

    index = build_index(chunk_texts)
    
    print(f"Total document processing time: {time.time() - start_time:.2f} seconds")
    return all_chunks, index

def process_single_document(content, filename):
    """Process a single document for parallel execution"""
    print(f"Processing file: {filename}")
    chunks = []

    try:
        if filename.endswith(".pdf"):
            full_text = extract_text_from_pdf(content)
            chunks = chunk_text(full_text)

        elif filename.endswith(".docx"):
            full_text = extract_text_from_docx(content)
            chunks = chunk_text(full_text)

        elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            print(f"Processing as image: {filename}")
            text = extract_text_from_image(content)
            print(f"Extracted text length: {len(text)}")
            print(f"First 100 chars: {text[:100]}...")
            chunks = chunk_text(text)

        elif filename.endswith(".csv"):
            text = extract_text_from_csv(content)
            chunks = chunk_text(text)

        else:  # assume .txt
            text = content.decode('utf-8')
            chunks = chunk_text(text)
            
        return chunks
    except Exception as e:
        print(f"Error in process_single_document for {filename}: {str(e)}")
        return []

def answer_question(index, chunks, question):
    q_embedding = get_embedding(question)
    scores = np.dot(index["embeddings"], q_embedding)
    top_k = scores.argsort()[-1:][::-1]  # Use only top-1 to pinpoint page

    best_chunk = chunks[top_k[0]]
    context = best_chunk["text"]
    page_number = best_chunk["page_number"]

    answer = get_answer(context, question)

    return {
        "answer": answer,
        "page_number": page_number
    }

def is_text_valid(text):
    if not text or not text.strip():
        return False
        
    # Use our gibberish detector
    if is_gibberish(text):
        return False
    
    # Check if text has a reasonable character distribution
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count / max(1, len(text)) < 0.3:  # At least 30% should be letters
        return False
        
    return True

def summarize_chunks(chunks):
    context = "\n".join(chunk["text"] for chunk in chunks)
    if not is_text_valid(context):
        return ["No meaningful text found to summarize."]

    prompt = f"Summarize the following document:\n\n{context}"

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "You are a summarizer."},
            {"role": "user", "content": prompt}
        ]
    )
    summary_text = response["message"]["content"]
    
    # Ensure we return a list for frontend
    return [point.strip() for point in summary_text.split('\n') if point.strip()]