# Document QA with RAG

A Retrieval-Augmented Generation (RAG) system for document question answering, with optimized OCR capabilities for processing various document types.

## Features

- **Document Processing**: Handle PDF, DOCX, CSV, TXT, and image files (PNG, JPG, etc.)
- **Advanced OCR**: Optimized text extraction from images and scanned documents using Tesseract and PaddleOCR
- **Vector Search**: Find the most relevant document sections for any query using embeddings
- **LLM Integration**: Generate accurate answers using context from your documents (Ollama + Mistral)
- **Summary Generation**: Create concise summaries of uploaded documents
- **Parallel & Cached Processing**: Fast, scalable, and avoids redundant OCR work
- **Modern UI**: React + Tailwind CSS frontend with chat history, document upload, and summary features

## OCR Optimization

The system uses a multi-stage OCR approach:

1. **Multiple OCR Engines**: Uses both Tesseract OCR and PaddleOCR for best results
2. **Advanced Preprocessing**: Applies various image preprocessing techniques to improve OCR quality
3. **Quality Scoring**: Intelligently selects the best OCR result based on text quality metrics
4. **Caching & Fallbacks**: Caches OCR results and ensures text extraction even from challenging documents

## Installation

### Backend Requirements

```bash
pip install -r backend/requirements.txt
```

Make sure to install Tesseract OCR on your system:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt install tesseract-ocr`
- Mac: `brew install tesseract`

You also need to have [Ollama](https://ollama.com/) installed and running locally for LLM and embedding support.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Usage

1. Start the backend server:
   ```powershell
   python backend/main.py
   ```
2. Make sure Ollama is running locally (see [Ollama docs](https://ollama.com/)).
3. Open the frontend application (usually at http://localhost:5173).
4. Upload documents (PDF, DOCX, TXT, CSV, images).
5. Ask questions about the content or generate summaries.

## API Endpoints

- `POST /upload-files/` — Upload and process documents
- `POST /ask-question/` — Ask a question about the uploaded documents
- `GET /summarize/` — Get a summary of the uploaded documents

## Technical Architecture

- **Backend**: Python Flask application with OCR, document parsing, vector search, and LLM integration
- **Frontend**: React (Vite + Tailwind CSS) application for document upload, Q&A, summary, and chat history
- **Model**: Uses Ollama to serve local LLMs (Mistral) for embeddings and text generation

## Example

- Upload a PDF or image file
- Ask: "What is the main topic of this document?"
- Get an answer and see the page number where the answer was found
- Click 'Summarize' to get a bullet-point summary

---

**Note:**
- For best OCR results, ensure Tesseract and PaddleOCR dependencies are installed and configured.
- Ollama must be running locally with the required models (e.g., `mistral`, `mxbai-embed-large`).
- This project is for local/private use and does not send your documents to any external cloud service.

Screenshots:
![image](https://github.com/user-attachments/assets/daa2f976-a319-4734-a25b-bf7cf8a84b95)
![image](https://github.com/user-attachments/assets/7cf69be5-bcda-44c5-b313-abd082bf0001)
