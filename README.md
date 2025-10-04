# AI Empower Career Coach

## Overview
AI Empower Career Coach is a Flask-based web application designed to analyze and summarize candidate resumes using large language models (LLMs) with fallback strategies.
It leverages LangChain integrations with OpenAI, Perplexity, and HuggingFace embeddings to provide detailed career coaching insights from uploaded PDF resumes.

---

## Features
- Upload PDF resumes for automated extraction and analysis
- Intelligent resume summaries including:
  - Career Objective
  - Skills and Expertise
  - Professional Experience
  - Educational Background
  - Notable Achievements
- Fallback mechanism between OpenAI and Perplexity LLM providers to ensure reliable responses
- QA interface to ask questions based on resume content
- Uses FAISS vector store for document similarity search and chunked text handling
- Modular LLM provider setup with extensible support for multiple models

---

## Installation

1. Clone the repository:
   git clone https://github.com/ksantosh7/ai-empower-career-coach.git
   cd ai-empower-career-coach
2. Create and activate a Python virtual environment:
   conda create -p venv python == 3.12.7 -y
   conda activate venv/
   pip install -r requirements.txt

---

## Configuration

Set up API keys in your environment variables:

- `OPENAI_API_KEY` — your OpenAI API key
- `PPLX_API_KEY` — your Perplexity API key

---

## Usage


3. Upload your resume in PDF format on the home page to receive a detailed AI-generated career summary.

4. Use the "Ask" feature to ask questions related to the resume content.

---

## Code Highlights

- **PDF Text Extraction:** Uses PyPDF2 to extract text content from uploaded PDF resumes.
- **Text Splitting and Embeddings:** Splits resume text for semantic similarity search using HuggingFace embeddings and FAISS vector store.
- **LLM Integration:** Supports switching between OpenAI and Perplexity with fallback for improved uptime.
- **Prompt Template:** Customized prompt crafted for resume summary tailored to career coaching insights.
- **Flask Routes:** `/` for the upload page, `/upload` to handle resume upload and analysis, and `/ask` for queries against the resume.

---

## Project Structure
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
├── setup.py # Package setup configuration
├── uploads/ # Folder for storing uploaded PDF resumes
└── templates/ # HTML templates for rendering web pages


---

## Author

Santosh Kumar  
Email: santosh.iitk7@gmail.com

---

## License

This project is licensed under the MIT License.
    


