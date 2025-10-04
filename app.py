from flask import Flask, request, render_template, redirect, url_for
import os, sys, traceback
from werkzeug.utils import secure_filename
import PyPDF2

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_perplexity import ChatPerplexity

try:
    import openai
except Exception:
    openai = None

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   
PPLX_API_KEY = os.getenv("PPLX_API_KEY")       

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#  Text splitter & embeddings 
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=200, length_function=len)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#  Prompt
resume_summary_template = """
Role: You are an AI Career Coach.

Task: Given the candidate's resume, provide a comprehensive summary that includes the following key aspects:

- Career Objective
- Skills and Expertise
- Professional Experience
- Educational Background
- Notable Achievements

Instructions:
Provide a concise summary of the resume, focusing on the candidate's skills, experience, and career trajectory. Ensure the summary is well-structured, clear, and highlights the candidate's strengths in alignment with industry standards.

Requirements:
{resume}
"""
resume_prompt = PromptTemplate(input_variables=["resume"], template=resume_summary_template)


# PDF extractor 
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# make LLM instance
def make_llm(provider: str, model_name: str):
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(model=model_name, temperature=0.2, openai_api_key=OPENAI_API_KEY)
    elif provider == "perplexity":
        if not PPLX_API_KEY:
            raise RuntimeError("PPLX_API_KEY is not set.")
        return ChatPerplexity(model=model_name, temperature=0.2, pplx_api_key=PPLX_API_KEY)
    else:
        raise ValueError("Unknown provider")


# Fallback handler 
def run_with_fallback(provider: str, candidate_models: list, invoke_fn):
    last_exception = None
    for model_name in candidate_models:
        llm = make_llm(provider, model_name)
        try:
            return invoke_fn(llm)
        except Exception as e:
            last_exception = e
            err_str = str(e).lower()

            # Case 1: invalid model--> try next
            if "model_not_found" in err_str or "does not exist" in err_str or "invalid model" in err_str:
                print(f"[fallback] Model {model_name} not found, trying next...", file=sys.stderr)
                continue

            # Case 2: insufficient quota --> switch provider
            if "insufficient_quota" in err_str and provider == "openai":
                print("[fallback] OpenAI quota exceeded, switching to Perplexity.", file=sys.stderr)
                pplx_candidates = ["sonar-pro", "sonar-reasoning-pro", "sonar-deep-research"]
                for pm in pplx_candidates:
                    try:
                        llm_pplx = make_llm("perplexity", pm)
                        return invoke_fn(llm_pplx)
                    except Exception as e2:
                        print(f"[fallback] Perplexity model {pm} failed: {e2}", file=sys.stderr)
                        last_exception = e2
                        continue
                raise last_exception

            # Case 3: other errors → stop
            raise
    raise last_exception if last_exception else RuntimeError("No models available")


# Retrieval QA with specified llm
def perform_qa_with_llm(query: str, llm):
    db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = rqa.invoke(query)
    return result["result"] if isinstance(result, dict) and "result" in result else str(result)


# Flask App 
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    provider = request.form.get("provider", "openai")
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    resume_text = extract_text_from_pdf(file_path)
    splitted_text = text_splitter.split_text(resume_text)
    vectorstore = FAISS.from_texts(splitted_text, embeddings)
    vectorstore.save_local("vector_index")

    if provider == "openai":
        candidates = ["gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo"]
    else:
        candidates = ["sonar-pro", "sonar-reasoning-pro", "sonar-deep-research"]

    def invoke_resume(llm):
        chain = resume_prompt | llm | StrOutputParser()
        return chain.invoke({"resume": resume_text})

    try:
        resume_analysis = run_with_fallback(provider, candidates, invoke_resume)
        return render_template("results.html", resume_analysis=resume_analysis)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return render_template("results.html", resume_analysis=f"Error: {e}")

@app.route("/ask", methods=["GET", "POST"])
def ask_query():
    if request.method == "POST":
        provider = request.form.get("provider", "openai")
        query = request.form["query"]

        if provider == "openai":
            candidates = ["gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo"]
        else:
            candidates = ["sonar-pro", "sonar-reasoning-pro", "sonar-deep-research"]

        def invoke_qa(llm):
            return perform_qa_with_llm(query, llm)

        try:
            result = run_with_fallback(provider, candidates, invoke_qa)
            return render_template("qa_results.html", query=query, result=result)
        except Exception as e:
            return render_template("qa_results.html", query=query, result=f"Error: {e}")

    return render_template("ask.html")

if __name__ == "__main__":
    app.run(debug=True)

