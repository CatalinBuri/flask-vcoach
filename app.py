import os
import re
import json
import traceback
import httpx
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from io import BytesIO
from docx import Document

# Configurare aplicație Flask
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

# Memorie temporară în RAM pentru sesiune
MEMORY = {
    "cv_text": "",
    "job_description": "",
    "interview_history": []
}

# ==========================================
# 1. FUNCȚII AUXILIARE PENTRU MISTRAL
# ==========================================

def call_mistral_api(
    prompt: str,
    model: str = "mistral-small-latest",
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
    if not MISTRAL_API_KEY:
        return ""

    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Ești un asistent AI profesionist specializat în resurse umane, optimizare CV-uri și interviuri."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Eroare API Mistral direct: {type(e).__name__} - {str(e)}", flush=True)
        return ""

# ==========================================
# 2. INIȚIALIZARE CLIENȚI AI
# ==========================================

gemini_client = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini ready | model:", MODEL_NAME, flush=True)
    except Exception as e:
        print(f"⚠️ Gemini nu a putut fi inițializat: {e}", flush=True)

groq_client = None
USE_GROQ = False
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        USE_GROQ = True
        print("✅ Groq ready", flush=True)
    except Exception as e:
        print(f"⚠️ Groq nu a putut fi inițializat: {e}", flush=True)

mistral_client = None
USE_MISTRAL = False
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

if MISTRAL_API_KEY:
    try:
        try:
            from mistralai import Mistral
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)
            USE_MISTRAL = True
            print("✅ Mistral ready (SDK)", flush=True)
        except ImportError:
            try:
                from mistralai.client import MistralClient
                mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
                USE_MISTRAL = True
                print("✅ Mistral ready (SDK Legacy)", flush=True)
            except ImportError:
                test_response = call_mistral_api("Spune 'test'")
                if "test" in test_response.lower():
                    USE_MISTRAL = True
                    print("✅ Mistral ready (Direct API)", flush=True)
    except Exception as e:
        print(f"⚠️ Mistral nu a putut fi inițializat: {e}", flush=True)

# ==========================================
# 3. FUNCȚII AUXILIARE & UTILS
# ==========================================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def remove_consecutive_duplicates(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'\b([a-zA-ZăâîșțĂÂÎȘȚ]+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?i)\b([A-Zăâîșț\s]+)(\r?\n\1\b)+', r'\1', cleaned)
    return cleaned

def enforce_factuality_and_language(target_lang: str) -> str:
    if target_lang == 'ro':
        lang_instruction = "REGULĂ LINGVISTICĂ STRICTĂ: Tot outputul trebuie să fie exclusiv în limba ROMÂNĂ."
    elif target_lang == 'en':
        lang_instruction = "STRICT LANGUAGE RULE: The entire output must be exclusively in ENGLISH."
    else:
        lang_instruction = "LANGUAGE RULE: Detect and use a single unified language consistently throughout."

    anti_hallucination = "REGULĂ ANTI-HALUCINAȚIE CRUCIALĂ: Este STRICT INTERZIS să inventezi date sau experiențe care nu există în textul original."
    return f"{anti_hallucination}\n{lang_instruction}"

def safe_json(raw_text: str) -> dict:
    if not raw_text:
        return {}
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}

def api_response(payload=None, error=None, code=200):
    if error:
        response = jsonify({
            "status": "error",
            "success": False,
            "ok": False,
            "message": error,
            "error": error
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, code

    base_response = {
        "status": "success",
        "success": True,
        "ok": True,
        "code": 200
    }

    if isinstance(payload, dict):
        base_response["data"] = payload
        base_response.update(payload)
        response = jsonify(base_response)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, code

    base_response["data"] = payload if payload is not None else {}
    response = jsonify(base_response)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response, code

def gemini_text(prompt: str) -> str:
    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"⚠️ Eroare Gemini: {e}. Fallback pe Groq...", flush=True)

    if USE_GROQ and groq_client:
        try:
            res = groq_client.with_options(max_retries=0).chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Ești un asistent AI profesionist specializat în resurse umane."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096,
                timeout=12.0
            )
            if res and res.choices and res.choices[0].message.content:
                return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Eroare Groq: {e}. Fallback pe Mistral...", flush=True)

    if USE_MISTRAL:
        try:
            if mistral_client and hasattr(mistral_client, "chat"):
                res = mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[
                        {"role": "system", "content": "Ești un asistent AI profesionist."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4096
                )
                if res and res.choices and res.choices[0].message.content:
                    return res.choices[0].message.content.strip()
            
            direct_res = call_mistral_api(prompt)
            if direct_res:
                return direct_res
        except Exception as e:
            print(f"❌ Eroare Mistral: {e}", flush=True)
            return call_mistral_api(prompt)

    return ""

# ==========================================
# 4. RUTELE API
# ==========================================

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({
        "status": "online",
        "success": True,
        "service": "vCoach AI API"
    }), 200

@app.route("/ping", methods=["GET", "HEAD"])
def ping():
    return "OK", 200

@app.route("/upload-cv", methods=["POST", "OPTIONS"], endpoint="upload_cv_root")
@app.route("/api/upload-cv", methods=["POST", "OPTIONS"], endpoint="upload_cv_api")
@cross_origin()
def upload_cv():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        text_content = ""
        if "file" in request.files:
            file = request.files["file"]
            filename = file.filename.lower()
            if filename.endswith(".pdf"):
                import fitz
                doc = fitz.open(stream=file.read(), filetype="pdf")
                for page in doc:
                    text_content += page.get_text()
            else:
                text_content = file.read().decode("utf-8", errors="ignore")
        elif request.is_json:
            data = request.get_json(force=True, silent=True) or {}
            text_content = data.get("cv_text", "")

        cleaned = clean_text(text_content)
        if not cleaned:
            return api_response(error="Nu s-a putut extrage text.", code=400)

        MEMORY["cv_text"] = cleaned
        return api_response(payload={"message": "Succes", "cv_text": cleaned})
    except Exception as e:
        return api_response(error=str(e), code=500)

@app.route("/analyze-cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_root")
@app.route("/api/cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_api")
@cross_origin()
def analyze_cv_quality():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv = clean_text(data.get("cv_text") or MEMORY.get("cv_text") or "")
        job = clean_text(data.get("job_description") or MEMORY.get("job_description") or "")
        target_lang = data.get("target_language") or "ro"

        if not cv:
            return api_response(error="CV lipsă.", code=400)

        prompt = f"""
{enforce_factuality_and_language(target_lang)}
Analizează CV-ul în raport cu jobul. Răspunde EXCLUSIV JSON:
{{
  "clarity_score": 8,
  "relevance_score": 7,
  "structure_score": 8,
  "matched_ats_keywords": ["Cuvant"],
  "missing_ats_keywords": ["Lipsește"],
  "concrete_improvements": ["Sfat"],
  "suggested_rephrasings": ["Exemplu"]
}}
CV: {cv}
JOB: {job}
"""
        parsed = safe_json(gemini_text(prompt))
        return api_response(payload={
            "clarity_score": parsed.get("clarity_score", 8),
            "relevance_score": parsed.get("relevance_score", 7),
            "structure_score": parsed.get("structure_score", 8),
            "ats_keywords": parsed.get("matched_ats_keywords", []),
            "missing_keywords": parsed.get("missing_ats_keywords", []),
            "concrete_improvements": parsed.get("concrete_improvements", []),
            "suggested_rephrasings": parsed.get("suggested_rephrasings", [])
        })
    except Exception as e:
        return api_response(error=str(e), code=500)

@app.route("/interview-question", methods=["POST", "OPTIONS"], endpoint="interview_question_root")
@app.route("/api/interview-question", methods=["POST", "OPTIONS"], endpoint="interview_question_api")
@cross_origin()
def interview_question():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = f"""
{enforce_factuality_and_language(data.get("target_language", "ro"))}
Răspunde DOAR JSON:
{{
  "feedback": "...",
  "score": 8,
  "next_question": "..."
}}
Răspuns candidat: "{data.get("user_answer", "")}"
"""
        parsed = safe_json(gemini_text(prompt)) or {}
        return api_response(payload={
            "feedback": parsed.get("feedback", ""),
            "score": parsed.get("score", 7),
            "next_question": parsed.get("next_question", "")
        })
    except Exception as e:
        return api_response(error=str(e), code=500)

@app.route("/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_root")
@app.route("/api/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_api")
@cross_origin()
def rephrase():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_text = data.get("text") or MEMORY.get("cv_text") or ""
        prompt = f"""
{enforce_factuality_and_language(data.get("target_language", "ro"))}
Rescrie CV-ul. Răspunde DOAR JSON:
{{
  "improved_text": "..."
}}
CV: {cv_text}
"""
        parsed = safe_json(gemini_text(prompt))
        improved = parsed.get("improved_text", cv_text)
        return api_response(payload={"improved_text": improved})
    except Exception as e:
        return api_response(error=str(e), code=500)

@app.route("/export-docx", methods=["POST", "OPTIONS"], endpoint="export_docx_root")
@app.route("/api/export-docx", methods=["POST", "OPTIONS"], endpoint="export_docx_api")
@cross_origin()
def export_docx():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        text_content = data.get("text") or MEMORY.get("cv_text") or ""

        if not text_content:
            return api_response(error="Text lipsă.", code=400)

        doc = Document()
        for line in text_content.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == stripped.upper() and len(stripped) > 3 and "|" not in stripped:
                doc.add_heading(stripped, level=2)
            elif stripped.startswith("* ") or stripped.startswith("- "):
                doc.add_paragraph(stripped[2:], style='List Bullet')
            else:
                doc.add_paragraph(stripped)

        file_stream = BytesIO()
        doc.save(file_stream)
        file_stream.seek(0)

        response = send_file(
            file_stream,
            as_attachment=True,
            download_name="CV_Optimizat.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    except Exception as e:
        return api_response(error=str(e), code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
