import os
import re
import json
import traceback
import httpx
from flask import Flask, request, jsonify, send_file
from io import BytesIO
from docx import Document

# Configurare aplicație Flask
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Configurare CORS global manuală pentru a evita duplicatele de antet
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# Middleware de diagnosticare generală pentru toate cererile intratoare
@app.before_request
def log_incoming_requests():
    print(f"\n--- [DIAGNOZĂ GLOBALĂ] Cerere primită ---", flush=True)
    print(f"Metodă: {request.method} | Path: {request.path}", flush=True)
    print(f"Antete (Headers): {dict(request.headers)}", flush=True)
    if request.method in ["POST", "PUT"]:
        if request.is_json:
            json_data = request.get_json(force=True, silent=True)
            print(f"Payload JSON primit: {json_data}", flush=True)
        elif request.form:
            print(f"Form data primit: {request.form.to_dict()}", flush=True)
        elif request.files:
            print(f"Fișiere primite: {list(request.files.keys())}", flush=True)
        else:
            print(f"Raw data / altele (lungime): {len(request.data)} bytes", flush=True)

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
    """Fallback direct API call când SDK-ul nu este disponibil."""
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
# 2. INIȚIALIZARE CLIENȚI AI (Gemini, Groq & Mistral)
# ==========================================

# 1. Gemini Client
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

# 2. Groq Client
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

# 3. Mistral Client
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
                else:
                    print("⚠️ Mistral API key invalid sau conexiune eșuată", flush=True)
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
    lang_instruction = ""
    if target_lang == 'ro':
        lang_instruction = "REGULĂ LINGVISTICĂ STRICTĂ: Tot outputul trebuie să fie exclusiv în limba ROMÂNĂ. Nu amesteca limbi."
    elif target_lang == 'en':
        lang_instruction = "STRICT LANGUAGE RULE: The entire output must be exclusively in ENGLISH. Do not mix languages."
    else:
        lang_instruction = "LANGUAGE RULE: Detect and use a single unified language consistently throughout."

    anti_hallucination = (
        "REGULĂ ANTI-HALUCINAȚIE CRUCIALĂ: Este STRICT INTERZIS să inventezi date, publicații, "
        "companii sau experiențe care nu există în textul original furnizat de utilizator."
    )
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
        return jsonify({
            "status": "error",
            "success": False,
            "ok": False,
            "message": error,
            "error": error
        }), code

    base_response = {
        "status": "success",
        "success": True,
        "ok": True,
        "code": 200
    }

    if isinstance(payload, dict):
        base_response["data"] = payload
        base_response.update(payload)
        return jsonify(base_response), code

    base_response["data"] = payload if payload is not None else {}
    return jsonify(base_response), code

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
            print(f"⚠️ Eroare Gemini: {type(e)} - {str(e)}", flush=True)

    if USE_GROQ and groq_client:
        try:
            res = groq_client.with_options(max_retries=0).chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Ești un asistent AI specializat în resurse umane."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096,
                timeout=12.0
            )
            if res and res.choices and res.choices[0].message.content:
                return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Eroare Groq: {type(e)} - {str(e)}", flush=True)

    if USE_MISTRAL:
        try:
            if mistral_client and hasattr(mistral_client, "chat"):
                res = mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[
                        {"role": "system", "content": "Ești un asistent AI specializat în resurse umane."},
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
            print(f"❌ Eroare Mistral: {type(e)} - {str(e)}", flush=True)
            return call_mistral_api(prompt)

    return ""


# ==========================================
# 4. RUTELE API
# ==========================================

@app.route("/", methods=["GET", "HEAD", "OPTIONS"])
def index():
    if request.method == "OPTIONS":
        return api_response(code=200)
    return jsonify({
        "status": "online",
        "success": True,
        "service": "vCoach AI API",
        "gemini_active": gemini_client is not None,
        "groq_active": USE_GROQ,
        "mistral_active": USE_MISTRAL
    }), 200

@app.route("/ping", methods=["GET", "HEAD", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return "", 200
    return "OK", 200

@app.route("/upload-cv", methods=["POST", "OPTIONS"], endpoint="upload_cv_root")
@app.route("/api/upload-cv", methods=["POST", "OPTIONS"], endpoint="upload_cv_api")
def upload_cv():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        text_content = ""
        if "file" in request.files:
            file = request.files["file"]
            filename = file.filename.lower()
            if filename.endswith(".pdf"):
                try:
                    import fitz
                    doc = fitz.open(stream=file.read(), filetype="pdf")
                    for page in doc:
                        text_content += page.get_text()
                except Exception as pdf_err:
                    return api_response(error=f"Eroare citire PDF: {str(pdf_err)}", code=400)
            else:
                text_content = file.read().decode("utf-8", errors="ignore")
        elif request.is_json:
            data = request.get_json(force=True, silent=True) or {}
            text_content = data.get("cv_text", "")

        cleaned = clean_text(text_content)
        if not cleaned:
            return api_response(error="Nu s-a putut extrage text din fișierul trimis.", code=400)

        MEMORY["cv_text"] = cleaned
        return api_response(payload={"message": "CV încărcat cu succes", "length": len(cleaned), "cv_text": cleaned})
    except Exception as e:
        return api_response(error=f"Eroare la procesare: {str(e)}", code=500)

@app.route("/analyze-cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_root")
@app.route("/api/cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_api")
def analyze_cv_quality():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_description") or data.get("job_text") or MEMORY.get("job_description") or ""
        target_lang = data.get("target_language") or data.get("language") or "ro"
        
        cv = clean_text(cv_raw)
        job = clean_text(job_raw)

        if not cv:
            return api_response(error="CV lipsă.", code=400)

        MEMORY["cv_text"] = cv
        if job:
            MEMORY["job_description"] = job

        factuality_rules = enforce_factuality_and_language(target_lang)

        if job:
            prompt = f"""
{factuality_rules}
Ești un recruiter senior și expert în sisteme ATS. Analizează CV-ul în raport direct cu Descrierea Jobului.
Răspunde EXCLUSIV cu un obiect JSON valid:
{{
  "clarity_score": 8,
  "relevance_score": 7,
  "structure_score": 8,
  "matched_ats_keywords": ["Cuvant1"],
  "missing_ats_keywords": ["CuvantLipsește"],
  "concrete_improvements": ["Sfat 1"],
  "suggested_rephrasings": ["Exemplu"]
}}
CV:
{cv}
DESCRIERE JOB:
{job}
"""
        else:
            prompt = f"""
{factuality_rules}
Ești un recruiter senior. Analizează structura și calitatea acestui CV.
Răspunde EXCLUSIV cu un obiect JSON valid:
{{
  "clarity_score": 8,
  "relevance_score": 6,
  "structure_score": 8,
  "detected_skills": ["Skill1"],
  "missing_ats_keywords": ["Adăugați un Job Description"],
  "concrete_improvements": ["Recomandare 1"],
  "suggested_rephrasings": ["Exemplu"]
}}
CV:
{cv}
"""

        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res)

        improvements = [remove_consecutive_duplicates(imp) for imp in (parsed.get("concrete_improvements") or [])]
        rephrasings = [remove_consecutive_duplicates(rep) for rep in (parsed.get("suggested_rephrasings") or [])]

        payload = {
            "clarity_score": parsed.get("clarity_score", 8),
            "relevance_score": parsed.get("relevance_score", 7 if job else 5),
            "structure_score": parsed.get("structure_score", 8),
            "has_job_context": bool(job),
            "ats_keywords": parsed.get("matched_ats_keywords") or parsed.get("detected_skills") or [],
            "missing_keywords": parsed.get("missing_ats_keywords") or [],
            "concrete_improvements": improvements,
            "suggested_rephrasings": rephrasings
        }

        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare analiză CV: {str(e)}", code=500)

@app.route("/interview-question", methods=["POST", "OPTIONS"], endpoint="interview_question_root")
@app.route("/api/interview-question", methods=["POST", "OPTIONS"], endpoint="interview_question_api")
def interview_question():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        user_answer = data.get("user_answer", "")
        role = data.get("role", "Software Developer")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        factuality_rules = enforce_factuality_and_language(target_lang)
        prompt = f"""
{factuality_rules}
Ești un recrutator pentru rolul: {role}. Răspuns candidat: "{user_answer}"
Returnează DOAR un obiect JSON valid:
{{
  "feedback": "Evaluare...",
  "score": 8,
  "next_question": "Următoarea întrebare..."
}}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res) or {}

        feedback = remove_consecutive_duplicates(parsed.get("feedback", ""))
        next_q = remove_consecutive_duplicates(parsed.get("next_question", ""))

        payload = {
            "feedback": feedback,
            "score": parsed.get("score", 7),
            "next_question": next_q,
            "question": next_q
        }

        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare interviu: {str(e)}", code=500)

@app.route("/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_root")
@app.route("/api/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_api")
def rephrase():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_text = data.get("text") or MEMORY.get("cv_text") or ""
        job_desc = data.get("job_description") or MEMORY.get("job_description") or ""
        target_lang = data.get("target_language") or data.get("language") or "ro"

        if not cv_text:
            return api_response(error="Textul CV-ului pentru reformulare lipsește.", code=400)

        factuality_rules = enforce_factuality_and_language(target_lang)

        if job_desc:
            prompt = f"""
{factuality_rules}
Ești un expert în scriere de CV-uri și optimizare ATS. 
Rescrie, structurează și refocalizează complet conținutul acestui CV bazându-te exclusiv pe faptele reale din CV și aliniindu-l cu Descrierea Jobului.

Returnează DOAR un obiect JSON valid cu structura:
{{
  "improved_text": "Textul complet rescris și optimizat al CV-ului..."
}}

CV ORIGINAL:
{cv_text}

DESCRIERE JOB:
{job_desc}
"""
        else:
            prompt = f"""
{factuality_rules}
Ești un expert în scriere de CV-uri. Îmbunătățește și reformulează acest CV pe baza exclusivă a datelor reale existente.

Returnează DOAR un obiect JSON valid cu structura:
{{
  "improved_text": "Textul optimizat..."
}}

CV ORIGINAL:
{cv_text}
"""

        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res)
        improved = parsed.get("improved_text") if parsed else raw_res
        improved = remove_consecutive_duplicates(improved)

        payload = {
            "improved_text": improved,
            "rephrased_text": improved,
            "text": improved
        }

        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare rephrase: {str(e)}", code=500)

@app.route("/export-docx", methods=["POST", "OPTIONS"], endpoint="export_docx_root")
@app.route("/api/export-docx", methods=["POST", "OPTIONS"], endpoint="export_docx_api")
def export_docx():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        text_content = data.get("text") or MEMORY.get("cv_text") or ""

        print(f"--- [DIAGNOZĂ EXPORT DOCX] --- Lungime text primit pentru generare: {len(text_content)} caractere", flush=True)

        if not text_content:
            print("❌ [DIAGNOZĂ EXPORT DOCX] Textul este gol sau lipsă!", flush=True)
            return api_response(error="Text lipsă pentru export.", code=400)

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

        print("✅ [DIAGNOZĂ EXPORT DOCX] Documentul DOCX a fost generat cu succes în memorie.", flush=True)

        return send_file(
            file_stream,
            as_attachment=True,
            download_name="CV_Optimizat.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        print(f"❌ EROARE CRITICĂ în export-docx: {type(e).__name__} - {str(e)}", flush=True)
        traceback.print_exc()
        return api_response(error=f"Eroare generare DOCX: {str(e)}", code=500)

@app.route("/get-session", methods=["GET", "OPTIONS"], endpoint="get_session_root")
@app.route("/api/get-session", methods=["GET", "OPTIONS"], endpoint="get_session_api")
def get_session():
    if request.method == "OPTIONS":
        return api_response(code=200)
    return api_response(payload={
        "has_cv": bool(MEMORY.get("cv_text")),
        "cv_length": len(MEMORY.get("cv_text", "")),
        "cv_text": MEMORY.get("cv_text", ""),
        "has_job": bool(MEMORY.get("job_description")),
        "job_description": MEMORY.get("job_description", "")
    })

@app.errorhandler(404)
def not_found(e):
    return api_response(error="Endpoint-ul căutat nu există pe server.", code=404)

@app.errorhandler(500)
def server_error(e):
    return api_response(error="Eroare internă pe server.", code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Serverul pornește pe portul {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
