import os
import re
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# Configurare aplicație Flask
app = Flask(__name__)
CORS(app)

# Memorie temporară în RAM pentru sesiune
MEMORY = {
    "cv_text": "",
    "job_description": "",
    "interview_history": []
}

# ==========================================
# 1. INIȚIALIZARE CLIENT AI (Groq & Gemini)
# ==========================================

# Groq Setup
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

# Gemini Setup
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


# ==========================================
# 2. FUNCȚII AUXILIARE & UTILS
# ==========================================

def clean_text(text: str) -> str:
    """Curăță textul extras de caractere inutile."""
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def safe_json(raw_text: str) -> dict:
    """Extrage și validează un obiect JSON dintr-un răspuns LLM."""
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
    """Format unitar pentru răspunsurile API."""
    if error:
        return jsonify({"status": "error", "message": error, "error": error}), code
    if isinstance(payload, dict):
        response_dict = {"status": "success", "data": payload}
        response_dict.update(payload)
        return jsonify(response_dict), code
    return jsonify({"status": "success", "data": payload if payload is not None else {}}), code

def gemini_text(prompt: str) -> str:
    """Invocă Groq cu fallback automat pe Gemini."""
    if USE_GROQ and groq_client:
        try:
            res = groq_client.with_options(max_retries=0).chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ești un asistent AI profesionist specializat în resurse umane, optimizare CV-uri și interviuri. "
                            "Răspunde STRICT în formatul solicitat (JSON valid când se cere JSON)."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096,
                timeout=12.0
            )
            if res and res.choices:
                return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Groq Rate Limit / Error ({type(e).__name__}). Fallback pe Gemini...", flush=True)

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"❌ Eroare Gemini: {type(e).__name__} - {str(e)}", flush=True)

    return ""


# ==========================================
# 3. TOATE RUTELE API (COMPLETE)
# ==========================================

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({
        "status": "online",
        "service": "vCoach AI API",
        "groq_active": USE_GROQ,
        "gemini_active": gemini_client is not None
    }), 200

@app.route("/ping", methods=["GET", "HEAD"])
def ping():
    return "OK", 200

# ------------------------------------------
# RUTA 1: UPLOAD CV & TEXT
# ------------------------------------------
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
                try:
                    import fitz  # PyMuPDF
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
        return api_response(payload={"message": "CV încărcat cu succes", "length": len(cleaned)})

    except Exception as e:
        return api_response(error=f"Eroare la procesare: {str(e)}", code=500)

# ------------------------------------------
# RUTA 2: ANALIZĂ CALITATE CV (CV Quality)
# ------------------------------------------
@app.route("/analyze-cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_root")
@app.route("/api/cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_api")
@cross_origin()
def analyze_cv_quality():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        target_lang = data.get("target_language") or data.get("language") or "ro"
        cv = clean_text(cv_raw)

        if not cv:
            return api_response(error="CV lipsă. Vă rugăm încărcați un CV.", code=400)

        MEMORY["cv_text"] = cv

        prompt = f"""
Ești un specialist ATS. Analizează acest CV.
Limba de răspuns: STRICTLY {target_lang}.

Returnează DOAR un obiect JSON valid:
{{
  "clarity_score": 8,
  "relevance_score": 7,
  "structure_score": 8,
  "ats_keywords": ["Cuvant1", "Cuvant2", "Cuvant3", "Cuvant4", "Cuvant5"],
  "concrete_improvements": ["Sfat 1", "Sfat 2"],
  "suggested_rephrasings": ["Rescriere 1"]
}}

CV:
{cv}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res)

        clarity = parsed.get("clarity_score") or parsed.get("clarity") or 7
        relevance = parsed.get("relevance_score") or parsed.get("relevance") or 7
        structure = parsed.get("structure_score") or parsed.get("structure") or 7

        keywords = parsed.get("ats_keywords") or parsed.get("keywords") or ["Experiență", "Abilități"]
        improvements = parsed.get("concrete_improvements") or parsed.get("improvements") or ["Adăugați detalii cuantificabile."]
        rephrasings = parsed.get("suggested_rephrasings") or parsed.get("rephrasings") or []

        payload = {
            "clarity_score": clarity,
            "clarity": clarity,
            "relevance_score": relevance,
            "relevance": relevance,
            "structure_score": structure,
            "structure": structure,
            "overall_assessment": "Analiza CV-ului a fost finalizată cu succes.",
            "ats_keywords": keywords,
            "keywords": keywords,
            "concrete_improvements": improvements,
            "improvements": improvements,
            "suggested_rephrasings": rephrasings,
            "rephrasings": rephrasings
        }

        return api_response(payload=payload)

    except Exception as e:
        return api_response(error=f"Eroare analiză CV: {str(e)}", code=500)

# ------------------------------------------
# RUTA 3: POTRIVIRE CV CU JOB (Match / Job Fit)
# ------------------------------------------
@app.route("/match-job", methods=["POST", "OPTIONS"], endpoint="match_job_root")
@app.route("/api/match-job", methods=["POST", "OPTIONS"], endpoint="match_job_api")
@cross_origin()
def match_job():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv = clean_text(data.get("cv_text") or MEMORY.get("cv_text") or "")
        job_desc = clean_text(data.get("job_description") or MEMORY.get("job_description") or "")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        if not cv or not job_desc:
            return api_response(error="Atât CV-ul cât și Descrierea Jobului sunt necesare.", code=400)

        MEMORY["job_description"] = job_desc

        prompt = f"""
Compară CV-ul cu Descrierea Iobului. Limba de răspuns: STRICTLY {target_lang}.
Returnează DOAR un obiect JSON valid:
{{
  "match_score": 75,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "recommendations": ["rec1", "rec2"]
}}

CV:
{cv}

JOB DESCRIPTION:
{job_desc}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res) or {
            "match_score": 65,
            "matching_skills": ["Experiență generală"],
            "missing_skills": ["Cerințe specifice"],
            "recommendations": ["Ajustați CV-ul cu cuvintele cheie din anunț."]
        }

        payload = {
            "match_score": parsed.get("match_score", 65),
            "score": parsed.get("match_score", 65),
            "matching_skills": parsed.get("matching_skills", []),
            "missing_skills": parsed.get("missing_skills", []),
            "recommendations": parsed.get("recommendations", [])
        }

        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare match-job: {str(e)}", code=500)

# ------------------------------------------
# RUTA 4: SIMULARE INTERVIU (Interview Simulation)
# ------------------------------------------
@app.route("/interview-question", methods=["POST", "OPTIONS"], endpoint="interview_question_root")
@app.route("/api/interview-question", methods=["POST", "OPTIONS"], endpoint="interview_question_api")
@cross_origin()
def interview_question():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        user_answer = data.get("user_answer", "")
        role = data.get("role", "Software Developer")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        prompt = f"""
Ești un recrutator pentru rolul: {role}.
Limba de răspuns: STRICTLY {target_lang}.
Răspunsul anterior al candidatului: "{user_answer}"

Returnează DOAR un obiect JSON valid:
{{
  "feedback": "Scurtă evaluare a răspunsului.",
  "score": 8,
  "next_question": "Următoarea întrebare de interviu."
}}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res) or {
            "feedback": "Răspuns bun. Încercați să oferiți un exemplu mai concret.",
            "score": 7,
            "next_question": "Ne puteți oferi un exemplu de proiect dificil pe care l-ați gestionat?"
        }

        payload = {
            "feedback": parsed.get("feedback", ""),
            "score": parsed.get("score", 7),
            "next_question": parsed.get("next_question", ""),
            "question": parsed.get("next_question", "")
        }

        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare interviu: {str(e)}", code=500)

# ------------------------------------------
# RUTA 5: TRADUCERE CV / TEXT (Translate)
# ------------------------------------------
@app.route("/translate", methods=["POST", "OPTIONS"], endpoint="translate_root")
@app.route("/api/translate", methods=["POST", "OPTIONS"], endpoint="translate_api")
@cross_origin()
def translate():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text") or MEMORY.get("cv_text") or ""
        target_lang = data.get("target_language") or data.get("language") or "en"

        if not text:
            return api_response(error="Textul de tradus lipsește.", code=400)

        prompt = f"Translate the following professional resume text into {target_lang}. Maintain formal tone:\n\n{text}"
        translated_text = gemini_text(prompt)

        payload = {
            "translated_text": translated_text,
            "translation": translated_text,
            "text": translated_text
        }

        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare traducere: {str(e)}", code=500)

# ------------------------------------------
# RUTA 6: RESCRIERE / REPHRASE BULLET POINT
# ------------------------------------------
@app.route("/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_root")
@app.route("/api/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_api")
@cross_origin()
def rephrase():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        bullet_point = data.get("text", "")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        if not bullet_point:
            return api_response(error="Textul pentru rescriere lipsește.", code=400)

        prompt = f"""
Îmbunătățește acest punct din CV folosind verbe de acțiune și un limbaj profesional.
Limba de răspuns: STRICTLY {target_lang}.
Returnează DOAR un obiect JSON valid:
{{
  "improved_text": "Rescrierea profesională..."
}}

Original: {bullet_point}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res)
        improved = parsed.get("improved_text") if parsed else raw_res

        payload = {
            "improved_text": improved,
            "rephrased_text": improved,
            "text": improved
        }

        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare rephrase: {str(e)}", code=500)


# ==========================================
# 4. TRATARE ERORI GLOBALE
# ==========================================

@app.errorhandler(404)
def not_found(e):
    return api_response(error="Endpoint-ul căutat nu există pe server.", code=404)

@app.errorhandler(500)
def server_error(e):
    return api_response(error="A apărut o eroare internă pe server.", code=500)


# ==========================================
# 5. PORNIRE APLICAȚIE (Render Compliant)
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Serverul pornește pe portul {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
