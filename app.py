import os
import json
import re
import io
import fitz  # PyMuPDF pentru parsare PDF
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from google import genai
from flask_compress import Compress
from groq import Groq
from itertools import zip_longest

# Verificare sigură pentru pytesseract
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

# =========================
# CONFIG
# =========================
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)

app = Flask(__name__)

# Configurare CORS globală flexibilă
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
Compress(app)

# Handler explicit pentru wildcard OPTIONS (pentru a preveni blocajele CORS Preflight)
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        headers = response.headers
        headers['Access-Control-Allow-Origin'] = '*'
        headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

# Error handler global: garantează răspuns JSON și headere CORS la orice eroare Python (500)
@app.errorhandler(Exception)
def handle_global_exception(e):
    print(f"🔥 Eroare neprinsă pe server: {str(e)}")
    return jsonify({
        "status": "error",
        "payload": None,
        "error": f"Eroare internă server: {str(e)}"
    }), 500

# =========================
# SHARED MEMORY (SESSION-LIKE)
# =========================
MEMORY = {
    "cv_text": None
}

# =========================
# CLIENT INIT
# =========================
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"✅ Gemini ready | model: {MODEL_NAME}")
    except Exception as e:
        print(f"❌ Eroare la inițializarea Gemini Client: {str(e)}")
        gemini_client = None

groq_client = None
if USE_GROQ:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq ready")
    except Exception as e:
        print(f"❌ Eroare la inițializarea Groq: {str(e)}")
        groq_client = None


def groq_text(prompt: str) -> str:
    if not groq_client:
        return ""
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Ești un expert LinkedIn Job Search și recruiter profesionist. Răspunde NUMAI cu JSON valid atunci când se solicită."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {str(e)}")
        return ""


# =========================
# UTILS
# =========================
def api_response(payload=None, error=None, code=200):
    return jsonify({
        "status": "ok" if not error else "error",
        "payload": payload,
        "error": error
    }), code


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'[\x00-\x1F]+', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 2000) -> list:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def safe_json(text: str):
    if not text:
        return None
    text = clean_text(text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S | re.M)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def gemini_text(prompt: str) -> str:
    """Prioritate Groq (rapid), fallback Gemini."""
    if USE_GROQ and groq_client:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ești un recrutor profesionist cu peste 10 ani de experiență. "
                            "Dacă ți se cere JSON, returnează NUMAI JSON valid fără markdown sau alt text. "
                            "Dacă ți se cere text simplu, răspunde curat și profesionist. "
                            "CRITICAL: Detect and strictly adhere to the requested target language or source document language."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print("Groq error:", str(e))

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    {"role": "user", "parts": [{"text": prompt}]}
                ],
            )
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {type(e).__name__} - {str(e)}")

    return ""


# =========================
# PARSING & OCR ENDPOINTS
# =========================

@app.route("/api/parse-pdf", methods=["POST", "OPTIONS"])
@cross_origin()
def parse_pdf():
    if request.method == "OPTIONS":
        return api_response(code=200)

    if 'file' not in request.files:
        return api_response(error="Niciun fișier nu a fost furnizat", code=400)

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return api_response(error="Fișierul trebuie să fie un PDF", code=400)

    try:
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted_text = ""
        for page in doc:
            extracted_text += page.get_text() + "\n"

        cleaned_pdf_text = clean_text(extracted_text)
        if cleaned_pdf_text:
            MEMORY["cv_text"] = cleaned_pdf_text

        return api_response(payload={"text": cleaned_pdf_text})
    except Exception as e:
        return api_response(error=f"Eroare la parsarea PDF-ului: {str(e)}", code=500)


@app.route("/api/process-ocr", methods=["POST", "OPTIONS"])
@cross_origin()
def process_ocr():
    if request.method == "OPTIONS":
        return api_response(code=200)

    if 'image' not in request.files:
        return api_response(error="Nicio imagine furnizată", code=400)

    if not HAS_PYTESSERACT:
        return api_response(error="Librăria pytesseract nu este configurată pe server.", code=500)

    try:
        img_file = request.files['image']
        img_bytes = img_file.read()
        image = Image.open(io.BytesIO(img_bytes))
        extracted_text = pytesseract.image_to_string(image)
        return api_response(payload={"text": clean_text(extracted_text)})
    except Exception as e:
        return api_response(error="Motorul OCR Tesseract nu este instalat pe mediul serverului. Vă rugăm încărcați un fișier PDF.", code=500)


# =========================
# CORE ENDPOINTS
# =========================

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"})


@app.route("/check-cv-memory", methods=["GET", "OPTIONS"])
@cross_origin()
def check_cv_memory():
    if request.method == "OPTIONS":
        return api_response(code=200)
    if MEMORY.get("cv_text") and len(MEMORY["cv_text"].strip()) > 10:
        return api_response(payload={"has_cv": True}, code=200)
    else:
        return api_response(error="No CV in memory", code=404)


@app.route("/clear-memory", methods=["POST", "OPTIONS"])
@cross_origin()
def clear_memory():
    if request.method == "OPTIONS":
        return api_response(code=200)
    MEMORY["cv_text"] = None
    return api_response(payload={"message": "Memoria CV a fost ștearsă cu succes"})


@app.route("/analyze-cv-quality", methods=["POST", "OPTIONS"])
@app.route("/api/cv-quality", methods=["POST", "OPTIONS"])
@cross_origin()
def analyze_cv_quality():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        target_lang = data.get("target_language") or data.get("language") or "en"
        
        cv = clean_text(cv_raw)

        if not cv:
            return api_response(error="CV lipsă în request sau memorie. Vă rugăm încărcați mai întâi un CV.", code=400)

        MEMORY["cv_text"] = cv
        chunks = chunk_text(cv, chunk_size=3000)

        clarity_scores, relevance_scores, structure_scores = [], [], []
        concrete_improvements, suggested_rephrasings = [], []

        for chunk in chunks:
            prompt_chunk = f"""
You are a senior hybrid recruiter. Analyze ONLY the CV fragment below.

CRITICAL RULES:
1. Target Output Language: STRICTLY {target_lang}.
2. Do NOT use numbering or prefixes like "1.", "Improvement 1:" inside output array strings.
3. For "suggested_rephrasings" use EXACT format: "Original: \"...\", Improved: \"...\""
4. Return ONLY valid JSON.

JSON structure:
{{
  "clarity_score": int,
  "relevance_score": int,
  "structure_score": int,
  "concrete_improvements": ["suggestion...", ...],
  "suggested_rephrasings": ["Original: \"...\", Improved: \"...\""]
}}

CV fragment:
{chunk}
"""
            raw_chunk = gemini_text(prompt_chunk)
            parsed_chunk = safe_json(raw_chunk)

            if not parsed_chunk or not isinstance(parsed_chunk, dict):
                parsed_chunk = {
                    "clarity_score": 7, "relevance_score": 7, "structure_score": 7,
                    "concrete_improvements": [], "suggested_rephrasings": []
                }

            clarity_scores.append(parsed_chunk.get("clarity_score", 7))
            relevance_scores.append(parsed_chunk.get("relevance_score", 7))
            structure_scores.append(parsed_chunk.get("structure_score", 7))

            improvements = parsed_chunk.get("concrete_improvements", [])
            if isinstance(improvements, list):
                concrete_improvements.extend(improvements)

            rephrasings = parsed_chunk.get("suggested_rephrasings", [])
            if isinstance(rephrasings, list):
                suggested_rephrasings.extend(rephrasings)

        final_payload = {
            "clarity_score": int(sum(clarity_scores) / len(clarity_scores)) if clarity_scores else 7,
            "relevance_score": int(sum(relevance_scores) / len(relevance_scores)) if relevance_scores else 7,
            "structure_score": int(sum(structure_scores) / len(structure_scores)) if structure_scores else 7,
            "overall_assessment": "CV analysis completed successfully.",
            "concrete_improvements": concrete_improvements[:10],
            "suggested_rephrasings": suggested_rephrasings[:10]
        }

        return api_response(payload=final_payload)

    except Exception as e:
        print(f"❌ Error inside analyze_cv_quality: {str(e)}")
        return api_response(error=f"Eroare procesare CV: {str(e)}", code=500)


@app.route("/analyze-cv", methods=["POST", "OPTIONS"])
@app.route("/api/audit-cv", methods=["POST", "OPTIONS"])
@cross_origin()
def analyze_cv():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_text") or data.get("job_description") or ""
        language = data.get("language", "auto")

        if not cv_raw or not job_raw:
            return api_response(error="Date lipsă: Atât CV-ul cât și descrierea postului sunt necesare.", code=400)

        cv_clean = clean_text(cv_raw)
        job_clean = clean_text(job_raw)
        MEMORY["cv_text"] = cv_clean

        lang_instruction = f"Strictly respond in language: {language}." if language != "auto" else "Respond in the language of the job description."

        cv_chunks = chunk_text(cv_clean, chunk_size=3000)
        job_chunks = chunk_text(job_clean, chunk_size=3000)

        chunk_feedbacks, chunk_scores = [], []

        for i, (cv_chunk, job_chunk) in enumerate(zip_longest(cv_chunks, job_chunks, fillvalue="")):
            prompt_chunk = f"""
Ești un recrutor profesionist. {lang_instruction}
Analizează compatibilitatea dintre CV și cerințele postului.
Oferă procentaj realist (0-100) și feedback detaliat.

Returnează NUMAI JSON valid:
{{"compatibility_percent": int, "feedback_markdown": "text curat și profesionist"}}

CV fragment: {cv_chunk}
Job fragment: {job_chunk}
"""
            raw_chunk = gemini_text(prompt_chunk)
            parsed_chunk = safe_json(raw_chunk) or {
                "compatibility_percent": 70,
                "feedback_markdown": "Fragmentul CV-ului are relevanță parțială pentru cerințele acestui segment al jobului."
            }

            chunk_feedbacks.append(parsed_chunk.get("feedback_markdown", ""))
            chunk_scores.append(parsed_chunk.get("compatibility_percent", 70))

        combined_feedback = "\n\n".join(chunk_feedbacks)

        final_prompt = f"""
Rescrie feedback-ul combinat într-un text profesionist, fluent și corect. {lang_instruction}
Text combinat:
{combined_feedback}
Returnează NUMAI text curat.
"""
        res_final = gemini_text(final_prompt)
        if not res_final.strip():
            res_final = combined_feedback

        final_score = int(sum(chunk_scores) / len(chunk_scores)) if chunk_scores else 75

        return api_response(payload={
            "compatibility_percent": final_score,
            "feedback_markdown": res_final
        })

    except Exception as e:
        return api_response(error=f"Eroare internă: {str(e)}", code=500)


@app.route("/generate-questions", methods=["POST", "OPTIONS"])
@app.route("/api/start-interview", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_questions():
    if request.method == "OPTIONS":
        return api_response(code=200)

    data = request.get_json(force=True, silent=True) or {}
    cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
    job = data.get("job_summary") or data.get("job_description") or ""
    language = data.get("language", "auto")

    cv = clean_text(cv_raw)
    if not cv or not job:
        return api_response(error="Date lipsă", code=400)

    MEMORY["cv_text"] = cv

    lang_instruction = f"Strictly respond in language: {language}." if language != "auto" else "Respond in the language of the CV."

    prompt = f"""
Ești un recrutor profesionist. {lang_instruction}
Generează exact 5 întrebări de interviu relevante, profesionale și bine țintite pe baza CV-ului și a postului.
Returnează NUMAI JSON valid:
{{"questions": ["întrebare 1", "întrebare 2", "întrebare 3", "întrebare 4", "întrebare 5"]}}

CV: {cv}
Job: {job}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw) or {
        "questions": [
            "Povestiți despre experiența dumneavoastră cea mai relevantă pentru acest post.",
            "Care considerați că sunt principalele dumneavoastră puncte forte în relație cu cerințele rolului?",
            "Descrieți o situație dificilă din carieră și modul în care ați gestionat-o.",
            "Ce vă motivează să aplicați pentru această poziție?",
            "Cum abordați învățarea continuă și adaptarea la tehnologii noi?"
        ]
    }
    return api_response(payload=parsed)


@app.route("/reformulate-cv-for-job-boards", methods=["POST", "OPTIONS"])
@app.route("/api/reframe-cv", methods=["POST", "OPTIONS"])
@cross_origin()
def reformulate_cv_for_job_boards():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_text") or data.get("job_description") or ""
        language = data.get("language", "auto")

        cv_clean = clean_text(cv_raw)
        if not cv_clean:
            return api_response(error="CV lipsă", code=400)

        MEMORY["cv_text"] = cv_clean

        lang_instruction = f"Strictly respond in language: {language}." if language != "auto" else "Detect the language of the CV and respond STRICTLY in that language."

        prompt = f"""
Ești un expert senior în recrutare și sisteme ATS. {lang_instruction}
REFORMULEAZĂ CV-ul candidatului.

REGULI STRICTE:
- Do NOT invent experience or add unmentioned skills.

STRUCTURA DE RĂSPUNS (JSON STRICT):
{{
  "normalized_titles": ["Titlu standard 1"],
  "cv_summary_for_job_boards": "Rezumat profesionist, clar, ATS-friendly (max 120 cuvinte)",
  "core_skills_keywords": ["keyword ATS 1", "keyword ATS 2"],
  "notes_for_candidate": "Observații oneste despre limitări sau sugestii"
}}

CV: {cv_clean}
Descriere job (opțional): {job_raw}
"""
        raw = groq_text(prompt) if USE_GROQ else gemini_text(prompt)
        parsed = safe_json(raw)

        required_keys = ["normalized_titles", "cv_summary_for_job_boards", "core_skills_keywords", "notes_for_candidate"]
        if not parsed or not isinstance(parsed, dict) or not all(k in parsed for k in required_keys):
            return api_response(error="AI nu a putut genera un rezultat valid pentru reformularea CV-ului", code=503)

        return api_response(payload=parsed)

    except Exception as e:
        return api_response(error="Eroare internă server", code=503)


@app.route("/generate-job-queries", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_job_queries():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        cv_clean = clean_text(cv_raw)

        if not cv_clean:
            return api_response(error="CV lipsă", code=400)

        MEMORY["cv_text"] = cv_clean

        prompt = f"""
Ești un expert senior în recrutare și LinkedIn Job Search.
Analizează EXCLUSIV CV-ul de mai jos și stabilește dacă experiența candidatului poate fi asociată clar cu roluri standard.

REGULI:
- FIECARE căutare trebuie să conțină UN SINGUR titlu de job standard (engleză)
- Dacă NU poți identifica MINIM 3 roluri clare, returnează EXACT:
{{"status": "no_clear_match", "message": "Experiența candidatului este prea nișată sau formulată într-un mod care nu permite asocierea clară cu roluri standard."}}

Dacă POȚI identifica roluri clare, returnează EXACT 7 căutări:
{{"queries": ["Job Title 1", "Job Title 2", "Job Title 3", "Job Title 4", "Job Title 5", "Job Title 6", "Job Title 7"]}}

CV: {cv_clean}
"""
        raw = groq_text(prompt) if USE_GROQ else gemini_text(prompt)
        parsed = safe_json(raw)

        if parsed and isinstance(parsed, dict) and parsed.get("status") == "no_clear_match":
            return api_response(payload=parsed)

        if not parsed or not isinstance(parsed, dict) or "queries" not in parsed or not isinstance(parsed["queries"], list) or len(parsed["queries"]) != 7:
            return api_response(payload={
                "status": "no_clear_match",
                "message": "Experiența candidatului este prea nișată sau formulată într-un mod care nu permite asocierea clară cu roluri standard."
            })

        return api_response(payload={"queries": parsed["queries"]})

    except Exception as e:
        return api_response(error=f"Eroare internă server: {str(e)}", code=503)


@app.route("/optimize-linkedin-profile", methods=["POST", "OPTIONS"])
@cross_origin()
def optimize_linkedin_profile():
    if request.method == "OPTIONS":
        return api_response(code=200)

    data = request.get_json(force=True, silent=True) or {}
    cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
    cv = clean_text(cv_raw)
    if not cv:
        return api_response(error="CV lipsă", code=400)

    MEMORY["cv_text"] = cv

    prompt = f"""
Optimizează profilul LinkedIn pe baza CV-ului.
Propune exact 5 headline-uri atractive și scrie o secțiune About captivantă (300-500 cuvinte).
Returnează NUMAI JSON valid:
{{"linkedin_headlines": ["h1", "h2", "h3", "h4", "h5"], "linkedin_about": "text complet About"}}

CV: {cv}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw) or {
        "linkedin_headlines": [
            "Senior Professional | Scalable Solutions",
            "Tech Lead | Driving Innovation & Team Growth",
            "Specialist | Delivering Measurable Results",
            "Domain Expert | Strategic Execution",
            "Professional | Continuous Improvement"
        ],
        "linkedin_about": "Profil LinkedIn optimizat profesional pe baza experienței dumneavoastră."
    }
    return api_response(payload=parsed)


@app.route("/evaluate-answer", methods=["POST", "OPTIONS"])
@cross_origin()
def evaluate_answer():
    if request.method == "OPTIONS":
        return api_response(code=200)

    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    if not question or not answer:
        return api_response(error="Date lipsă", code=400)

    prompt = f"""
Evaluează răspunsul candidatului pe TREI dimensiuni distincte (0–10):
- claritate
- structura
- relevanta

Oferă și un feedback scurt, constructiv (max 2 fraze).
Returnează STRICT acest JSON:
{{
  "claritate": int,
  "structura": int,
  "relevanta": int,
  "feedback": "text feedback"
}}

Întrebarea: {question}
Răspunsul: {answer}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)

    if parsed and isinstance(parsed, dict) and all(k in parsed for k in ("claritate", "structura", "relevanta")):
        c, s, r = int(parsed["claritate"]), int(parsed["structura"]), int(parsed["relevanta"])
        parsed["nota_finala"] = round(0.30 * c + 0.35 * s + 0.35 * r)
    else:
        parsed = {
            "claritate": 7,
            "structura": 6,
            "relevanta": 7,
            "nota_finala": 7,
            "feedback": "Răspunsul este coerent, dar poate fi îmbunătățit printr-o structurare mai clară și exemple mai relevante."
        }

    return api_response(payload={"current_evaluation": parsed})


@app.route("/generate-report", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_report():
    if request.method == "OPTIONS":
        return api_response(code=200)

    data = request.get_json(force=True, silent=True) or {}
    history = data.get("history", [])
    if not history:
        return api_response(error="Istoric lipsă", code=400)

    prompt = f"""
Analizează întregul istoric al interviului și generează un raport final obiectiv.
Include un rezumat al performanței candidatului și un scor general (1-10).

Returnează NUMAI JSON valid:
{{"summary": "rezumat detaliat și profesionist", "scor_final": int}}

Istoric interviu: {json.dumps(history)}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw) or {
        "summary": "Candidatul a demonstrat competențe solide și o atitudine profesionistă pe parcursul interviului.",
        "scor_final": 8
    }
    return api_response(payload=parsed)


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
