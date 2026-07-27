import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from google import genai
from flask_compress import Compress
from groq import Groq
from itertools import zip_longest

# =========================
# CONFIG & INITIALIZATION
# =========================
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)

app = Flask(__name__)

# Configurare CORS Globală Permisivă
CORS(app, resources={r"/*": {"origins": "*"}})
Compress(app)

# =========================
# SHARED IN-MEMORY STORAGE
# =========================
MEMORY = {
    "cv_text": None
}

# =========================
# LLM CLIENTS CONFIG
# =========================
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Gemini API configurat cu succes (gemini-1.5-flash).")
    except Exception as e:
        print(f"❌ Eroare la configurarea Gemini API: {str(e)}")

groq_client = None
if USE_GROQ:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq Client pregătit.")
    except Exception as e:
        print(f"❌ Eroare la inițializarea Groq: {str(e)}")


# =========================
# UTILITY FUNCTIONS
# =========================
def api_response(payload=None, error=None, code=200):
    """
    Răspuns JSON garantat cu antete CORS, chiar și în caz de erori 500 sau 400.
    """
    response_data = {
        "status": "ok" if not error else "error",
        "payload": payload,
        "error": error
    }
    res = jsonify(response_data)
    res.status_code = code
    res.headers["Access-Control-Allow-Origin"] = "*"
    res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return res


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'[\x00-\x1F]+', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 3000) -> list:
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


def ask_ai(prompt: str) -> str:
    """ Funcție unificată de interogare LLM cu fallback (Groq -> Gemini) """
    # 1. Încercare cu Groq (dacă există API Key)
    if USE_GROQ and groq_client:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "Ești un recruiter profesionist și Career Coach. Dacă ți se cere JSON, răspunde STRICT în format JSON valid fără alt text în afara structurii JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Groq error: {str(e)}")

    # 2. Încercare cu Gemini
    if gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"❌ Gemini error: {type(e).__name__} - {str(e)}")

    return ""


# =========================
# MIDDLEWARE & HANDLERS
# =========================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.errorhandler(Exception)
def handle_global_exception(e):
    print(f"🔥 Unhandled Server Exception: {str(e)}")
    return api_response(error=f"A apărut o eroare pe server: {str(e)}", code=500)


# =========================
# ALL ENDPOINTS
# =========================

@app.route("/", methods=["GET"])
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake", "service": "AI Career Suite Backend"})


@app.route("/check-cv-memory", methods=["GET", "OPTIONS"])
@cross_origin()
def check_cv_memory():
    if request.method == "OPTIONS":
        return api_response(code=200)
    if MEMORY.get("cv_text") and len(MEMORY["cv_text"].strip()) > 10:
        return api_response(payload={"has_cv": True}, code=200)
    else:
        return api_response(error="Nu există CV salvat în memorie.", code=404)


@app.route("/clear-memory", methods=["POST", "OPTIONS"])
@cross_origin()
def clear_memory():
    if request.method == "OPTIONS":
        return api_response(code=200)
    MEMORY["cv_text"] = None
    return api_response(payload={"message": "Memoria temporară a fost curățată."})


@app.route("/analyze-cv-quality", methods=["POST", "OPTIONS"])
@app.route("/api/cv-quality", methods=["POST", "OPTIONS"])
@cross_origin()
def analyze_cv_quality():
    """ Auditează CV-ul și oferă scoruri + sugestii de îmbunătățire unice """
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_description") or data.get("job_text") or ""

        cv = clean_text(cv_raw)

        if not cv:
            return api_response(error="CV-ul lipsește. Încarcă mai întâi fișierul PDF.", code=400)

        MEMORY["cv_text"] = cv

        context_extra = ""
        if job_raw:
            context_extra = f"\n\nJOB DESCRIPTION / CERINȚE POST:\n{clean_text(job_raw)}"

        chunks = chunk_text(cv, chunk_size=3000)

        clarity_scores, relevance_scores, structure_scores = [], [], []
        raw_improvements, raw_rephrasings = [], []

        for chunk in chunks:
            prompt_chunk = f"""
You are a senior recruiter. Analyze ONLY the CV fragment below.

CRITICAL RULES:
1. Output MUST be written STRICTLY IN ROMANIAN.
2. Provide specific, actionable, distinct recommendations. Do NOT repeat yourself.
3. For "suggested_rephrasings" use EXACT format: "Original: \"...\", Improved: \"...\""
4. Return ONLY valid JSON.

JSON structure:
{{
  "clarity_score": 8,
  "relevance_score": 7,
  "structure_score": 8,
  "concrete_improvements": [
    "Evidențiază rezultatele obținute folosind metrici și procente.",
    "Formatează secțiunea de abilități tehnice sub formă de listă structurată.",
    "Elimină frazele generice și detaliază rolul tău exact în proiecte."
  ],
  "suggested_rephrasings": [
    "Original: \"Am lucrat la proiect\", Improved: \"Am coordonat dezvoltarea modulului X creșterea eficienței cu 15%\""
  ]
}}

CV fragment:
{chunk}
{context_extra}
"""
            raw_chunk = ask_ai(prompt_chunk)
            parsed_chunk = safe_json(raw_chunk)

            if not parsed_chunk or not isinstance(parsed_chunk, dict):
                parsed_chunk = {
                    "clarity_score": 8, "relevance_score": 7, "structure_score": 8,
                    "concrete_improvements": [
                        "Adaugă realizări cuantificabile și metrici în experiența profesională.",
                        "Evidențiază mai bine tehnologiile și uneltele folosite la fiecare job.",
                        "Optimizează structura secțiunii de profil pentru a fi citită mai ușor."
                    ],
                    "suggested_rephrasings": []
                }

            clarity_scores.append(parsed_chunk.get("clarity_score", 7))
            relevance_scores.append(parsed_chunk.get("relevance_score", 7))
            structure_scores.append(parsed_chunk.get("structure_score", 7))

            imp = parsed_chunk.get("concrete_improvements", [])
            if isinstance(imp, list):
                raw_improvements.extend(imp)

            reph = parsed_chunk.get("suggested_rephrasings", [])
            if isinstance(reph, list):
                raw_rephrasings.extend(reph)

        # Eliminăm duplicatele păstrând ordinea (Deduplication)
        unique_improvements = []
        for item in raw_improvements:
            clean_item = clean_text(str(item))
            if clean_item and clean_item not in unique_improvements:
                unique_improvements.append(clean_item)

        unique_rephrasings = []
        for item in raw_rephrasings:
            clean_item = clean_text(str(item))
            if clean_item and clean_item not in unique_rephrasings:
                unique_rephrasings.append(clean_item)

        final_payload = {
            "clarity_score": int(sum(clarity_scores) / len(clarity_scores)) if clarity_scores else 7,
            "relevance_score": int(sum(relevance_scores) / len(relevance_scores)) if relevance_scores else 7,
            "structure_score": int(sum(structure_scores) / len(structure_scores)) if structure_scores else 7,
            "overall_assessment": "Analiza CV-ului a fost finalizată cu succes.",
            "concrete_improvements": unique_improvements[:6],
            "suggested_rephrasings": unique_rephrasings[:6]
        }

        return api_response(payload=final_payload)

    except Exception as e:
        print(f"❌ Error inside analyze_cv_quality: {str(e)}")
        return api_response(error=f"Eroare internă la analiza CV: {str(e)}", code=500)


@app.route("/analyze-cv", methods=["POST", "OPTIONS"])
@cross_origin()
def analyze_cv():
    """ Calculează procentul de potrivire (Matching score) dintre CV și Job """
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_description") or data.get("job_text") or ""

        if not cv_raw or not job_raw:
            return api_response(error="Date incomplete: Atât CV-ul cât și descrierea jobului sunt necesare.", code=400)

        cv_clean = clean_text(cv_raw)
        job_clean = clean_text(job_raw)
        MEMORY["cv_text"] = cv_clean

        cv_chunks = chunk_text(cv_clean, chunk_size=3000)
        job_chunks = chunk_text(job_clean, chunk_size=3000)

        chunk_feedbacks, chunk_scores = [], []

        for cv_chunk, job_chunk in zip_longest(cv_chunks, job_chunks, fillvalue=""):
            prompt_chunk = f"""
Ești un recruiter profesionist. Analizează potrivirea dintre CV și cerințele postului.
Returnează un procent de compatibilitate realist (0-100) și un feedback în format JSON:

{{"compatibility_percent": int, "feedback_markdown": "text feedback"}}

CV: {cv_chunk}
Job: {job_chunk}
"""
            raw_chunk = ask_ai(prompt_chunk)
            parsed_chunk = safe_json(raw_chunk) or {
                "compatibility_percent": 75,
                "feedback_markdown": "CV-ul conține experiențe relevante pentru cerințele menționate."
            }

            chunk_feedbacks.append(parsed_chunk.get("feedback_markdown", ""))
            chunk_scores.append(parsed_chunk.get("compatibility_percent", 75))

        combined_feedback = "\n\n".join(chunk_feedbacks)

        final_prompt = f"""
Sintetizează feedback-ul următor într-un raport clar și bine structurat în limba dominantă a textului:
{combined_feedback}
"""
        res_final = ask_ai(final_prompt)
        if not res_final.strip():
            res_final = combined_feedback

        final_score = int(sum(chunk_scores) / len(chunk_scores)) if chunk_scores else 75

        return api_response(payload={
            "compatibility_percent": final_score,
            "feedback_markdown": res_final
        })

    except Exception as e:
        return api_response(error=f"Eroare internă la potrivirea CV-ului: {str(e)}", code=500)


@app.route("/generate-cover-letter", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_cover_letter():
    """ Generează o scrisoare de intenție personalizată """
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_description") or data.get("job_text") or ""

        if not cv_raw or not job_raw:
            return api_response(error="CV-ul și descrierea jobului sunt obligatorii.", code=400)

        prompt = f"""
Ești un expert în scrierea scrisorilor de intenție (Cover Letters).
Creează o scrisoare de intenție adaptată perfect pentru jobul descris, bazată pe experiența din CV.

Limbă: Folosește limba în care este redactat anunțul de job.
Formatați rezultatul în text curat.

CV:
{clean_text(cv_raw)[:3000]}

JOB:
{clean_text(job_raw)[:3000]}
"""
        cover_letter = ask_ai(prompt)
        return api_response(payload={"cover_letter": cover_letter})

    except Exception as e:
        return api_response(error=f"Eroare la generarea scrisorii de intenție: {str(e)}", code=500)


@app.route("/interview", methods=["POST", "OPTIONS"])
@app.route("/api/interview", methods=["POST", "OPTIONS"])
@cross_origin()
def interview_simulation():
    """ Generează întrebări de interviu și simulator interactiv """
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_description") or data.get("job_text") or ""

        if not cv_raw:
            return api_response(error="CV-ul este necesar pentru generarea întrebărilor de interviu.", code=400)

        prompt = f"""
Ești un Hiring Manager riguros. Generează un set de 5 întrebări tehnice și comportamentale de interviu bazate pe CV și job.

Returnează STRICT un JSON valid cu următoarea structură:
{{
  "questions": [
    {{"id": 1, "question": "întrebare...", "category": "Tehnic / Comportamental"}},
    ...
  ]
}}

CV:
{clean_text(cv_raw)[:3000]}

JOB:
{clean_text(job_raw)[:3000]}
"""
        raw_res = ask_ai(prompt)
        parsed = safe_json(raw_res) or {
            "questions": [
                {"id": 1, "question": "Descrie cel mai complex proiect din experiența ta recentă.", "category": "Experiență"},
                {"id": 2, "question": "Cum gestionezi o situație cu deadline-uri strânse?", "category": "Comportamental"}
            ]
        }

        return api_response(payload=parsed)

    except Exception as e:
        return api_response(error=f"Eroare la generarea interviului: {str(e)}", code=500)


# =========================
# SERVER LAUNCH
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
