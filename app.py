import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from google import genai
from google.genai import types
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
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configurat cu succes.")
    except Exception as e:
        print(f"❌ Eroare la configurarea Gemini API: {str(e)}")

groq_client = None
if USE_GROQ:
    try:
        # Prevenim erori de proxy în SDK-ul Groq
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)

        # Setăm timeout strict (10s) și 1 retry max pentru a împiedica blocarea worker-ului Gunicorn
        groq_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=10.0,
            max_retries=1
        )
        print("✅ Groq Client pregătit cu timeout de 10s și max 1 retry.")
    except Exception as e:
        print(f"❌ Eroare la inițializarea Groq: {str(e)}")


# =========================
# UTILITY & FILTERING FUNCTIONS
# =========================
def api_response(payload=None, error=None, code=200):
    """ Răspuns JSON garantat cu antete CORS """
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


def optimize_cv_text(text: str) -> str:
    """ Curățare backend pentru textul din CV """
    if not text:
        return ""
    # Eliminăm spațiile/tab-urile multiple
    text = re.sub(r'[ \t]+', ' ', str(text))
    # Reducem liniile goale consecutive la maxim una
    text = re.sub(r'\n\s*\n+', '\n', text)
    # Eliminăm caractere de control invizibile
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    # Înlocuim URL-uri extrem de lungi cu un placeholder scurt
    text = re.sub(r'https?://\S{40,}', '[link_scurtat]', text)
    return text.strip()


def truncate_smart(text: str, max_chars: int = 5000) -> str:
    """ Trunchiază textul la ultimul cuvânt complet fără a depăși limita maximă """
    text = optimize_cv_text(text)
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space != -1:
        truncated = truncated[:last_space]
    return truncated + "\n...[Text prescurtat pentru optimizare viteză]"


def chunk_text(text: str, chunk_size: int = 2500) -> list:
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
    text_clean = re.sub(r'```(?:json)?\n?', '', text)
    text_clean = text_clean.replace('```', '').strip()
    
    try:
        return json.loads(text_clean)
    except Exception:
        match = re.search(r"\{.*\}", text_clean, re.S | re.M)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def ask_ai(prompt: str, force_json: bool = False) -> str:
    """ Interogare LLM cu fallback rapid (Groq -> Gemini) """
    if USE_GROQ and groq_client:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "Ești un recruiter profesionist și Career Coach. Dacă ți se cere JSON, răspunde STRICT în format JSON valid."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2048
            )
            val = res.choices[0].message.content.strip()
            if val:
                return val
        except Exception as e:
            print(f"⚠️ Groq timeout/error, fallback la Gemini: {str(e)}")

    if gemini_client:
        try:
            config = None
            if force_json:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=config
            )
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"❌ Gemini error: {str(e)}")

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
    print(f"🔥 Unhandled Exception: {str(e)}")
    return api_response(error=f"A apărut o eroare pe server: {str(e)}", code=500)


# =========================
# TOATE ENDPOINT-URILE (100% INTEGRAL)
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
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_description") or data.get("job_text") or ""

        # Filtrare + Trunchiere inteligentă la 5000 caractere max
        cv = truncate_smart(cv_raw, max_chars=5000)

        if not cv:
            return api_response(error="CV-ul lipsește. Încarcă mai întâi fișierul PDF.", code=400)

        MEMORY["cv_text"] = cv

        context_extra = ""
        if job_raw:
            context_extra = f"\n\nJOB DESCRIPTION:\n{truncate_smart(job_raw, max_chars=2000)}"

        chunks = chunk_text(cv, chunk_size=2500)

        clarity_scores, relevance_scores, structure_scores = [], [], []
        raw_improvements, raw_rephrasings = [], []

        for chunk in chunks:
            try:
                prompt_chunk = f"""
You are a senior recruiter. Analyze ONLY the CV fragment below.

RULES:
1. Output MUST be written STRICTLY IN ROMANIAN.
2. Provide specific, actionable recommendations. Do NOT repeat yourself.
3. For "suggested_rephrasings" use format: "Original: \"...\", Improved: \"...\""
4. Return ONLY valid JSON.

JSON structure:
{{
  "clarity_score": 8,
  "relevance_score": 7,
  "structure_score": 8,
  "concrete_improvements": ["sugestie 1"],
  "suggested_rephrasings": ["Original: \"...\", Improved: \"...\""]
}}

CV fragment:
{chunk}
{context_extra}
"""
                raw_chunk = ask_ai(prompt_chunk, force_json=True)
                parsed_chunk = safe_json(raw_chunk)

                if not parsed_chunk or not isinstance(parsed_chunk, dict):
                    continue

                clarity_scores.append(parsed_chunk.get("clarity_score", 8))
                relevance_scores.append(parsed_chunk.get("relevance_score", 8))
                structure_scores.append(parsed_chunk.get("structure_score", 8))

                imp = parsed_chunk.get("concrete_improvements", [])
                if isinstance(imp, list):
                    raw_improvements.extend(imp)

                reph = parsed_chunk.get("suggested_rephrasings", [])
                if isinstance(reph, list):
                    raw_rephrasings.extend(reph)
            except Exception as chunk_err:
                print(f"⚠️ Eroare la procesare chunk: {str(chunk_err)}")

        unique_improvements = list(dict.fromkeys([optimize_cv_text(str(i)) for i in raw_improvements if i]))
        unique_rephrasings = list(dict.fromkeys([optimize_cv_text(str(i)) for i in raw_rephrasings if i]))

        final_payload = {
            "clarity_score": int(sum(clarity_scores) / len(clarity_scores)) if clarity_scores else 8,
            "relevance_score": int(sum(relevance_scores) / len(relevance_scores)) if relevance_scores else 8,
            "structure_score": int(sum(structure_scores) / len(structure_scores)) if structure_scores else 8,
            "overall_assessment": "Analiza CV-ului a fost finalizată cu succes.",
            "concrete_improvements": unique_improvements[:6] if unique_improvements else [
                "Adaugă realizări cuantificabile și metrici în experiența profesională.",
                "Evidențiază mai bine tehnologiile folosite la fiecare job."
            ],
            "suggested_rephrasings": unique_rephrasings[:6]
        }

        return api_response(payload=final_payload)

    except Exception as e:
        print(f"❌ Error inside analyze_cv_quality: {str(e)}")
        return api_response(error=f"Eroare internă: {str(e)}", code=500)


@app.route("/analyze-cv", methods=["POST", "OPTIONS"])
@cross_origin()
def analyze_cv():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_clean = truncate_smart(data.get("cv_text") or MEMORY.get("cv_text") or "", 5000)
        job_clean = truncate_smart(data.get("job_description") or data.get("job_text") or "", 2000)

        if not cv_clean or not job_clean:
            return api_response(error="Atât CV-ul cât și jobul sunt necesare.", code=400)

        MEMORY["cv_text"] = cv_clean

        prompt_chunk = f"""
Ești un recruiter profesionist. Analizează potrivirea dintre CV și cerințele postului.
Returnează un procent de compatibilitate (0-100) și un feedback scurt în format JSON:

{{"compatibility_percent": 80, "feedback_markdown": "text feedback"}}

CV: {cv_clean}
Job: {job_clean}
"""
        raw_res = ask_ai(prompt_chunk, force_json=True)
        parsed = safe_json(raw_res) or {
            "compatibility_percent": 75,
            "feedback_markdown": "CV-ul conține experiențe relevante pentru cerințele menționate."
        }

        return api_response(payload=parsed)

    except Exception as e:
        return api_response(error=f"Eroare la potrivire CV: {str(e)}", code=500)


@app.route("/reframe-cv", methods=["POST", "OPTIONS"])
@app.route("/api/reframe-cv", methods=["POST", "OPTIONS"])
@cross_origin()
def reframe_cv():
    """ Rescrie și optimizează textul din CV pentru a corespunde descrierii jobului """
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = truncate_smart(data.get("cv_text") or MEMORY.get("cv_text") or "", 4000)
        job_raw = truncate_smart(data.get("job_description") or data.get("job_text") or "", 2000)

        if not cv_raw:
            return api_response(error="CV-ul este necesar pentru optimizare.", code=400)

        prompt = f"""
Ești un expert ATS și Resume Writer. Optimizează și rescrie punctele din CV pentru a evidenția abilitățile relevante pentru job.

CV:
{cv_raw}

JOB DESCRIPTION:
{job_raw}

Răspunde în limba română sub formă de Markdown structurat, împărțit pe secțiuni (Experiență, Abilități, Sumar).
"""
        reframed_text = ask_ai(prompt)
        return api_response(payload={"reframed_cv": reframed_text})

    except Exception as e:
        return api_response(error=f"Eroare la optimizarea CV-ului: {str(e)}", code=500)


@app.route("/generate-cover-letter", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_cover_letter():
    """ Generează o scrisoare de intenție personalizată """
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = truncate_smart(data.get("cv_text") or MEMORY.get("cv_text") or "", 3500)
        job_raw = truncate_smart(data.get("job_description") or data.get("job_text") or "", 2000)

        if not cv_raw or not job_raw:
            return api_response(error="CV-ul și descrierea jobului sunt obligatorii.", code=400)

        prompt = f"""
Ești un expert în scrierea scrisorilor de intenție (Cover Letters).
Creează o scrisoare de intenție adaptată perfect pentru jobul descris, bazată pe experiența din CV.

CV:
{cv_raw}

JOB:
{job_raw}
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
        cv_raw = truncate_smart(data.get("cv_text") or MEMORY.get("cv_text") or "", 3500)
        job_raw = truncate_smart(data.get("job_description") or data.get("job_text") or "", 2000)

        if not cv_raw:
            return api_response(error="CV-ul este necesar pentru generarea întrebărilor de interviu.", code=400)

        prompt = f"""
Ești un Hiring Manager riguros. Generează un set de 5 întrebări tehnice și comportamentale de interviu bazate pe CV și job.

Returnează STRICT un JSON valid cu următoarea structură:
{{
  "questions": [
    {{"id": 1, "question": "întrebare...", "category": "Tehnic / Comportamental"}},
    {{"id": 2, "question": "întrebare...", "category": "Tehnic / Comportamental"}}
  ]
}}

CV:
{cv_raw}

JOB:
{job_raw}
"""
        raw_res = ask_ai(prompt, force_json=True)
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
