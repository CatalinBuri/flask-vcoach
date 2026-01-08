import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
import orjson
from flask_compress import Compress
from groq import Groq
from itertools import zip_longest

# =========================
# CONFIG
# =========================
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"           # recomandat în 2026 – rapid și bun la instrucțiuni
# MODEL_NAME = "gemini-1.5-flash-latest"  # dacă vrei varianta mai veche

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
Compress(app)

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
        print(f"✅ Gemini (new SDK) ready | model: {MODEL_NAME}")
    except Exception as e:
        print(f"❌ Eroare la inițializarea Gemini Client: {str(e)}")
        gemini_client = None

groq_client = None
if USE_GROQ:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq ready (rapid și gratuit)")
    except Exception as e:
        print(f"❌ Eroare la inițializarea Groq: {str(e)}")
        groq_client = None


def groq_text(prompt: str) -> str:
    """
    Trimite prompt-ul la Groq și returnează textul complet.
    Dacă apare eroare, returnează string gol.
    """
    if not groq_client:
        print("Groq client nu e inițializat!")
        return ""
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ești un expert LinkedIn Job Search și recruiter profesionist. "
                        "Răspunde NUMAI cu JSON valid atunci când se solicită."
                    )
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
    return app.response_class(
        orjson.dumps({
            "status": "ok" if not error else "error",
            "payload": payload,
            "error": error
        }),
        status=code,
        mimetype="application/json"
    )


def clean_text(text: str) -> str:
    """Curăță textul eliminând spații multiple și caractere de control, păstrând diacritice."""
    text = re.sub(r'\s+', ' ', text)           # spații multiple → 1 spațiu
    text = re.sub(r'[\x00-\x1F]+', '', text)   # caractere de control
    return text.strip()


def chunk_text(text: str, chunk_size: int = 2000) -> list:
    """Împarte textul în bucăți de lungime maximă chunk_size."""
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
    except:
        # încercăm să extragem primul obiect JSON valid
        match = re.search(r"\{.*\}", text, re.S | re.M)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None


def gemini_text(prompt: str) -> str:
    """Prioritate Groq (mai rapid), fallback Gemini (new SDK)."""
    if USE_GROQ and groq_client:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ești un recrutor profesionist cu peste 10 ani de experiență umană, asistat de inteligență artificială avansată. "
                            "Îmbină empatia, intuiția și feedback-ul constructiv uman cu analiza obiectivă, riguroasă și bazată pe date a unui sistem AI. "
                            "Răspunde mereu cu maxim de profesionalism, obiectivitate și motivație pentru dezvoltarea candidatului. "
                            "Respectă EXACT instrucțiunile: dacă ți se cere JSON, returnează NUMAI JSON valid, fără text suplimentar, fără ```, fără markdown. "
                            "Dacă ți se cere text simplu, răspunde NUMAI cu text curat, fluent și profesionist în română, fără asteriscuri, bold, liste marcate sau alte elemente de formatare."
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
            return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {type(e).__name__} - {str(e)}")

    return ""


# =========================
# ROUTES
# =========================
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"})


@app.route("/process-text", methods=["POST"])
def process_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return api_response(error="Text lipsă", code=400)

    prompt = f"""
Realizează un rezumat clar, concis și extrem de profesionist al textului următor.
Răspunde NUMAI cu rezumatul, fără introduceri, titluri sau orice formatare.
Text de rezumat:
{text}
"""
    summary = clean_text(gemini_text(prompt))
    return api_response(payload={"t": summary})


@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json(force=True)
    cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
    job = data.get("job_summary", "").strip()
    cv = clean_text(cv_raw)
    if not cv or not job:
        return api_response(error="Date lipsă", code=400)

    MEMORY["cv_text"] = cv

    prompt = f"""
Ești un recrutor profesionist hibrid (experiență umană + AI avansată).
Generează exact 5 întrebări de interviu relevante, profesionale și bine țintite, bazate pe CV și descrierea postului.
Îmbină întrebări comportamentale și de motivare (perspectivă umană) cu întrebări tehnice și de competențe măsurabile (rigurozitate AI).
Formulează-le în română corectă, naturală și profesională.
Returnează NUMAI JSON valid cu structura exactă:
{{"questions": ["întrebare 1", "întrebare 2", "întrebare 3", "întrebare 4", "întrebare 5"]}}
CV:
{cv}
Descrierea postului:
{job}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)
    if not parsed or "questions" not in parsed or len(parsed.get("questions", [])) != 5:
        parsed = {
            "questions": [
                "Povestiți despre experiența dumneavoastră cea mai relevantă pentru acest post.",
                "Care considerați că sunt principalele dumneavoastră puncte forte în relație cu cerințele rolului?",
                "Descrieți o situație challenging din carieră și modul în care ați gestionat-o.",
                "Ce vă motivează să aplicați pentru această poziție în compania noastră?",
                "Cum abordați învățarea continuă și adaptarea la tehnologii noi?"
            ]
        }
    return api_response(payload=parsed)


@app.route("/analyze-cv", methods=["POST"])
def analyze_cv():
    """
    Analizează compatibilitatea unui CV cu descrierea jobului.
    """
    try:
        data = request.get_json(force=True)
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_text", "").strip()

        if not cv_raw or not job_raw:
            return api_response(error="Date lipsă: CV și Job sunt necesare", code=400)

        cv_clean = clean_text(cv_raw)
        job_clean = clean_text(job_raw)
        MEMORY["cv_text"] = cv_clean

        print("DEBUG /analyze-cv - CV length:", len(cv_clean), "Job length:", len(job_clean))

        cv_chunks = chunk_text(cv_clean, chunk_size=3000)
        job_chunks = chunk_text(job_clean, chunk_size=3000)

        chunk_feedbacks = []
        chunk_scores = []

        for cv_chunk, job_chunk in zip_longest(cv_chunks, job_chunks, fillvalue=""):
            prompt_chunk = f"""
Ești un recrutor profesionist cu experiență umană + AI.
Analizează compatibilitatea dintre CV și cerințele postului.
Oferă procentaj realist (0-100) și feedback detaliat pentru acest fragment.
Returnează NUMAI JSON valid:
{{"compatibility_percent": număr_întreg, "feedback_markdown": "text curat și profesionist"}}
CV fragment:
{cv_chunk}
Job fragment:
{job_chunk}
"""
            raw_chunk = gemini_text(prompt_chunk)
            parsed_chunk = safe_json(raw_chunk)

            if not parsed_chunk or "compatibility_percent" not in parsed_chunk or "feedback_markdown" not in parsed_chunk:
                parsed_chunk = {
                    "compatibility_percent": 70,
                    "feedback_markdown": "Fragmentul CV-ului are relevanță parțială pentru cerințele acestui segment al jobului."
                }

            chunk_feedbacks.append(parsed_chunk["feedback_markdown"])
            chunk_scores.append(parsed_chunk["compatibility_percent"])

        combined_feedback = "\n\n".join(chunk_feedbacks)

        final_prompt = f"""
Ai primit mai multe feedback-uri parțiale despre compatibilitatea unui CV cu descrierea unui job.
Rescrie-le într-un text **profesionist, fluent și corect gramatical**, în română, fără erori.
Text combinat din AI:
{combined_feedback}
Returnează NUMAI text curat, profesional și coerent.
"""
        res_final = gemini_text(final_prompt)
        if not res_final.strip():
            res_final = combined_feedback

        final_score = int(sum(chunk_scores) / len(chunk_scores)) if chunk_scores else 75

        payload = {
            "compatibility_percent": final_score,
            "feedback_markdown": res_final
        }

        return api_response(payload=payload)

    except Exception as e:
        print("ERROR /analyze-cv:", str(e))
        return api_response(error=f"Eroare internă: {str(e)}", code=500)


@app.route("/analyze-cv-quality", methods=["POST"])
def analyze_cv_quality():
    data = request.get_json(force=True)
    cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
    cv = clean_text(cv_raw)
    if not cv:
        return api_response(error="CV lipsă", code=400)

    MEMORY["cv_text"] = cv

    chunks = chunk_text(cv, chunk_size=3000)

    clarity_scores = []
    relevance_scores = []
    structure_scores = []
    concrete_improvements = []
    suggested_rephrasings = []

    for chunk in chunks:
        prompt_chunk = f"""
You are a senior hybrid recruiter (human experience + AI). Analyze the CV fragment below.
Instructions (STRICT):
1. Detect the language of the fragment.
2. All scores, improvement suggestions, and rephrasings must be in the same language as the fragment.
3. Assign scores from 0 to 10:
   - clarity_score: clarity and ease of understanding
   - relevance_score: relevance for recruiters
   - structure_score: structure and logical flow
4. Suggest 2-3 concrete improvements, each with an example, in the fragment's original language.
5. Suggest 2-3 rephrased sentences, showing Original / New, in the fragment's original language.
6. Do NOT translate anything. Preserve the original language.
7. Return ONLY valid JSON, no extra text or commentary.
Expected JSON structure:
{{
    "clarity_score": integer,
    "relevance_score": integer,
    "structure_score": integer,
    "concrete_improvements": ["Improvement 1 example...", "Improvement 2 example..."],
    "suggested_rephrasings": ["Rephrasing 1: Original: '...', New: '...'", "Rephrasing 2: Original: '...', New: '...'"]
}}
CV fragment:
{chunk}
"""
        raw_chunk = gemini_text(prompt_chunk)
        parsed_chunk = safe_json(raw_chunk)

        if not parsed_chunk:
            parsed_chunk = {
                "clarity_score": 7,
                "relevance_score": 7,
                "structure_score": 7,
                "concrete_improvements": [],
                "suggested_rephrasings": []
            }

        clarity_scores.append(parsed_chunk.get("clarity_score", 7))
        relevance_scores.append(parsed_chunk.get("relevance_score", 7))
        structure_scores.append(parsed_chunk.get("structure_score", 7))
        concrete_improvements.extend(parsed_chunk.get("concrete_improvements", []))
        suggested_rephrasings.extend(parsed_chunk.get("suggested_rephrasings", []))

    final_payload = {
        "clarity_score": int(sum(clarity_scores) / len(clarity_scores)) if clarity_scores else 0,
        "relevance_score": int(sum(relevance_scores) / len(relevance_scores)) if relevance_scores else 0,
        "structure_score": int(sum(structure_scores) / len(structure_scores)) if structure_scores else 0,
        "overall_assessment": "CV-ul a fost analizat și scorurile au fost calculate.",
        "concrete_improvements": concrete_improvements[:10],     # max 10 sugestii
        "suggested_rephrasings": suggested_rephrasings[:10]      # max 10 propuneri
    }

    return api_response(payload=final_payload)


# ... restul endpoint-urilor (generate-job-queries, optimize-linkedin-profile, etc.)
# le poți adăuga la fel, indentarea este deja corectată în cele de mai sus


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
