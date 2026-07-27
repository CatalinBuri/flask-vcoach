import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)

app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})
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
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1F]+', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 2000) -> list:
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
        match = re.search(r"\{.*\}", text, re.S | re.M)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None


def gemini_text(prompt: str) -> str:
    """Prioritate Groq (mai rapid), fallback Gemini."""
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
                            "CRITICAL: Detect and strictly adhere to the source document's native language when giving recommendations."
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
# ROUTES (TOATE CELE 14 ENDPOINTURI)
# =========================

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"})


@app.route("/check-cv-memory", methods=["GET"])
@cross_origin(origins="*", methods=["POST", "OPTIONS", "GET"])
def check_cv_memory():
    if MEMORY.get("cv_text") and len(MEMORY["cv_text"].strip()) > 10:
        return api_response(payload={"has_cv": True}, code=200)
    else:
        return api_response(error="No CV in memory", code=404)


@app.route("/clear-memory", methods=["POST", "OPTIONS"])
@cross_origin()
def clear_memory():
    MEMORY["cv_text"] = None
    return jsonify({
        "status": "ok",
        "payload": {"message": "Memoria CV a fost ștearsă cu succes"}
    })


@app.route("/generate-coach-questions", methods=["POST"])
def generate_coach_questions():
    prompt = """
Ești un coach de interviu profesionist.
Generează EXACT 7 întrebări de interviu GENERALISTE, potrivite pentru ORICE candidat.
REGULI:
- Formulează în română profesională, clară și naturală.
- Returnează NUMAI JSON valid:
{
  "questions": [
    "întrebare 1", "întrebare 2", "întrebare 3", "întrebare 4", "întrebare 5", "întrebare 6", "întrebare 7"
  ]
}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)

    if not parsed or "questions" not in parsed or len(parsed["questions"]) != 7:
        parsed = {
            "questions": [
                "Unde te vezi din punct de vedere profesional peste 5 ani?",
                "Care consideri că este cel mai mare punct forte al tău?",
                "În ce domeniu simți că mai ai cel mai mult de crescut?",
                "Povestește despre o situație dificilă pe care ai gestionat-o la locul de muncă.",
                "Ce te motivează cel mai mult atunci când lucrezi într-o echipă?",
                "Cum recepționezi și aplici feedback-ul primit de la colegi sau manageri?",
                "Care sunt așteptările tale realiste de la următorul rol profesional?"
            ]
        }

    return api_response(payload=parsed)


@app.route("/coach-generic-eval", methods=["POST"])
def coach_generic_eval():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    answer = data.get("user_answer", "").strip()

    if not question or not answer:
        return api_response(error="Întrebare sau răspuns lipsă", code=400)

    if len(answer.split()) < 5:
        return api_response(payload={
            "feedback": "Răspunsul este prea scurt pentru o evaluare detaliată.",
            "improved_answer": "Dezvoltă-ți ideile cu exemple personale pentru a primi feedback complet și o variantă optimizată.",
            "nota_finala": 4
        })

    prompt = f"""
Ești un recrutor senior. Evaluează răspunsul candidatului la o întrebare generalistă de interviu.
Returnează NUMAI JSON valid:
{{
  "feedback": "text feedback (maxim 3 fraze)",
  "improved_answer": "răspunsul reformulat profesional",
  "nota_finala": <score>
}}

Întrebarea: {question}
Răspunsul candidatului: {answer}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)

    if not parsed or "feedback" not in parsed:
        parsed = {
            "feedback": "Răspunsul tău arată potențial și autenticitate. Adaugă un exemplu concret pentru un impact mai mare.",
            "improved_answer": answer,
            "nota_finala": 6
        }

    return api_response(payload=parsed)


@app.route("/process-text", methods=["POST"])
def process_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return api_response(error="Text lipsă", code=400)

    prompt = f"Realizează un rezumat clar și extrem de profesionist al textului următor:\n{text}"
    summary = clean_text(gemini_text(prompt))
    return api_response(payload={"t": summary})


@app.route("/analyze-cv-quality", methods=["POST"])
def analyze_cv_quality():
    data = request.get_json(force=True)
    cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
    cv = clean_text(cv_raw)
    if not cv:
        return api_response(error="CV lipsă", code=400)

    MEMORY["cv_text"] = cv
    chunks = chunk_text(cv, chunk_size=3000)

    clarity_scores, relevance_scores, structure_scores = [], [], []
    concrete_improvements, suggested_rephrasings = [], []

    for chunk in chunks:
        prompt_chunk = f"""
You are a senior hybrid recruiter with 10+ years of experience. Analyze ONLY the CV fragment below.

CRITICAL RULES - MUST FOLLOW EXACTLY:
1. Detect the dominant language of the fragment. ALL output (scores, concrete_improvements, suggested_rephrasings) MUST be written STRICTLY IN THAT LANGUAGE ONLY.
2. If the fragment is in English -> ALL output MUST BE IN ENGLISH ONLY. NEVER output Romanian sentences or prefixes like "Asigură", "Implementează", "Am realizat", "Nou:", etc.
3. Do NOT use numbering, prefixes, "Improvement 1:", "Rephrasing 1:", "1.", or bullet points inside the output array strings.
4. For "suggested_rephrasings" use EXACT format:
   "Original: \"exact original phrase\", Improved: \"better version\""
5. Return ONLY valid JSON — nothing else.

Assign scores 0–10:
- clarity_score: clarity & readability
- relevance_score: attractiveness to recruiters
- structure_score: logical flow & organization

JSON structure (strict):
{{
  "clarity_score": int,
  "relevance_score": int,
  "structure_score": int,
  "concrete_improvements": ["suggestion with example...", ...],
  "suggested_rephrasings": [
    "Original: \"...\", Improved: \"...\"",
    ...
  ]
}}

CV fragment:
{chunk}
"""
        raw_chunk = gemini_text(prompt_chunk)
        parsed_chunk = safe_json(raw_chunk)

        if not parsed_chunk:
            parsed_chunk = {
                "clarity_score": 7, "relevance_score": 7, "structure_score": 7,
                "concrete_improvements": [], "suggested_rephrasings": []
            }

        clarity_scores.append(parsed_chunk.get("clarity_score", 7))
        relevance_scores.append(parsed_chunk.get("relevance_score", 7))
        structure_scores.append(parsed_chunk.get("structure_score", 7))
        concrete_improvements.extend(parsed_chunk.get("concrete_improvements", []))
        suggested_rephrasings.extend(parsed_chunk.get("suggested_rephrasings", []))

    final_payload = {
        "clarity_score": int(sum(clarity_scores)/len(clarity_scores)) if clarity_scores else 0,
        "relevance_score": int(sum(relevance_scores)/len(relevance_scores)) if relevance_scores else 0,
        "structure_score": int(sum(structure_scores)/len(structure_scores)) if structure_scores else 0,
        "overall_assessment": "CV analysis completed successfully.",
        "concrete_improvements": concrete_improvements[:10],
        "suggested_rephrasings": suggested_rephrasings[:10]
    }

    return api_response(payload=final_payload)


@app.route("/analyze-cv", methods=["POST"])
def analyze_cv():
    try:
        data = request.get_json(force=True)
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_text", "").strip()

        if not cv_raw or not job_raw:
            return api_response(error="Date lipsă: CV și Job sunt necesare", code=400)

        cv_clean = clean_text(cv_raw)
        job_clean = clean_text(job_raw)
        MEMORY["cv_text"] = cv_clean

        cv_chunks = chunk_text(cv_clean, chunk_size=3000)
        job_chunks = chunk_text(job_clean, chunk_size=3000)

        chunk_feedbacks, chunk_scores = [], []

        for i, (cv_chunk, job_chunk) in enumerate(zip_longest(cv_chunks, job_chunks, fillvalue="")):
            prompt_chunk = f"""
Ești un recrutor profesionist. Analizează compatibilitatea dintre CV și cerințele postului.
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
Rescrie feedback-ul combinat într-un text profesionist, fluent și corect, în limba dominantă a analizei.
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
Ești un recrutor profesionist. Generează exact 5 întrebări de interviu relevante, profesionale și bine țintite.
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
            "Descrieți o situație challenging din carieră și modul în care ați gestionat-o.",
            "Ce vă motivează să aplicați pentru această poziție în compania noastră?",
            "Cum abordați învățarea continuă și adaptarea la tehnologii noi?"
        ]
    }
    return api_response(payload=parsed)


@app.route("/generate-job-queries", methods=["POST"])
def generate_job_queries():
    try:
        data = request.get_json(force=True)
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

        if parsed and parsed.get("status") == "no_clear_match":
            return api_response(payload=parsed)

        if not parsed or "queries" not in parsed or not isinstance(parsed["queries"], list) or len(parsed["queries"]) != 7:
            return api_response(payload={
                "status": "no_clear_match",
                "message": "Experiența candidatului este prea nișată sau formulată într-un mod care nu permite asocierea clară cu roluri standard."
            })

        return api_response(payload={"queries": parsed["queries"]})

    except Exception as e:
        return api_response(error=f"Eroare internă server: {str(e)}", code=503)


@app.route("/optimize-linkedin-profile", methods=["POST"])
def optimize_linkedin_profile():
    data = request.get_json(force=True)
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


@app.route("/coach-next", methods=["POST", "OPTIONS"])
@cross_origin()
def coach_next():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True)
        answer = data.get("user_answer", "").strip()

        if len(answer.split()) < 5:
            return api_response(payload={"star_answer": "Răspunsul este prea scurt pentru a fi restructurat în format STAR."})

        prompt = f"""
Ești un recrutor profesionist. Rescrie răspunsul candidatului în structura STAR (Situație, Sarcină, Acțiune, Rezultat).
Fiecare secțiune trebuie să înceapă pe un rând nou.

Răspuns original: {answer}

Format dorit:
SITUAȚIE: ...
SARCINĂ: ...
ACȚIUNE: ...
REZULTAT: ...
"""
        ai_response_text = gemini_text(prompt)
        return api_response(payload={"star_answer": clean_text(ai_response_text)})

    except Exception as e:
        return api_response(error="Eroare procesare STAR", code=500)


@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    data = request.get_json(force=True)
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

    if parsed and all(k in parsed for k in ("claritate", "structura", "relevanta")):
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


@app.route("/generate-report", methods=["POST"])
def generate_report():
    data = request.get_json(force=True)
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


@app.route("/reformulate-cv-for-job-boards", methods=["POST"])
def reformulate_cv_for_job_boards():
    try:
        data = request.get_json(force=True)
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_text", "").strip()

        cv_clean = clean_text(cv_raw)
        if not cv_clean:
            return api_response(error="CV lipsă", code=400)

        MEMORY["cv_text"] = cv_clean

        prompt = f"""
Ești un expert senior în recrutare internațională și sisteme ATS.
REFORMULEAZĂ CV-ul candidatului.

REGULI STRICTE:
- Detect the language of the CV and respond STRICTLY in that language (e.g., English CV -> English response).
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
        if not parsed or not all(k in parsed for k in required_keys):
            return api_response(error="AI nu a putut genera un rezultat valid pentru reformularea CV-ului", code=503)

        return api_response(payload=parsed)

    except Exception as e:
        return api_response(error="Eroare internă server", code=503)


# =========================
# START
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
