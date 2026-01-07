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
# MODEL_NAME = "gemini-1.5-flash-latest"  # dacă vrei varianta mai veche (încă funcționează pe unele conturi)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
Compress(app)

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
    cv = data.get("cv_text", "").strip()
    job = data.get("job_summary", "").strip()
    if not cv or not job:
        return api_response(error="Date lipsă", code=400)
    
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
    - Curăță textul (clean_text)
    - Împarte textul lung în chunk-uri dacă e necesar
    - Generează scor și feedback pentru fiecare chunk
    - Reunește și rescrie feedback-ul într-un text coerent, fluent, corect gramatical
    """
    try:
        data = request.get_json(force=True)
        cv_raw = data.get("cv_text", "").strip()
        job_raw = data.get("job_text", "").strip()

        if not cv_raw or not job_raw:
            return api_response(error="Date lipsă: CV și Job sunt necesare", code=400)

        # =========================
        # CLEAN TEXT
        # =========================
        cv_clean = clean_text(cv_raw)
        job_clean = clean_text(job_raw)

        print("DEBUG /analyze-cv - CV length:", len(cv_clean), "Job length:", len(job_clean))

        # =========================
        # CHUNKING
        # =========================
        cv_chunks = chunk_text(cv_clean, chunk_size=3000)
        job_chunks = chunk_text(job_clean, chunk_size=3000)

        # =========================
        # FEEDBACK PER CHUNK
        # =========================
        chunk_feedbacks = []
        chunk_scores = []

        for i, (cv_chunk, job_chunk) in enumerate(zip(cv_chunks, job_chunks)):
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

            # fallback dacă AI nu răspunde corect
            if not parsed_chunk or "compatibility_percent" not in parsed_chunk or "feedback_markdown" not in parsed_chunk:
                parsed_chunk = {
                    "compatibility_percent": 70,
                    "feedback_markdown": "Fragmentul CV-ului are relevanță parțială pentru cerințele acestui segment al jobului."
                }

            chunk_feedbacks.append(parsed_chunk["feedback_markdown"])
            chunk_scores.append(parsed_chunk["compatibility_percent"])

        # =========================
        # COMBINĂ ȘI RESCRIE FLUENT
        # =========================
        combined_feedback = "\n\n".join(chunk_feedbacks)

        final_prompt = f"""
Ai primit mai multe feedback-uri parțiale despre compatibilitatea unui CV cu descrierea unui job.
Rescrie-le într-un text **profesionist, fluent și corect gramatical**, în română, fără erori, fără propoziții tăiate sau fără diacritice lipsă.
Text combinat din AI:
{combined_feedback}
Returnează NUMAI text curat, profesional și coerent.
"""
        res_final = gemini_text(final_prompt)
        # fallback dacă rescrierea eșuează
        if not res_final.strip():
            res_final = combined_feedback  # măcar combinarea brută

        # =========================
        # SCOR FINAL MEDIU
        # =========================
        if chunk_scores:
            final_score = int(sum(chunk_scores) / len(chunk_scores))
        else:
            final_score = 75

        payload = {
            "compatibility_percent": final_score,
            "feedback_markdown": res_final
        }

        return api_response(payload=payload)

    except Exception as e:
        print("ERROR /analyze-cv:", str(e))
        return api_response(error=f"Eroare internă: {str(e)}", code=500)



@app.route("/generate-job-queries", methods=["POST"])
def generate_job_queries():
    """
    Generează exact 7 căutări de joburi compatibile cu profilul candidatului.
    - FĂRĂ OR
    - FIECARE query = un singur titlu / rol
    - Dacă CV-ul este prea nișat / ambiguu → mesaj clar, fără rezultate false
    """
    try:
        data = request.get_json(force=True)
        cv_raw = data.get("cv_text", "").strip()

        if not cv_raw:
            return api_response(error="CV lipsă", code=400)

        cv_clean = clean_text(cv_raw)

        prompt = f"""
Ești un expert senior în recrutare și LinkedIn Job Search (2026).

Analizează EXCLUSIV CV-ul de mai jos și stabilește dacă experiența candidatului
poate fi asociată CLAR cu roluri standard existente pe platforme de joburi
(LinkedIn, Indeed, Glassdoor).

REGULI CRITICE:
- NU inventa roluri
- NU forța potriviri
- NU oferi alternative dacă nu există o asociere clară
- NU folosi OR, paranteze sau combinații
- FIECARE căutare trebuie să conțină UN SINGUR titlu de job standard (engleză)

Dacă NU poți identifica MINIM 3 roluri clare și realiste,
returnează EXACT acest JSON și NIMIC altceva:

{{"status": "no_clear_match", "message": "Experiența candidatului este prea nișată sau formulată într-un mod care nu permite asocierea clară cu roluri standard de pe platformele de joburi."}}

Dacă POȚI identifica roluri clare, returnează EXACT 7 căutări:

{{"queries": [
  "Job Title 1",
  "Job Title 2",
  "Job Title 3",
  "Job Title 4",
  "Job Title 5",
  "Job Title 6",
  "Job Title 7"
]}}

CV:
{cv_clean}
"""

        # =========================
        # Apel Groq (prioritar)
        # =========================
        raw = ""
        if USE_GROQ and groq_client:
            raw = groq_text(prompt)
            print("DEBUG /generate-job-queries - raw Groq response:", raw)

        # =========================
        # Fallback Gemini
        # =========================
        if not raw and gemini_client:
            raw = gemini_text(prompt)
            print("DEBUG /generate-job-queries - raw Gemini response:", raw)

        parsed = safe_json(raw)
        print("DEBUG /generate-job-queries - parsed:", parsed)

        # =========================
        # Caz: NU există potrivire clară
        # =========================
        if parsed and parsed.get("status") == "no_clear_match":
            return api_response(payload=parsed)

        # =========================
        # Validare strictă rezultate
        # =========================
        if (
            not parsed
            or "queries" not in parsed
            or not isinstance(parsed["queries"], list)
            or len(parsed["queries"]) != 7
        ):
            return api_response(
                payload={
                    "status": "no_clear_match",
                    "message": (
                        "Experiența candidatului este prea nișată sau formulată într-un mod "
                        "care nu permite asocierea clară cu roluri standard de pe platformele de joburi."
                    )
                }
            )

        return api_response(payload={"queries": parsed["queries"]})

    except Exception as e:
        print("ERROR /generate-job-queries:", str(e))
        return api_response(
            error=f"Eroare internă server: {str(e)}",
            code=503
        )

@app.route("/optimize-linkedin-profile", methods=["POST"])
def optimize_linkedin():
    data = request.get_json(force=True)
    cv = data.get("cv_text", "").strip()
    if not cv:
        return api_response(error="CV lipsă", code=400)
    
    prompt = f"""
Ești un recrutor profesionist hibrid (uman + AI). Optimizează profilul LinkedIn pe baza CV-ului.
Propune exact 5 headline-uri atractive, profesionale și concise, care combină povestea personală cu cuvinte-cheie esențiale pentru algoritmul LinkedIn.
Scrie o secțiune About captivantă, autentică și profesională (300-500 cuvinte), care îmbină narativul uman (pasiune, valori, parcurs) cu optimizări AI (structură clară, rezultate măsurabile, SEO).
Returnează NUMAI JSON valid:
{{"linkedin_headlines": ["headline 1", "headline 2", "headline 3", "headline 4", "headline 5"], "linkedin_about": "text complet About"}}
CV:
{cv}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)
    if not parsed:
        parsed = {
            "linkedin_headlines": [
                "Senior Software Engineer | Python & AI Enthusiast",
                "Full Stack Developer | Building Scalable Solutions",
                "Tech Lead | Driving Innovation & Team Growth",
                "Data Engineer | Transforming Data into Insights",
                "Software Architect | 10+ Years Crafting Robust Systems"
            ],
            "linkedin_about": "Profil LinkedIn optimizat profesional pe baza experienței dumneavoastră."
        }
    return api_response(payload=parsed)


@app.route("/coach-next", methods=["POST"])
def coach_next():
    data = request.get_json(force=True)
    answer = data.get("user_answer", "").strip()
    if len(answer.split()) < 5:
        return api_response(payload={"star_answer": "Răspunsul este prea scurt pentru a fi restructurat în format STAR."})
    
    prompt = f"""
Ești un recrutor profesionist hibrid. Rescrie răspunsul candidatului în structura STAR (Situație, Sarcină, Acțiune, Rezultat), păstrând toate detaliile esențiale.
Folosește un limbaj profesionist, fluent, natural și empatic, care reflectă atât rigurozitatea structurii, cât și autenticitatea umană.
Răspuns original:
{answer}
"""
    text = clean_text(gemini_text(prompt))
    return api_response(payload={"star_answer": text})


@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    if not question or not answer:
        return api_response(error="Date lipsă", code=400)
    
    prompt = f"""
Ești un recrutor profesionist hibrid (experiență umană + analiză AI).
Evaluează răspunsul candidatului pe o scară de la 1 la 10 și oferă feedback detaliat, obiectiv, constructiv și motivant.
Îmbină intuiția umană (claritate, autenticitate, impact emoțional) cu rigurozitatea AI (structură, relevanță, exemple concrete).
Returnează NUMAI JSON valid:
{{"nota_finala": număr_întreg_de_la_1_la_10, "feedback": "feedback text curat și profesionist"}}
Întrebarea:
{question}
Răspunsul:
{answer}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)
    if not parsed or "nota_finala" not in parsed:
        parsed = {
            "nota_finala": 7,
            "feedback": "Răspunsul este solid și demonstrează experiență relevantă. Pentru un impact mai puternic, recomandăm includerea unor rezultate cuantificabile și o structură mai clară (ex: STAR)."
        }
    return api_response(payload={"current_evaluation": parsed})


@app.route("/generate-report", methods=["POST"])
def generate_report():
    data = request.get_json(force=True)
    history = data.get("history", [])
    if not history:
        return api_response(error="Istoric lipsă", code=400)
    
    prompt = f"""
Ești un recrutor profesionist hibrid. Analizează întregul istoric al interviului și generează un raport final obiectiv și profesionist.
Include un rezumat al performanței candidatului și un scor general (1-10), îmbinând empatia umană cu analiza detaliată AI.
Returnează NUMAI JSON valid:
{{"summary": "rezumat detaliat și profesionist", "scor_final": număr_întreg_de_la_1_la_10}}
Istoric interviu:
{history}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)
    if not parsed:
        parsed = {
            "summary": "Candidatul a demonstrat competențe solide și o atitudine profesionistă pe parcursul interviului. Există potențial ridicat, cu recomandări minore de îmbunătățire a structurii răspunsurilor.",
            "scor_final": 8
        }
    return api_response(payload=parsed)

@app.route("/reformulate-cv-for-job-boards", methods=["POST"])
def reformulate_cv_for_job_boards():
    """
    Reformulează CV-ul pentru job boards (LinkedIn, Indeed, ATS).
    - Normalizează titluri
    - Clarifică senioritatea
    - Extrage keywords ATS
    - NU inventează experiență
    """
    try:
        data = request.get_json(force=True)

        cv_raw = data.get("cv_text", "").strip()
        job_raw = data.get("job_text", "").strip()  # opțional, NU obligatoriu

        if not cv_raw:
            return api_response(error="CV lipsă", code=400)

        cv_clean = clean_text(cv_raw)

        prompt = f"""
Ești un expert senior în recrutare internațională și sisteme ATS (2026).

Sarcina ta este să REFORMULEZI CV-ul candidatului pentru a fi:
- ușor de înțeles
- corect interpretat
- eficient pe platforme de joburi (LinkedIn, Indeed, Glassdoor)
- compatibil cu ATS-uri automate

REGULI STRICTE (OBLIGATORII):
- NU inventa experiență
- NU exagera senioritatea
- NU adăuga skill-uri care nu apar explicit sau logic în CV
- Păstrează realitatea profesională a candidatului
- Tradu titluri interne / nișate doar dacă există corespondență clară
- Dacă NU există corespondență, MENȚIONEAZĂ EXPLICIT acest lucru

STRUCTURA DE RĂSPUNS (JSON STRICT – fără text suplimentar):

{{
  "normalized_titles": [
    "Titlu standard (dacă există)"
  ],
  "cv_summary_for_job_boards": "Rezumat profesionist, clar, ATS-friendly (max 120 cuvinte)",
  "core_skills_keywords": [
    "keyword ATS 1",
    "keyword ATS 2"
  ],
  "notes_for_candidate": "Observații oneste despre limitări, ambiguități sau lipsă de mapare clară"
}}

CV:
{cv_clean}

Descriere job (opțional – dacă este relevantă):
{job_raw}
"""

        # =========================
        # Apel AI (Groq → Gemini)
        # =========================
        raw = ""

        if USE_GROQ and groq_client:
            raw = groq_text(prompt)
            app.logger.debug("DEBUG /reformulate-cv - raw Groq response")

        if not raw and gemini_client:
            raw = gemini_text(prompt)
            app.logger.debug("DEBUG /reformulate-cv - raw Gemini response")

        parsed = safe_json(raw)
        app.logger.debug(f"DEBUG /reformulate-cv - parsed: {parsed}")

        # =========================
        # Validare strictă rezultat
        # =========================
        required_keys = [
            "normalized_titles",
            "cv_summary_for_job_boards",
            "core_skills_keywords",
            "notes_for_candidate"
        ]

        if not parsed or not all(k in parsed for k in required_keys):
            return api_response(
                error="AI nu a putut genera un rezultat valid pentru reformularea CV-ului",
                code=503
            )

        return api_response(payload=parsed)

    except Exception as e:
        app.logger.exception("ERROR /reformulate-cv-for-job-boards")
        return api_response(
            error="Eroare internă server în reformularea CV-ului",
            code=503
        )


# =========================
# START
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)









