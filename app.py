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
CORS(app, resources={r"/*": {"origins": "*"}})
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
    """Curăță textul eliminând spații multiple și caractere non-ASCII."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
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
    - Curăță textul (păstrând diacriticele)
    - Împarte textul lung în chunk-uri dacă e necesar
    - Returnează JSON cu scor și feedback fluent, corect gramatical
    """
    try:
        data = request.get_json(force=True)
        cv_raw = data.get("cv_text", "").strip()
        job_raw = data.get("job_text", "").strip()

        if not cv_raw or not job_raw:
            return api_response(error="Date lipsă: CV și Job sunt necesare", code=400)

        # =========================
        # CLEAN TEXT - păstrăm diacritice
        # =========================
        def clean_text_utf8(text: str) -> str:
            # Eliminăm doar caractere neprintabile, păstrăm diacriticele
            return "".join(c for c in text if c.isprintable()).strip()

        cv_clean = clean_text_utf8(cv_raw)
        job_clean = clean_text_utf8(job_raw)

        print("DEBUG /analyze-cv - CV length:", len(cv_clean), "Job length:", len(job_clean))

        # =========================
        # CHUNKING
        # =========================
        def chunk_text(text: str, max_len: int = 3000) -> list:
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + max_len, len(text))
                chunks.append(text[start:end])
                start = end
            return chunks

        cv_chunks = chunk_text(cv_clean)
        job_chunks = chunk_text(job_clean)

        # =========================
        # COMBINARE CHUNK-URI
        # =========================
        combined_prompt_text = ""
        for cv_chunk, job_chunk in zip(cv_chunks, job_chunks):
            combined_prompt_text += f"CV:\n{cv_chunk}\n\nJob:\n{job_chunk}\n\n"

        # =========================
        # PROMPT FINAL - instrucțiuni clare pentru text fluent
        # =========================
        model_prompt = f"""
Ești un recrutor profesionist hibrid. Analizează compatibilitatea dintre CV și cerințele postului.
Oferă procentaj realist (0-100) și feedback detaliat, profesionist, motivant.
Scrie **text fluent, corect gramatical, în română**, cu toate diacriticele.
Nu folosi prescurtări sau formulări agramate.
Returnează NUMAI JSON valid:
{{"compatibility_percent": număr_întreg, "feedback_markdown": "text curat și profesionist"}}

{combined_prompt_text}
"""

        # =========================
        # CERERE LA MODEL
        # =========================
        raw = gemini_text(model_prompt)
        print("DEBUG /analyze-cv - Raw AI response:", raw)

        # =========================
        # PARSARE JSON
        # =========================
        parsed = safe_json(raw)
        if not parsed or "compatibility_percent" not in parsed or "feedback_markdown" not in parsed:
            print("DEBUG /analyze-cv - Fallback JSON")
            parsed = {
                "compatibility_percent": 75,
                "feedback_markdown": (
                    "CV-ul prezintă o aliniere bună cu cerințele postului. "
                    "Se recomandă evidențierea mai clară a rezultatelor cuantificabile și a proiectelor relevante."
                )
            }

        # =========================
        # OPTIONAL: POLISH FINAL (dacă vrei să refacem textul după chunking)
        # =========================
        final_prompt = f"""
        Reformulează următorul feedback într-un text fluent si profesional, corect din punct de vedere gramatical si al acordurilor de gen și persoana; folosește diacritice:
        {parsed['feedback_markdown']}
        """
        parsed['feedback_markdown'] = gemini_text(final_prompt) or parsed['feedback_markdown']

        return api_response(payload=parsed)

    except Exception as e:
        print("ERROR /analyze-cv:", str(e))
        return api_response(error=f"Eroare internă: {str(e)}", code=500)



@app.route("/generate-job-queries", methods=["POST"])
def generate_job_queries():
    """
    Generează 7 interogări de job, câte un item de căutare pentru fiecare,
    folosind modelul AI (Gemini/Groq).
    Fiecare query trebuie să fie un singur item de căutare, fără OR sau splituri multiple.
    """
    try:
        data = request.get_json(force=True)
        cv_raw = data.get("cv_text", "").strip()
        if not cv_raw:
            return api_response(error="CV lipsă", code=400)

        cv_clean = clean_text(cv_raw)
        cv_chunks = chunk_text(cv_clean)  # chunking dacă e text lung

        # Promptul AI
        model_prompt = f"""
Ești un expert în LinkedIn Job Search și recrutare. Pe baza CV-ului următor, generează **exact 7 interogări de job**, 
fiecare având **un singur item de căutare** (rol sau denumire job conforma cu experienta descrisa in cv). Nu folosi OR sau combinații multiple.
Răspunde NUMAI cu JSON valid:
{{"queries": ["query1", "query2", "query3", "query4", "query5", "query6", "query7"]}}

CV:
{cv_clean}
"""

        # =========================
        # Apel Groq
        # =========================
        raw = ""
        if USE_GROQ and groq_client:
            raw = groq_text(model_prompt)
            print("DEBUG /generate-job-queries - raw Groq response:", raw)
            if not raw:
                print("DEBUG /generate-job-queries - fallback la Gemini...")

        if not raw and gemini_client:
            raw = gemini_text(model_prompt)
            print("DEBUG /generate-job-queries - raw Gemini response:", raw)

        parsed = safe_json(raw)
        if not parsed or "queries" not in parsed or not isinstance(parsed["queries"], list) or len(parsed["queries"]) != 7:
            # fallback simplu: extragem primele 7 cuvinte din CV
            words = re.findall(r'\b[A-Z][a-zA-Z0-9\+\#&]+\b', cv_clean)
            queries = words[:7] if len(words) >= 7 else words + ["ExtraSkill"]*(7-len(words))
            parsed = {"queries": queries}

        return api_response(payload=parsed)

    except Exception as e:
        print("ERROR /generate-job-queries:", str(e))
        return api_response(error=f"Excepție la server: {str(e)}", code=503)


# =========================
# Restul route-urilor rămân nemodificate
# =========================

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


# =========================
# START
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


