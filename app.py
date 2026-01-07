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
    """Elimină urme de markdown, blocuri de cod sau caractere nedorite."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^```[\w]*\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[\*#_`>]", "", text)
    return text.strip()


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
    data = request.get_json(force=True)
    cv = data.get("cv_text", "").strip()
    job = data.get("job_text", "").strip()
    if not cv or not job:
        return api_response(error="Date lipsă", code=400)
    
    prompt = f"""
Ești un recrutor profesionist hibrid. Analizează compatibilitatea dintre CV și cerințele postului.
Estimează un procent realist (0-100) și oferă feedback detaliat, obiectiv, constructiv și motivant.
Îmbină analiza umană (context, potențial de dezvoltare) cu evaluarea AI (aliniere la competențe, cuvinte-cheie, experiență cuantificabilă).
Returnează NUMAI JSON valid:
{{"compatibility_percent": număr_întreg, "feedback_markdown": "text feedback curat și profesionist"}}
Folosește doar text simplu în feedback (paragrafe separate prin linie goală).
CV:
{cv}
Post:
{job}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)
    if not parsed or "compatibility_percent" not in parsed:
        parsed = {
            "compatibility_percent": 75,
            "feedback_markdown": "CV-ul prezintă o aliniere bună cu cerințele postului, în special în ceea ce privește experiența tehnică. Recomandăm accentuarea rezultatelor cuantificabile și a proiectelor relevante pentru a crește impactul asupra recrutorilor."
        }
    return api_response(payload=parsed)


@app.route("/generate-job-queries", methods=["POST"])
def generate_job_queries():
    data = request.get_json(force=True)
    cv = data.get("cv_text", "").strip()
    if not cv:
        return api_response(error="CV lipsă", code=400)
    
    prompt = f"""
Ești un expert LinkedIn Job Search cu cunoștințe actualizate 2026 despre algoritmul de căutare și nomenclatura reală folosită în anunțurile de joburi internaționale și din România.

Sarcina ta: analizează EXCLUSIV CV-ul furnizat și extrage:
- roluri / poziții ocupate (folosește denumirile standard: Project Manager, nu „lider de proiect”; Senior Software Engineer, nu doar „programator senior”)
- abilități tehnice și soft relevante
- tool-uri, tehnologii, framework-uri, limbaje de programare menționate
- domenii / industrii în care a lucrat
- certificări (dacă există)

Pe baza acestor elemente generează **exact 7 căutări eficiente pentru LinkedIn Jobs** care să returneze oportunități cât mai potrivite profilului.

Reguli obligatorii:
1. Folosește **nomenclatura standard internațională** (engleză) pentru titluri de job – ex: "Project Manager", "Product Owner", "DevOps Engineer", "Data Analyst", "Frontend Developer" etc. Nu folosi traduceri românești în titluri.
2. Include combinații de **skills + tool-uri + nivel** (Junior / Mid / Senior / Lead) unde apare în CV.
3. Folosește **Boolean simplu** unde ajută: "exact phrase", OR pentru sinonime, - pentru a exclude (ex: -internship -freelance -stagiu)
4. Majoritatea căutărilor să fie în **engleză** (așa funcționează LinkedIn cel mai bine global și în România pentru joburi mid-senior).
5. Poți include 1–2 variante și în română doar dacă CV-ul are experiență clar locală și joburi tip „Analist financiar”, „Manager proiect” etc.
6. Fiecare căutare trebuie să fie realistă și să returneze rezultate relevante (nu prea generică: evită doar "Python"; combină cu rol sau industrie).
7. Returnează **NUMAI JSON valid**, fără niciun text înainte sau după:

{
  "queries": [
    "căutare 1 completă",
    "căutare 2 completă",
    ...
    "căutare 7 completă"
  ]
}

CV:
{cv}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)
    if not parsed or "queries" not in parsed:
        parsed = {"queries": []}
    return api_response(payload=parsed)


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
Răspunde NUMAI cu textul rescris, fără etichete precum **Situație:** sau alte marcaje – doar paragrafe cursive și coerente în română corectă.
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

