import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

# ------------------------------------------------------------
#  ÎNCĂRCARE VARIABILE DE MEDIU ȘI INIȚIALIZARE
# ------------------------------------------------------------
load_dotenv()
app = Flask(__name__)

# ------------------------------------------------------------
#  CONFIGURARE CORS SIGUR PENTRU PROD + DEV
# ------------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://www.pixelplayground3d.ro",
    "https://pixelplayground3d.ro",
    "https://cvcoach-ai.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

CORS(
    app,
    origins=ALLOWED_ORIGINS,
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)

@app.after_request
def after_request(response):
    """Asigură headere CORS corecte pentru toate răspunsurile."""
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    """Răspuns global la cererile OPTIONS (preflight)."""
    response = jsonify({"status": "preflight ok"})
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# ------------------------------------------------------------
#  INITIALIZARE CLIENT GEMINI
# ------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
try:
    if not API_KEY:
        print("❌ EROARE: Lipsă GEMINI_API_KEY în fișierul .env")
        gemini_client = None
    else:
        gemini_client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ Eroare la inițializarea clientului Gemini: {e}")
    gemini_client = None

# ------------------------------------------------------------
#  FUNCȚIE UTILITARĂ: SAFE JSON PARSING
# ------------------------------------------------------------
def safe_json_extract(text):
    """Parsează în siguranță un răspuns JSON, chiar dacă are delimitatori Markdown."""
    if not text:
        raise ValueError("Răspuns AI gol.")
    txt = text.strip()

    if txt.startswith("```json"):
        txt = txt.replace("```json", "").strip()
    if txt.endswith("```"):
        txt = txt[:-3].strip()

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        try:
            start = txt.index("{")
            end = txt.rindex("}") + 1
            return json.loads(txt[start:end])
        except Exception as e:
            raise ValueError(f"Nu s-a putut extrage JSON-ul din text: {e}\n{text[:300]}...")

# ------------------------------------------------------------
#  RUTĂ DE VERIFICARE
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "OK", "message": "Flask VCoach API online ✅"}), 200

# ------------------------------------------------------------
#  RUTA 1: ANALIZA CV + JOB DESCRIPTION
# ------------------------------------------------------------
@app.route("/analyze-cv", methods=["POST", "OPTIONS"])
def analyze_cv():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight ok"}), 200

    if not gemini_client:
        return jsonify({"error": "Gemini client neinițializat"}), 500

    data = request.get_json()
    cv_text = data.get("cv_text", "").strip()
    job_text = data.get("job_text", "").strip()

    if not cv_text or not job_text:
        return jsonify({"error": "Lipsă text CV sau descriere job"}), 400

    prompt = f"""
    Ești un analist HR AI. Compară textul CV-ului cu descrierea jobului și oferă o evaluare completă.

    CV:
    {cv_text}

    JOB DESCRIPTION:
    {job_text}

    Returnează în format JSON STRICT:
    {{
      "compatibility_percent": 0-100,
      "feedback_markdown": "Feedback detaliat, în format Markdown, cu puncte tari și recomandări."
    }}
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        print(f"❌ Eroare la analiza CV: {e}")
        return jsonify({"error": "Analiza CV a eșuat.", "details": str(e)}), 500

# ------------------------------------------------------------
#  RUTA 2: GENERARE JOB QUERIES
# ------------------------------------------------------------
@app.route("/generate-job-queries", methods=["POST"])
def generate_job_queries():
    if not gemini_client:
        return jsonify({"error": "Gemini API indisponibil"}), 503

    data = request.get_json()
    cv_text = data.get("cv_text", "").strip()

    if not cv_text:
        return jsonify({"error": "Lipsă text CV în cerere"}), 400

    prompt = f"""
    Ești un asistent de carieră AI. Extrage din CV profesia și competențele principale și generează 5 interogări de căutare relevante pentru joburi.

    Format JSON STRICT:
    {{
      "queries": ["interogare 1", "interogare 2", "interogare 3", "interogare 4", "interogare 5"]
    }}

    CV:
    {cv_text}
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        print(f"❌ Eroare la generarea interogărilor job: {e}")
        return jsonify({"error": "Eroare la generarea interogărilor", "details": str(e)}), 500

# ------------------------------------------------------------
#  RUTA 3: GENERARE FAQ (ÎNTREBĂRI)
# ------------------------------------------------------------
@app.route('/generate-beginner-faq', methods=['POST'])
def generate_beginner_faq():
    if not gemini_client:
        return jsonify({"error": "Serviciul AI nu este disponibil"}), 503

    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()

    if not cv_text or cv_text == 'GENERIC_FAQ_MODE':
        context_prompt = "Generează 5 întrebări standard pentru candidați entry-level."
    else:
        context_prompt = f"Generează 5 întrebări bazate pe CV-ul următor:\n{cv_text}"

    prompt = f"""
    Ești un recrutor AI. Creează 5 întrebări frecvente (FAQ) și explicația lor.

    Format JSON STRICT:
    {{
      "faq": [
        {{
          "question": "Întrebarea 1?",
          "explanation": "Scurtă explicație a intenției recrutorului."
        }}
      ]
    }}

    {context_prompt}
    """

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        print(f"❌ Eroare FAQ: {e}")
        return jsonify({"error": "Eroare la generarea FAQ", "details": str(e)}), 500

# ------------------------------------------------------------
#  RUTA 4: ANALIZA RĂSPUNSURILOR FAQ
# ------------------------------------------------------------
@app.route('/analyze-faq-answers', methods=['POST'])
def analyze_faq_answers():
    if not gemini_client:
        return jsonify({"error": "Clientul Gemini nu este inițializat"}), 500

    data = request.get_json()
    faq_data = data.get('faq_data')

    if not faq_data or not isinstance(faq_data, list):
        return jsonify({"error": "Format invalid pentru 'faq_data'."}), 400

    item = faq_data[0]
    prompt = f"""
    Ești un antrenor de interviu. Analizează răspunsul utilizatorului la întrebarea următoare.

    ÎNTREBAREA: {item.get('question', 'N/A')}
    EXPLICAȚIA INTENȚIEI: {item.get('explanation', 'N/A')}
    RĂSPUNS UTILIZATOR: {item.get('user_answer', 'N/A')}

    Returnează în format JSON STRICT:
    {{
      "analysis_results": [
        {{
          "question": "{item.get('question', 'N/A')}",
          "user_answer": "{item.get('user_answer', 'N/A')}",
          "evaluation": {{
            "nota_finala": 0-10,
            "claritate": 0-10,
            "relevanta": 0-10,
            "structura": 0-10,
            "feedback": "Feedback constructiv în Markdown"
          }}
        }}
      ]
    }}
    """

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        print(f"❌ Eroare analiză FAQ: {e}")
        return jsonify({"error": "Analiza răspunsului FAQ a eșuat", "details": str(e)}), 500

# ------------------------------------------------------------
#  RUTA 5: GENERARE RAPORT FINAL
# ------------------------------------------------------------
@app.route('/generate-final-report', methods=['POST'])
def generate_final_report():
    if not gemini_client:
        return jsonify({"error": "Gemini API neconfigurat"}), 500

    data = request.json
    faq_history = data.get('faq_history', [])

    if not faq_history:
        return jsonify({"error": "Istoricul FAQ este gol."}), 400

    history_text = ""
    for idx, entry in enumerate(faq_history):
        q = entry.get('question_data', {}).get('question', 'N/A')
        a = entry.get('user_answer', 'N/A')
        note = entry.get('analysis', {}).get('evaluation', {}).get('nota_finala', 'N/A')
        feedback = entry.get('analysis', {}).get('evaluation', {}).get('feedback', 'N/A')
        history_text += f"--- Întrebarea {idx+1} (Nota: {note}/10) ---\nÎntrebare: {q}\nRăspuns: {a}\nFeedback: {feedback}\n\n"

    prompt = f"""
    Ești un coach AI. Generează un raport final sumarizând performanța candidatului.

    FORMAT JSON STRICT:
    {{
      "final_score": "Nota medie (ex: 8.2)",
      "summary": "Sinteză generală în Markdown",
      "key_strengths": ["Punct forte 1", "Punct forte 2", "Punct forte 3"],
      "areas_for_improvement": ["Arie 1", "Arie 2", "Arie 3"],
      "next_steps_recommendation": "Recomandări practice (Markdown)"
    }}

    ---
    ISTORIC:
    {history_text}
    """

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        print(f"❌ Eroare raport final: {e}")
        return jsonify({"error": "Generarea raportului final a eșuat", "details": str(e)}), 500

# ------------------------------------------------------------
#  START SERVER
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Flask VCoach server running on port 5000...")
    app.run(host="0.0.0.0", port=5000)
