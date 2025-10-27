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

# ✅ CONFIGURARE CORS SIGURĂ
CORS(
    app,
    origins=[
        "https://www.pixelplayground3d.ro",
        "https://pixelplayground3d.ro",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True
)

@app.after_request
def after_request(response):
    """Asigură că toate răspunsurile (inclusiv erorile) conțin headere CORS"""
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response

# ------------------------------------------------------------
#  CONFIGURARE CLIENT GEMINI
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
#  UTILS
# ------------------------------------------------------------
def safe_json_extract(text):
    """Curăță textul AI și extrage un JSON valid"""
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
#  RUTĂ DE TEST
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "OK", "message": "Flask VCoach API online ✅"}), 200


# ------------------------------------------------------------
#  RUTĂ: /analyze-cv
# ------------------------------------------------------------
@app.route("/analyze-cv", methods=["POST", "OPTIONS"])
def analyze_cv():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    if not gemini_client:
        return jsonify({"error": "Gemini client neinițializat"}), 500

    data = request.get_json()
    cv_text = data.get("cv_text", "").strip()

    if not cv_text:
        return jsonify({"error": "Lipsă text CV în cerere"}), 400

    prompt = f"""
    Ești un analist HR AI. Analizează CV-ul de mai jos și oferă o scurtă evaluare.

    CV:
    {cv_text}

    Format JSON STRICT:
    {{
      "summary": "Scurt rezumat al profilului candidatului.",
      "skills_detected": ["listă de competențe identificate"],
      "experience_level": "junior / mid / senior",
      "recommendations": "Sugestii practice pentru îmbunătățirea CV-ului."
    }}
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        print(f"❌ Eroare Gemini / analiză CV: {e}")
        return jsonify({"error": "Analiza CV a eșuat.", "details": str(e)}), 500


# ------------------------------------------------------------
#  RUTĂ: /generate-beginner-faq
# ------------------------------------------------------------
@app.route("/generate-beginner-faq", methods=["POST"])
def generate_beginner_faq():
    if not gemini_client:
        return jsonify({"error": "Gemini API indisponibil"}), 503

    data = request.get_json()
    cv_text = data.get("cv_text", "").strip()

    if not cv_text or cv_text == "GENERIC_FAQ_MODE":
        context_prompt = "Generează 5 întrebări standard potrivite pentru candidații entry-level."
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
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        print(f"❌ Eroare FAQ: {e}")
        return jsonify({"error": "Eroare la generarea FAQ", "details": str(e)}), 500


# ------------------------------------------------------------
#  RUTĂ: /analyze-faq-answers
# ------------------------------------------------------------
@app.route("/analyze-faq-answers", methods=["POST"])
def analyze_faq_answers():
    if not gemini_client:
        return jsonify({"error": "Gemini API indisponibil"}), 500

    data = request.get_json()
    faq_data = data.get("faq_data")

    if not faq_data or not isinstance(faq_data, list):
        return jsonify({"error": "Format invalid pentru 'faq_data'."}), 400

    item = faq_data[0]
    prompt = f"""
    Ești un antrenor de interviu. Analizează răspunsul utilizatorului.

    ÎNTREBARE: {item.get('question', 'N/A')}
    INTENȚIE: {item.get('explanation', 'N/A')}
    RĂSPUNS UTILIZATOR: {item.get('user_answer', 'N/A')}

    Format JSON STRICT:
    {{
      "analysis_results": [
        {{
          "question": "{item.get('question', 'N/A')}",
          "user_answer": "{item.get('user_answer', 'N/A')}",
          "evaluation": {{
            "nota_finala": 8,
            "claritate": 9,
            "relevanta": 7,
            "structura": 8,
            "feedback": "Feedback detaliat și constructiv (Markdown)."
          }}
        }}
      ]
    }}
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        print(f"❌ Eroare analiză FAQ: {e}")
        return jsonify({"error": "Eroare analiză răspunsuri", "details": str(e)}), 500


# ------------------------------------------------------------
#  RUTĂ: /generate-final-report
# ------------------------------------------------------------
@app.route("/generate-final-report", methods=["POST"])
def generate_final_report():
    if not gemini_client:
        return jsonify({"error": "Gemini API indisponibil"}), 500

    data = request.get_json()
    faq_history = data.get("faq_history", [])

    if not faq_history:
        return jsonify({"error": "Istoricul FAQ este gol."}), 400

    history_text = ""
    for idx, entry in enumerate(faq_history):
        q = entry.get("question_data", {}).get("question", "N/A")
        a = entry.get("user_answer", "N/A")
        nota = entry.get("analysis", {}).get("evaluation", {}).get("nota_finala", "N/A")
        feedback = entry.get("analysis", {}).get("evaluation", {}).get("feedback", "N/A")

        history_text += (
            f"--- Întrebarea {idx+1} (Nota: {nota}/10) ---\n"
            f"Întrebare: {q}\n"
            f"Răspuns Utilizator: {a}\n"
            f"Feedback Coach: {feedback}\n\n"
        )

    prompt = f"""
    Ești un coach de carieră. Generează o sinteză bazată pe istoricul următor:

    {history_text}

    Format JSON STRICT:
    {{
      "final_score": "media notelor, rotunjită la o zecimală",
      "summary": "Sinteză generală (Markdown)",
      "key_strengths": ["cel puțin 3 puncte forte"],
      "areas_for_improvement": ["cel puțin 3 arii de îmbunătățire"],
      "next_steps_recommendation": "Recomandări practice (Markdown)"
    }}
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        print(f"❌ Eroare raport final: {e}")
        return jsonify({"error": "Eroare la generarea raportului", "details": str(e)}), 500


# ------------------------------------------------------------
#  START SERVER
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Flask VCoach server running on port 5000...")
    app.run(host="0.0.0.0", port=5000)
