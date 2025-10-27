import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==============================================================
# 🔐 CONFIGURARE ȘI INIȚIALIZARE
# ==============================================================

load_dotenv()  # Încarcă GEMINI_API_KEY din .env

app = Flask(__name__)

# ✅ CORS securizat: permite doar domeniile de încredere
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://cvcoach-ai.vercel.app"
]}})

@app.after_request
def after_request(response):
    """Asigură CORS și pentru răspunsurile de eroare."""
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response

# --------------------------------------------------------------
# Inițializare client Gemini
# --------------------------------------------------------------

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("\n[EROARE CRITICĂ] GEMINI_API_KEY lipsește din .env.")
    gemini_client = None
else:
    try:
        gemini_client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(f"Eroare la inițializarea Gemini Client: {e}")
        gemini_client = None


# ==============================================================
# 🧩 FUNCȚII UTILE
# ==============================================================

def safe_json_extract(text: str):
    """
    Extrage obiect JSON curat dintr-un răspuns AI.
    Curăță blocuri Markdown și text suplimentar.
    """
    if not text:
        raise ValueError("Text gol primit de la AI.")
    
    full_text = text.strip()
    if full_text.startswith("```json"):
        full_text = full_text.replace("```json", "", 1).strip()
    if full_text.endswith("```"):
        full_text = full_text[:-3].strip()

    try:
        return json.loads(full_text)
    except json.JSONDecodeError:
        try:
            start = full_text.index("{")
            end = full_text.rindex("}") + 1
            json_str = full_text[start:end]
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Eroare la extragerea JSON: {e}\nText: {full_text[:500]}")


# ==============================================================
# 🤖 RUTĂ: Analiză CV și compatibilitate cu job description
# ==============================================================

@app.route("/analyze-cv", methods=["POST"])
def analyze_cv():
    if not gemini_client:
        return jsonify({"error": "Serviciul AI indisponibil. Verifică cheia API."}), 503

    data = request.get_json()
    cv_text = data.get("cv_text", "").strip()
    job_text = data.get("job_description", "").strip()

    if not cv_text or not job_text:
        return jsonify({"error": "Lipsesc textul CV sau descrierea jobului."}), 400

    prompt = f"""
    Ești un expert HR. Analizează compatibilitatea dintre următoarele texte:

    CV:
    {cv_text}

    JOB DESCRIPTION:
    {job_text}

    Returnează un JSON STRICT:
    {{
      "match_score": 0-100,
      "summary": "Analiză generală (Markdown)",
      "strengths": ["Punct forte 1", "Punct forte 2"],
      "gaps": ["Lacună 1", "Lacună 2"],
      "recommendations": "Recomandări concrete (Markdown)"
    }}
    """

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(resp.text)), 200
    except Exception as e:
        print("Eroare analiză CV:", e)
        return jsonify({"error": "Eroare AI la analiza CV.", "details": str(e)}), 500


# ==============================================================
# 💼 RUTĂ: Generare interogări de joburi relevante
# ==============================================================

@app.route("/generate-job-queries", methods=["POST"])
def generate_job_queries():
    if not gemini_client:
        return jsonify({"error": "Clientul Gemini indisponibil."}), 503

    data = request.get_json()
    cv_text = data.get("cv_text", "").strip()

    if not cv_text:
        return jsonify({"error": "Textul CV lipsește."}), 400

    prompt = f"""
    Ești un expert în HR și căutări de joburi. Pe baza următorului CV,
    generează 5 interogări relevante pentru motoare de căutare joburi.

    CV:
    {cv_text}

    Returnează JSON STRICT:
    {{
      "job_queries": ["interogare1", "interogare2", "interogare3", "interogare4", "interogare5"]
    }}
    """

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(resp.text)), 200
    except Exception as e:
        return jsonify({"error": "Eroare AI la generarea interogărilor.", "details": str(e)}), 500


# ==============================================================
# 🧠 RUTĂ: Generare întrebări FAQ pentru candidați
# ==============================================================

@app.route("/generate-beginner-faq", methods=["POST"])
def generate_beginner_faq():
    if gemini_client is None:
        return jsonify({"error": "Serviciul AI nu este disponibil."}), 503
        
    data = request.get_json()
    cv_text = data.get("cv_text", "").strip()

    context_prompt = (
        "Generează 5 întrebări standard pentru candidați începători."
        if not cv_text or cv_text == "GENERIC_FAQ_MODE"
        else f"Generează 5 întrebări bazate pe următorul CV: {cv_text}"
    )

    prompt = f"""
    Ești un Recrutor AI. Creează 5 întrebări FAQ și explicațiile lor.

    Format JSON STRICT:
    {{
      "faq": [
        {{"question": "...", "explanation": "..."}}
      ]
    }}

    {context_prompt}
    """

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(resp.text)), 200
    except Exception as e:
        return jsonify({"error": "Eroare generare FAQ.", "details": str(e)}), 500


# ==============================================================
# 🧩 RUTĂ: Analiză răspunsuri FAQ (scor + feedback)
# ==============================================================

@app.route("/analyze-faq-answers", methods=["POST"])
def analyze_faq_answers():
    if not gemini_client:
        return jsonify({"error": "Clientul Gemini nu este inițializat."}), 500

    data = request.get_json()
    faq_data = data.get("faq_data")
    if not faq_data or not isinstance(faq_data, list):
        return jsonify({"error": "Date FAQ invalide."}), 400

    item = faq_data[0]
    analysis_context = f"""
    ÎNTREBAREA: {item.get('question')}
    EXPLICAȚIA: {item.get('explanation')}
    RĂSPUNS: {item.get('user_answer')}
    """

    prompt = f"""
    Analizează răspunsul utilizatorului la o întrebare de interviu.

    Returnează JSON STRICT:
    {{
      "analysis_results": [
        {{
          "question": "{item.get('question')}",
          "user_answer": "{item.get('user_answer')}",
          "evaluation": {{
            "nota_finala": 1-10,
            "claritate": 1-10,
            "relevanta": 1-10,
            "structura": 1-10,
            "feedback": "Text Markdown constructiv"
          }}
        }}
      ]
    }}

    Context:
    {analysis_context}
    """

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(resp.text)), 200
    except Exception as e:
        return jsonify({"error": "Eroare analiză răspuns.", "details": str(e)}), 500


# ==============================================================
# 📊 RUTĂ: Generare raport final (sinteză sesiune)
# ==============================================================

@app.route("/generate-final-report", methods=["POST"])
def generate_final_report():
    if not gemini_client:
        return jsonify({"error": "Gemini API nu este configurat."}), 500
    
    data = request.json
    faq_history = data.get("faq_history", [])

    if not faq_history:
        return jsonify({"error": "Istoricul FAQ este gol."}), 400

    history_text = ""
    for idx, entry in enumerate(faq_history):
        q = entry.get("question_data", {}).get("question", "N/A")
        a = entry.get("user_answer", "N/A")
        note = entry.get("analysis", {}).get("evaluation", {}).get("nota_finala", "N/A")
        feedback = entry.get("analysis", {}).get("evaluation", {}).get("feedback", "N/A")
        history_text += (
            f"--- Întrebarea {idx+1} (Nota: {note}/10) ---\n"
            f"Întrebare: {q}\nRăspuns: {a}\nFeedback: {feedback}\n\n"
        )

    prompt = f"""
    Ești un Expert Coach de Carieră. Generează un raport sintetic bazat pe următorul istoric:

    {history_text}

    Returnează JSON STRICT:
    {{
      "final_score": "media notelor",
      "summary": "Sinteză generală (Markdown)",
      "key_strengths": ["..."],
      "areas_for_improvement": ["..."],
      "next_steps_recommendation": "Recomandări (Markdown)"
    }}
    """

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify(safe_json_extract(resp.text)), 200
    except Exception as e:
        return jsonify({"error": "Eroare generare raport final.", "details": str(e)}), 500


# ==============================================================
# 🚀 PORNIRE SERVER
# ==============================================================

if __name__ == "__main__":
    print("\n=======================================================")
    print("🌐 Flask AI Interview Coach pornit cu app.run()")
    print("🔗 Acces: http://127.0.0.1:5000/")
    print("=======================================================\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
