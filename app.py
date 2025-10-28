import os
import json
import re
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

# -----------------------
# UTILITAR: extracție sigură JSON
# -----------------------
def safe_json_extract(text):
    if not text:
        raise ValueError("Răspunsul AI este gol.")
    text = text.strip()
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    json_str = match.group(1).strip() if match else text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = json_str.replace('\\"', '"').replace('\n', '').replace('\t', '')
        return json.loads(json_str)

# -----------------------
# CONFIG FLASK + GEMINI
# -----------------------
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Lipsește GEMINI_API_KEY din .env!")

gemini_client = genai.Client(api_key=API_KEY)

# -----------------------
# DECORATOR PENTRU PRE-FLIGHT OPTIONS
# -----------------------
def allow_options(func):
    def wrapper(*args, **kwargs):
        if request.method == "OPTIONS":
            return '', 200
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# -----------------------
# ENDPOINTURI CU CORS + OPTIONS
# -----------------------
@app.route("/generate-questions", methods=["POST", "OPTIONS"])
@allow_options
def generate_questions():
    try:
        data = request.get_json()
        cv_text = data.get("cv_text", "").strip()
        job_text = data.get("job_text", "").strip()
        if not cv_text or not job_text:
            return jsonify({"error": "CV-ul și Job Description sunt obligatorii."}), 400

        prompt = f"""Ești un recrutor AI. CV: {cv_text}, JOB: {job_text}. Format JSON cu summary și 5 întrebări."""
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare întrebări eșuată", "details": str(e)}), 500

@app.route("/analyze-answer", methods=["POST", "OPTIONS"])
@allow_options
def analyze_answer():
    try:
        data = request.get_json()
        question = data.get("question")
        user_answer = data.get("user_answer")
        history = data.get("history", [])
        if not question or not user_answer:
            return jsonify({"error": "Întrebarea și răspunsul sunt obligatorii."}), 400

        prompt = f"Întrebare: {question}, Răspuns: {user_answer}, Istoric: {json.dumps(history)}. Răspuns JSON cu current_evaluation."
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        if not ai_data.get("current_evaluation"):
            raise ValueError("Structură JSON lipsă current_evaluation.")
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Analiza răspunsului eșuată", "details": str(e)}), 500

@app.route("/generate-report", methods=["POST", "OPTIONS"])
@allow_options
def generate_report():
    try:
        data = request.get_json()
        job_summary = data.get("job_summary")
        history = data.get("history")
        if not job_summary or not history:
            return jsonify({"error": "Sinteza jobului și istoricul sunt obligatorii."}), 400

        prompt = f"Sinteză job: {job_summary}, Istoric: {json.dumps(history)}. Răspuns JSON cu overall_report_markdown."
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare raport eșuat", "details": str(e)}), 500

@app.route("/process-text", methods=["POST", "OPTIONS"])
@allow_options
def process_text():
    try:
        data = request.get_json()
        job_text = data.get("text", "").strip()
        if not job_text:
            return jsonify({"error": "Job description is required."}), 400

        clean_text = re.sub(r"(?i)\b(bullet\s*icon)\b", "", job_text)
        clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()
        prompt = f"Rezumat responsabilități și competențe: {clean_text}. Răspuns JSON."
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        summary = response.text.strip()
        return jsonify({"processed_text": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Eroare procesare text", "details": str(e)}), 500

# --- Exemple pentru celelalte endpointuri: ---
@app.route("/analyze-cv", methods=["POST", "OPTIONS"])
@allow_options
def analyze_cv():
    try:
        data = request.get_json()
        cv_text = data.get("cv_text", "")
        if not cv_text:
            return jsonify({"error": "cv_text obligatoriu"}), 400
        prompt = f"Analizează CV-ul: {cv_text}. Răspuns strict JSON."
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Analiză CV eșuată", "details": str(e)}), 500

# Poți adăuga similar /generate-cover-letter, /generate-linkedin-summary, /generate-job-hunt-optimization etc.

# -----------------------
# PORNIRE SERVER
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
