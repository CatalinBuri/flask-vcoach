import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ==========================
# Încarcă variabilele de mediu
# ==========================
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    if not API_KEY:
        print("EROARE: GEMINI_API_KEY lipseste!")
        gemini_client = None
    else:
        gemini_client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Eroare inițializare Gemini: {e}")
    gemini_client = None

# ==========================
# Inițializare Flask + CORS
# ==========================
app = Flask(__name__)
CORS(app)

# ==========================
# Funcții auxiliare
# ==========================
def safe_json_extract(text):
    """Extrage JSON din răspuns AI, chiar dacă e în Markdown sau fragmentat"""
    if not text:
        raise ValueError("Text gol primit de la AI.")
    full_text = text.strip()
    try:
        if full_text.startswith('```json'):
            full_text = full_text.replace('```json', '', 1)
        if full_text.endswith('```'):
            full_text = full_text[:-3]
        start_index = full_text.index('{')
        end_index = full_text.rindex('}') + 1
        json_string = full_text[start_index:end_index]
        return json.loads(json_string)
    except Exception as e:
        raise ValueError(f"Eroare la parsare JSON: {e}. Text: {full_text}")

# ==========================
# Scheme JSON
# ==========================
feedback_schema = {
    "type": "object",
    "properties": {
        "compatibility_percent": {"type": "integer"},
        "feedback_markdown": {"type": "string"}
    },
    "required": ["compatibility_percent", "feedback_markdown"]
}

evaluation_schema = {
    "type": "object",
    "properties": {
        "nota_finala": {"type": "integer"},
        "claritate": {"type": "integer"},
        "relevanta": {"type": "integer"},
        "structura": {"type": "integer"},
        "feedback": {"type": "string"}
    },
    "required": ["nota_finala", "claritate", "relevanta", "structura", "feedback"]
}

comparative_feedback_schema = {
    "type": "object",
    "properties": {
        "feedback": {"type": "string"}
    },
    "required": ["feedback"]
}

coach_schema = {
    "type": "object",
    "properties": {
        "model_answer": {"type": "string"},
        "methodology_used": {"type": "string"},
        "methodology_explanation": {"type": "string"}
    },
    "required": ["model_answer", "methodology_used", "methodology_explanation"]
}

# ==========================
# Endpoint-uri CV & Job Hunt
# ==========================

@app.route('/analyze-cv', methods=['POST'])
def analyze_cv():
    if not gemini_client:
        return jsonify({"error": "AI indisponibil"}), 503
    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()
    job_text = data.get('job_text', '').strip()
    if not cv_text or not job_text:
        return jsonify({"error": "cv_text și job_text obligatorii"}), 400

    prompt = f"""
    Ești expert în recrutare. Analizează CV-ul vs job.
    Returnează JSON conform schemei feedback_schema.
    DESCRIERE JOB: {job_text}
    CV: {cv_text}
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=feedback_schema
            )
        )
        return jsonify(json.loads(response.text)), 200
    except Exception as e:
        return jsonify({"error": f"Eroare analiză CV: {e}"}), 500

@app.route('/generate-job-queries', methods=['POST'])
def generate_job_queries():
    if not gemini_client:
        return jsonify({"error": "AI indisponibil"}), 503
    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()
    if not cv_text:
        return jsonify({"error": "cv_text obligatoriu"}), 400

    prompt = f"Generează 5 interogări job optimizate pentru Google, LinkedIn etc. pe baza CV-ului:\n{cv_text}\nReturnează STRICT JSON cu cheie 'queries'."
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        return jsonify({"error": f"Eroare generare queries: {e}"}), 500

@app.route('/generate-cover-letter', methods=['POST'])
def generate_cover_letter():
    if not gemini_client:
        return jsonify({"error": "AI indisponibil"}), 503
    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()
    job_text = data.get('job_text', '').strip()
    if not cv_text or not job_text:
        return jsonify({"error": "cv_text și job_text obligatorii"}), 400

    prompt = f"""
    Generează o scrisoare de intenție profesională pentru începător.
    Evidențiază abilități transferabile și pasiune.
    DESCRIERE JOB: {job_text}
    CV: {cv_text}
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return jsonify({"cover_letter": response.text}), 200
    except Exception as e:
        return jsonify({"error": f"Eroare cover letter: {e}"}), 500

@app.route('/optimize-linkedin-profile', methods=['POST'])
def optimize_linkedin_profile():
    if not gemini_client:
        return jsonify({"error": "AI indisponibil"}), 503
    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()
    domain = data.get('domain', '').strip()
    if not cv_text or not domain:
        return jsonify({"error": "cv_text și domain obligatorii"}), 400

    prompt = f"""
    Optimizează profil LinkedIn pentru începător. Returnează JSON cu 'headlines' și 'about_section'.
    CV: {cv_text}
    Domeniu: {domain}
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        return jsonify({"error": f"Eroare LinkedIn: {e}"}), 500

@app.route('/generate-beginner-faq', methods=['POST'])
def generate_beginner_faq():
    if not gemini_client:
        return jsonify({"error": "AI indisponibil"}), 503
    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()
    if not cv_text:
        return jsonify({"error": "cv_text obligatoriu"}), 400

    prompt = f"""
    Generează 5 întrebări frecvente pentru începători + răspuns model.
    Returnează STRICT JSON cu 'faq'.
    CV: {cv_text}
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
    except Exception as e:
        return jsonify({"error": f"Eroare FAQ: {e}"}), 500

# ==========================
# Endpoint-uri interviu & evaluare (vechi)
# ==========================
@app.route('/process-text', methods=['POST'])
def process_text():
    if not gemini_client:
        return jsonify({"error": "AI indisponibil"}), 503
    data = request.get_json()
    job_text = data.get('text', '').strip()
    if not job_text:
        return jsonify({"error": "text obligatoriu"}), 400

    prompt = f"Analizează descriere job: {job_text} și oferă rezumat scurt pentru interviu."
    try:
        response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return jsonify({"processed_text": response.text})
    except Exception as e:
        return jsonify({"error": f"Eroare procesare text: {e}"}), 500

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    if not gemini_client:
        return jsonify({"error": "AI indisponibil"}), 503
    data = request.get_json()
    processed_text = data.get('processed_text', '')
    prompt = f"Generează 5 întrebări interviu pe baza: {processed_text}. Returnează STRICT JSON cu 'questions'."
    try:
        response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return jsonify(safe_json_extract(response.text))
    except Exception as e:
        return jsonify({"error": f"Eroare generare întrebări: {e}"}), 500

# ==========================
# Rulare aplicație
# ==========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
