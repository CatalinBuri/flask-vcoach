# ===============================================
# app.py — VERSIUNE INTEGRALĂ ȘI CORECTATĂ
# ===============================================
import os
import json
import hashlib
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
import orjson
from flask_compress import Compress

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
Compress(app)
CORS(app, resources={r"/*": {"origins": "*"}})

# Identificator model - Folosim 2.0-flash pentru viteză și stabilitate
MODEL_NAME = "gemini-2.0-flash"

gemini_client = None
try:
    if API_KEY:
        gemini_client = genai.Client(api_key=API_KEY)
        print(f"✅ Gemini {MODEL_NAME} gata.")
except Exception as e:
    print(f"❌ Eroare Gemini init: {e}")

# --- UTILS ---
def jsonify_fast(data, code=200):
    return app.response_class(orjson.dumps(data), status=code, mimetype='application/json')

def api_response(payload=None, error=None, code=200):
    base = {"status": "ok" if not error else "error", "payload": payload, "error": str(error) if error else None}
    return jsonify_fast(base, code)

def validate_fields(data, required_fields):
    missing = [f for f in required_fields if not data.get(f)]
    if missing: raise ValueError(f"Lipsesc: {', '.join(missing)}")

def safe_json_extract(text):
    if not text: raise ValueError("AI return empty")
    t = text.strip().replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(t)
    except:
        s, e = t.find('{'), t.rfind('}') + 1
        return json.loads(t[s:e])

def call_gemini_raw(prompt):
    try:
        response = gemini_client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return response.text
    except Exception as e:
        return {"error": str(e)}

def call_gemini_json(prompt):
    raw = call_gemini_raw(prompt)
    if isinstance(raw, dict) and "error" in raw: return raw
    try: return safe_json_extract(raw)
    except Exception as e: return {"error": "JSON Parse Error", "details": str(e)}

# --- RUTE ---

@app.route('/process-text', methods=['POST'])
def process_text():
    try:
        data = request.get_json(force=True)
        text_input = data.get('text', '').strip()
        if not text_input: return api_response(error="Text lipsă", code=400)
        
        prompt = f"Fă un rezumat fluid de max 4 paragrafe pentru: {text_input}. Fără bullet points, fără bold."
        res = call_gemini_raw(prompt)
        return api_response(payload={"t": res})
    except Exception as e: return api_response(error=str(e), code=500)

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['cv_text', 'job_summary'])
        prompt = f"Generează 5 întrebări de interviu bazate pe CV: {data['cv_text']} și Job: {data['job_summary']}. Returnează JSON: {{'questions': []}}"
        res = call_gemini_json(prompt)
        return api_response(payload=res)
    except Exception as e: return api_response(error=str(e), code=400)

@app.route('/analyze-cv', methods=['POST'])
def analyze_cv():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['cv_text', 'job_text'])
        prompt = f"Analizează compatibilitatea CV: {data['cv_text']} cu Job: {data['job_text']}. Returnează JSON: {{'compatibility_percent': 85, 'feedback_markdown': '...'}}"
        res = call_gemini_json(prompt)
        return api_response(payload=res)
    except Exception as e: return api_response(error=str(e), code=400)

@app.route('/generate-job-queries', methods=['POST'])
def generate_job_queries():
    try:
        data = request.get_json(force=True)
        cv = data.get('cv_text', '')
        prompt = f"Generează 7 căutări LinkedIn scurte pentru acest CV: {cv}. Returnează JSON: {{'queries': []}}"
        res = call_gemini_json(prompt)
        return api_response(payload=res)
    except Exception as e: return api_response(error=str(e), code=400)

@app.route('/optimize-linkedin-profile', methods=['POST'])
def optimize_linkedin():
    try:
        data = request.get_json(force=True)
        prompt = f"Optimizează profilul LinkedIn pentru CV: {data.get('cv_text')}. Returnează JSON: {{'linkedin_headlines': [], 'linkedin_about': ''}}"
        res = call_gemini_json(prompt)
        return api_response(payload=res)
    except Exception as e: return api_response(error=str(e), code=400)

@app.route('/evaluate-answer', methods=['POST'])
def evaluate_answer():
    try:
        data = request.get_json(force=True)
        prompt = f"Evaluează răspunsul: {data['answer']} la întrebarea: {data['question']}. Returnează JSON cu nota_finala și feedback."
        res = call_gemini_json(prompt)
        return api_response(payload=res)
    except Exception as e: return api_response(error=str(e), code=400)

@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json(force=True)
        prompt = f"Generează raport final pentru istoricul: {data.get('history')}. Returnează JSON cu summary și scor."
        res = call_gemini_json(prompt)
        return api_response(payload=res)
    except Exception as e: return api_response(error=str(e), code=500)

@app.route('/coach-next', methods=['POST'])
def coach_next():
    try:
        data = request.get_json(force=True)
        user_answer = data.get('user_answer', '')
        if len(user_answer.split()) < 5:
            return api_response(payload={"star_answer": "Răspuns prea scurt."})
        prompt = f"Transformă în format STAR: {user_answer}"
        res = call_gemini_raw(prompt)
        return api_response(payload={"star_answer": res})
    except Exception as e: return api_response(error=str(e), code=400)

@app.route('/ping', methods=['GET'])
def ping(): return jsonify({"status": "awake"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
