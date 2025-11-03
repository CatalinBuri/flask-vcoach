# ===============================================
# server_vcoach_robust.py — Versiune optimizată JSON
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

# --------------------------
# Încarcă variabilele de mediu (.env)
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

# --------------------------
# Inițializare Flask + Compresie HTTP
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
Compress(app)
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------
# Logging minimal pentru debugging
@app.before_request
def log_request():
    print(f"[{request.method}] {request.path} | from: {request.headers.get('Origin', 'local')}")

# --------------------------
# Inițializare client Gemini
gemini_client = None
try:
    if not API_KEY:
        print("❌ EROARE: GEMINI_API_KEY lipsește!")
    else:
        gemini_client = genai.Client(api_key=API_KEY)
        print("✅ Conexiune Gemini inițializată corect.")
except Exception as e:
    print(f"❌ Eroare la inițializarea Gemini: {e}")

# ===========================================================
# 🔧 FUNCȚII UTILE GENERALE (JSON, VALIDARE, RĂSPUNSURI)
# ===========================================================

def jsonify_fast(data, code=200):
    """Serializare rapidă + minificată cu orjson."""
    return app.response_class(
        orjson.dumps(data),
        status=code,
        mimetype='application/json'
    )

def api_response(payload=None, error=None, code=200, meta=None):
    """Formatează răspunsurile JSON într-o structură uniformă."""
    base = {
        "status": "ok" if not error else "error",
        "payload": payload if not error else None,
        "error": str(error) if error else None,
        "meta": meta or {}
    }
    # Calcul checksum pentru integritate
    checksum_data = json.dumps(payload or {}, separators=(',', ':'), sort_keys=True)
    base["_checksum"] = hashlib.md5(checksum_data.encode()).hexdigest()
    return jsonify_fast(base, code)

def validate_fields(data, required_fields):
    """Verifică existența câmpurilor obligatorii în request."""
    if not isinstance(data, dict):
        raise ValueError("Body JSON invalid.")
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        raise ValueError(f"Lipsesc câmpurile: {', '.join(missing)}")

# ===========================================================
# 🔍 UTILITĂȚI AI (Gemini + Extracție JSON)
# ===========================================================

def safe_json_extract(text):
    """Extragere robustă JSON din textul răspuns AI."""
    if not text:
        raise ValueError("Text gol primit pentru extracția JSON.")
    full_text = text.strip()
    if full_text.startswith('```json'):
        full_text = full_text.replace('```json', '', 1).strip()
    if full_text.endswith('```'):
        full_text = full_text[:-3].strip()

    try:
        return json.loads(full_text)
    except json.JSONDecodeError:
        start_index = full_text.find('{')
        end_index = full_text.rfind('}') + 1
        if start_index == -1 or end_index == -1:
            raise ValueError("Format JSON invalid sau incomplet.")
        return json.loads(full_text[start_index:end_index])

def call_gemini_raw(prompt):
    """Apelează modelul Gemini și returnează text brut."""
    if gemini_client is None:
        return {"error": "Eroare configurare server", "details": "Client AI neinițializat."}
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return {"error": "Eroare comunicare AI", "details": str(e)}

def call_gemini_json(prompt):
    """Apelează Gemini și extrage JSON valid."""
    raw = call_gemini_raw(prompt)
    if isinstance(raw, dict) and "error" in raw:
        return raw
    try:
        return safe_json_extract(raw)
    except Exception as e:
        return {"error": "Eroare parsare JSON", "details": str(e), "raw_text": raw[:400]}

# ===========================================================
# 🔹 ROUTE DEFINITIONS (API)
# ===========================================================

@app.route('/process-text', methods=['POST'])
def process_text():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['text'])
        job_text = data['text'].strip()

        prompt = (
            f"Analizează această descriere de job: '{job_text}'. "
            "Extrage informațiile cheie (rol, cerințe, responsabilități) și oferă un rezumat scurt (max 4 paragrafe)."
        )

        raw = call_gemini_raw(prompt)
        if isinstance(raw, dict) and "error" in raw:
            return api_response(error=raw.get("error"), code=500)
        return api_response(payload={"t": raw})
    except Exception as e:
        traceback.print_exc()
        return api_response(error=str(e), code=400)

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['cv_text', 'job_summary'])
        prompt = (
            f"Ești un recrutor AI. Pe baza rezumatului postului: {data['job_summary']} "
            f"și CV: {data['cv_text']}, generează 5 întrebări de interviu comportamentale relevante. "
            "Returnează JSON strict: {'questions': ['Întrebarea 1?', 'Întrebarea 2?', ...]}"
        )
        res = call_gemini_json(prompt)
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        return api_response(error=str(e), code=400)

@app.route('/analyze-cv', methods=['POST'])
def analyze_cv():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['cv_text', 'job_text'])
        prompt = f"""
        Evaluează compatibilitatea CV-ului cu Job-ul:
        CV: {data['cv_text']}
        Job: {data['job_text']}
        Returnează JSON strict:
        {{
          "compatibility_percent": 0-100,
          "feedback_markdown": "Feedback detaliat în Markdown"
        }}
        """
        res = call_gemini_json(prompt)
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        return api_response(error=str(e), code=400)

@app.route('/generate-job-queries', methods=['POST'])
def generate_job_queries():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['cv_text'])
        prompt = f"Generează 5-10 interogări optimizate pentru job hunt bazate pe CV:\n{data['cv_text']}\nReturnează JSON cu 'queries': ['q1','q2',...]"
        res = call_gemini_json(prompt)
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        return api_response(error=str(e), code=400)

@app.route('/generate-cover-letter', methods=['POST'])
def generate_cover_letter():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['cv_text', 'job_summary'])
        prompt = f"Generează o scrisoare de intenție bazată pe:\nCV: {data['cv_text']}\nJOB: {data['job_summary']}\nReturnează JSON cu 'cover_letter'."
        res = call_gemini_json(prompt)
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        return api_response(error=str(e), code=400)

@app.route('/optimize-linkedin-profile', methods=['POST'])
def optimize_linkedin_profile():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['cv_text'])
        prompt = f"Oferă recomandări pentru optimizarea profilului LinkedIn bazat pe CV:\n{data['cv_text']}\nReturnează JSON cu 'linkedin_tips': [...]."
        res = call_gemini_json(prompt)
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        return api_response(error=str(e), code=400)

@app.route('/generate-beginner-faq', methods=['POST'])
def generate_beginner_faq():
    try:
        data = request.get_json(force=True)
        cv_text = data.get('cv_text', '').strip()
        prompt = (
            f"Ești un recrutor AI. Generează 5 întrebări FAQ pentru începători bazate pe CV:\n{cv_text or 'Standard entry-level'}\n"
            "Returnează JSON: {'questions':[{'q':'Întrebarea?','exp':'Explicație Markdown'}]}"
        )
        res = call_gemini_json(prompt)
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        return api_response(error=str(e), code=400)

@app.route('/evaluate-answer', methods=['POST'])
def evaluate_answer():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['question', 'answer'])
        history = data.get('history', [])
        history_text = "\n".join([f"Q:{h.get('question')} A:{h.get('answer')}" for h in history])

        prompt = f"""
        Evaluează răspunsul utilizatorului.
        Context:
        {history_text}
        Întrebare: {data['question']}
        Răspuns: {data['answer']}
        Returnează JSON strict:
        {{
          "current_evaluation": {{"nota_finala":0-10,"claritate":0-10,"relevanta":0-10,"structura":0-10,"feedback":"Markdown"}},
          "comparative_feedback": {{"feedback":"Markdown"}}
        }}
        """
        res = call_gemini_json(prompt)
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        return api_response(error=str(e), code=400)

@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['history', 'job_summary', 'cv_text'])
        faq_history = data['history']
        if not faq_history:
            return api_response(error="Istoric interviu gol", code=400)

        hist = ""
        for i, h in enumerate(faq_history):
            q, a, ev = h.get('question', ''), h.get('answer', ''), h.get('evaluation', {})
            hist += f"Q{i+1}: {q}\nA:{a}\nNote:{ev.get('nota_finala','N/A')}/10\nFeedback:{ev.get('feedback','')}\n"

        prompt = f"""
        Ești un Career Coach AI. Generează raport final.
        Format JSON:
        {{
          "final_score": "medie scoruri",
          "summary": "Markdown",
          "key_strengths": ["3 puncte forte"],
          "areas_for_improvement": ["3 arii de îmbunătățire"],
          "next_steps_recommendation": "Text"
        }}
        Istoric:\n{hist}\nJOB:\n{data['job_summary']}\nCV:\n{data['cv_text']}
        """
        res = call_gemini_json(prompt)
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        return api_response(error=str(e), code=400)

@app.route('/coach-next', methods=['POST'])
def coach_next():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['question', 'user_answer'])
        prompt = f"Rescrie răspunsul utilizatorului într-un format STAR. Returnează DOAR textul rezultat."
        res = call_gemini_raw(f"{prompt}\nÎntrebare:{data['question']}\nRăspuns:{data['user_answer']}")
        if isinstance(res, dict) and "error" in res:
            return api_response(error=res["error"], code=500)
        return api_response(payload={"q": data['question'], "a": data['user_answer'], "star": res})
    except Exception as e:
        return api_response(error=str(e), code=400)

# ===========================================================
# 🚀 PORNIRE SERVER
# ===========================================================
if __name__ == '__main__':
    print("🚀 Server Flask compact și robust pornit pe http://0.0.0.0:5000/")
    # Pentru producție: folosește gunicorn
    # app.run(host='0.0.0.0', port=5000, debug=False)
    pass
