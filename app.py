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
    checksum_data = orjson.dumps(payload or {}, option=orjson.OPT_SORT_KEYS)
    base["_checksum"] = hashlib.md5(checksum_data).hexdigest()
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
        # 1. Obține datele JSON trimise de JavaScript
        data = request.get_json(force=True)
        
        # 2. Validează (opțional, dar recomandat)
        validate_fields(data, ['cv_text', 'job_text'])
        
        # 3. EXTRASE VARIABILELE DIN DICTIONARUL 'data'
        cv_text = data.get('cv_text', '')  # Variabila cv_text este DEFINITĂ AICI!
        job_text = data.get('job_text', '') # Variabila job_text este DEFINITĂ AICI!

        # 4. Construiește prompt-ul (acum cv_text și job_text sunt definite)
        prompt = f"""
        Ești un expert în resurse umane. Analizează următorul CV în raport cu descrierea postului.
        Obiectivul tău este să returnezi **DOAR** un obiect JSON care respectă STRICT următoarea schemă:
        {{
          "compatibility_percent": <un număr întreg de la 0 la 100 care reprezintă scorul de potrivire>,
          "feedback_markdown": "<O analiză detaliată și constructivă, formatată în Markdown, care explică scorul, punctele forte și lacunele CV-ului în raport cu jobul. NU include cod JSON sau alte marcaje în acest câmp.>"
        }}

---
CV:
{cv_text}

---
JOB DESCRIPTION:
{job_text}

Răspunde DOAR cu obiectul JSON.
"""
        res = call_gemini_json(prompt)
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        traceback.print_exc() 
        return api_response(error=f"Eroare internă. Detaliu: {str(e)}", code=500)
        return api_response(error=str(e), code=400)

@app.route('/generate-job-queries', methods=['POST'])
def generate_job_queries():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['cv_text'])
        
        # 1. CORECȚIE: Variabila cv_text trebuie DEFINITĂ din data primită
        cv_text = data.get('cv_text', '')
        
        # Am redenumit PROMPT_JOB_HUNT în 'prompt' pentru a se potrivi cu apelul de mai jos
        prompt = f"""
        Ești un expert în căutarea de joburi. Analizează următorul CV și generează o listă de 7 interogări de căutare (query-uri) extrem de eficiente și realiste, potrivite pentru motoare de căutare de joburi precum LinkedIn și eJobs.

        Reguli stricte:
        1. Returnează DOAR un obiect JSON cu schema solicitată.
        2. Interogările generate trebuie să fie scurte (maxim 4 cuvinte).
        3. Nu folosi operatori logici booleeni (AND, OR, NOT).
        4. Concentrează fiecare interogare pe un Rol, o Competență Cheie sau o Combinație Rol + Industrie.

        Schema JSON AȘTEPTATĂ:
{{
  "queries": ["Interogare 1", "Interogare 2", "Interogare 3", "Interogare 4", "Interogare 5", "Interogare 6", "Interogare 7"]
}}

---
CV:
{cv_text}
"""
        # 2. CORECȚIE: Variabila 'prompt' este acum definită corect
        res = call_gemini_json(prompt) 
        
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
    except Exception as e:
        # Păstrăm logica de eroare 400 pentru validări eșuate
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
        validate_fields(data, ['cv_text', 'domain']) 
        
        cv_text = data['cv_text']
        domain = data.get('domain', '') 
        
        domain_context = f"pentru postul din domeniul: {domain}" if domain else ""
        
        # 🟢 CORECȚIE CRITICĂ: Instrucțiunea către AI pentru a genera cele două chei distincte
        prompt = (
            f"Ești un expert în optimizare LinkedIn. Analizează CV-ul de mai jos și generează recomandări stricte de conținut {domain_context}.\n"
            "Returnează DOAR un obiect JSON care respectă STRICT următoarea schemă:\n"
            "{\n"
            "  \"linkedin_headlines\": [\"Sloganul 1\", \"Sloganul 2\", \"Sloganul 3\"], \n"
            "  \"linkedin_about\": \"O secțiune 'Despre mine' profesională, formatată în Markdown, bazată pe CV.\"\n"
            "}\n"
            f"CV:\n{cv_text}"
        )
        
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
            "Returnează JSON: {'faq':[{'q':'Întrebarea?','exp':'Explicație Markdown'}]}"
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
            # Asigură-te că h este un dicționar înainte de a apela .get()
            if not isinstance(h, dict):
                # Opțional: forțează un mesaj de eroare clar dacă un element nu e dict
                raise ValueError(f"Istoric invalid la elementul {i}. Nu este dicționar.")
            
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
        # Dacă răspunsul AI are eroare, ar trebui să returneze 500, nu 400
        return api_response(payload=res) if "error" not in res else api_response(error=res["error"], code=500)
        
    except Exception as e:
        # ⚠️ LINIA ESENȚIALĂ ADAUGATĂ PENTRU DIAGNOZĂ
        traceback.print_exc() 
        return api_response(error=f"Eroare internă. Verifică log-urile. Detaliu: {str(e)}", code=500)

@app.route('/coach-next', methods=['POST'])
def coach_next():
    try:
        data = request.get_json(force=True)
        validate_fields(data, ['question', 'user_answer'])
        user_answer = data['user_answer'].strip() # Extrage și curăță răspunsul

        # 🟢 VERIFICAREA LOGICĂ A RĂSPUNSULUI SCURT
        if len(user_answer.split()) < 5: 
            error_message = "Răspunsul este prea scurt (min. 5 cuvinte) pentru o analiză STAR relevantă."
            # Răspunsul este trimis înapoi ca "star_answer" pentru ca frontend-ul să îl afișeze corect
            return api_response(payload={"q": data['question'], "a": user_answer, "star_answer": error_message})
        
        
        prompt = f"Rescrie răspunsul utilizatorului într-un format STAR. Returnează DOAR textul rezultat."
        res = call_gemini_raw(f"{prompt}\nÎntrebare:{data['question']}\nRăspuns:{user_answer}")
        
        if isinstance(res, dict) and "error" in res:
            return api_response(error=res["error"], code=500)
            
        # CORECȚIA ESENȚIALĂ: Asigură-te că cheia este "star_answer"
        return api_response(payload={"q": data['question'], "a": user_answer, "star_answer": res})
        
    except Exception as e:
        return api_response(error=str(e), code=400)
# ===========================================================
# 🚀 ROUTĂ DE WAKEUP / PING (Pentru Keep-Alive/Render Cron Jobs)
# ===========================================================

@app.route('/ping', methods=['GET'])
def ping_server():
    """
    Endpoint rapid pentru a răspunde cu succes (200 OK).
    Folosit de Render Cron Job sau servicii externe de Keep-Alive.
    """
    # Returnăm un răspuns minimalist, dar care confirmă starea serverului
    # Nu necesită api_response sau orjson, un simplu jsonify este suficient de rapid
    return jsonify({"status": "ok", "message": "Server is awake and responding."}), 200
# ===========================================================
# 🚀 PORNIRE SERVER
# ===========================================================
if __name__ == '__main__':
    print("🚀 Server Flask compact și robust pornit pe [http://0.0.0.0:5000/](http://0.0.0.0:5000/)")
    # Pentru producție: folosește gunicorn
    # app.run(host='0.0.0.0', port=5000, debug=False)
    pass

