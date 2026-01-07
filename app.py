import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
import orjson
from flask_compress import Compress

# =========================
# CONFIG
# =========================
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
Compress(app)

# =========================
# GEMINI INIT
# =========================
client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY)
    print("✅ Gemini ready")

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

def safe_json(text):
    if not text:
        return None
    text = text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
    return None

def gemini_text(prompt):
    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return res.text
    except Exception as e:
        print("Gemini error:", e)
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

    prompt = f"Fă un rezumat clar și concis:\n{text}"
    summary = gemini_text(prompt)
    return api_response(payload={"t": summary})

@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json(force=True)
    cv = data.get("cv_text", "")
    job = data.get("job_summary", "")

    if not cv or not job:
        return api_response(error="Date lipsă", code=400)

    prompt = f"""
Generează EXACT 5 întrebări de interviu.
Răspunde DOAR JSON:
{{"questions": ["..."]}}

CV:
{cv}

JOB:
{job}
"""

    raw = gemini_text(prompt)
    parsed = safe_json(raw)

    if not parsed or "questions" not in parsed:
        parsed = {
            "questions": [
                "Povestește despre experiența ta relevantă.",
                "Care sunt punctele tale forte?",
                "Cum gestionezi situațiile dificile?",
                "De ce îți dorești acest job?",
                "Unde te vezi peste 3 ani?"
            ]
        }

    return api_response(payload=parsed)

@app.route("/analyze-cv", methods=["POST"])
def analyze_cv():
    data = request.get_json(force=True)
    cv = data.get("cv_text", "")
    job = data.get("job_text", "")

    if not cv or not job:
        return api_response(error="Date lipsă", code=400)

    prompt = f"""
Analizează compatibilitatea CV vs Job.
Returnează JSON:
{{"compatibility_percent": 0-100, "feedback_markdown": "..."}}

CV:
{cv}

JOB:
{job}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)

    if not parsed:
        parsed = {
            "compatibility_percent": 70,
            "feedback_markdown": "Compatibilitate medie."
        }

    return api_response(payload=parsed)

@app.route("/generate-job-queries", methods=["POST"])
def generate_job_queries():
    data = request.get_json(force=True)
    cv = data.get("cv_text", "")

    prompt = f"""
Generează 7 căutări LinkedIn.
Returnează JSON:
{{"queries": []}}

CV:
{cv}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)

    if not parsed:
        parsed = {"queries": []}

    return api_response(payload=parsed)

@app.route("/optimize-linkedin-profile", methods=["POST"])
def optimize_linkedin():
    data = request.get_json(force=True)
    cv = data.get("cv_text", "")

    prompt = f"""
Optimizează profil LinkedIn.
Returnează JSON:
{{"linkedin_headlines": [], "linkedin_about": ""}}

CV:
{cv}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)

    if not parsed:
        parsed = {
            "linkedin_headlines": [],
            "linkedin_about": ""
        }

    return api_response(payload=parsed)

@app.route("/coach-next", methods=["POST"])
def coach_next():
    data = request.get_json(force=True)
    answer = data.get("user_answer", "")

    if len(answer.split()) < 5:
        return api_response(payload={"star_answer": "Răspuns prea scurt."})

    prompt = f"Rescrie răspunsul în format STAR:\n{answer}"
    text = gemini_text(prompt)

    return api_response(payload={"star_answer": text})

@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    data = request.get_json(force=True)
    question = data.get("question", "")
    answer = data.get("answer", "")

    prompt = f"""
Evaluează răspunsul.
Returnează JSON:
{{"nota_finala": 1-10, "feedback": "..."}}

Întrebare:
{question}

Răspuns:
{answer}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)

    if not parsed:
        parsed = {
            "nota_finala": 5,
            "feedback": "Răspuns acceptabil."
        }

    return api_response(payload={"current_evaluation": parsed})

@app.route("/generate-report", methods=["POST"])
def generate_report():
    data = request.get_json(force=True)
    history = data.get("history", [])

    prompt = f"""
Generează raport final.
Returnează JSON:
{{"summary": "...", "scor_final": 1-10}}

Istoric:
{history}
"""
    raw = gemini_text(prompt)
    parsed = safe_json(raw)

    if not parsed:
        parsed = {
            "summary": "Interviu finalizat.",
            "scor_final": 7
        }

    return api_response(payload=parsed)

# =========================
# START
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
