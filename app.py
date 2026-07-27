import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from google import genai
import orjson
from flask_compress import Compress
from groq import Groq
from itertools import zip_longest

# =========================
# CONFIG
# =========================
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)

app = Flask(__name__)

# Configurare CORS Permisivă
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})
Compress(app)

# =========================
# SHARED MEMORY
# =========================
MEMORY = {
    "cv_text": None
}

# =========================
# CLIENT INIT
# =========================
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"✅ Gemini ready | model: {MODEL_NAME}")
    except Exception as e:
        print(f"❌ Eroare la inițializarea Gemini Client: {str(e)}")

groq_client = None
if USE_GROQ:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq ready")
    except Exception as e:
        print(f"❌ Eroare la inițializarea Groq: {str(e)}")


def groq_text(prompt: str) -> str:
    if not groq_client:
        return ""
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Ești un expert LinkedIn Job Search și recruiter profesionist. Răspunde NUMAI cu JSON valid atunci când se solicită."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {str(e)}")
        return ""


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
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'[\x00-\x1F]+', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 2000) -> list:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def safe_json(text: str):
    if not text:
        return None
    text = clean_text(text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S | re.M)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def gemini_text(prompt: str) -> str:
    if USE_GROQ and groq_client:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ești un recrutor profesionist cu peste 10 ani de experiență. "
                            "Dacă ți se cere JSON, returnează NUMAI JSON valid fără markdown sau alt text. "
                            "Dacă ți se cere text simplu, răspunde curat și profesionist. "
                            "CRITICAL: Detect and strictly adhere to the source document's native language when giving recommendations."
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
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {type(e).__name__} - {str(e)}")

    return ""


# =========================
# GLOBAL ERROR HANDLER (Previne erori CORS la HTTP 500)
# =========================
@app.errorhandler(Exception)
def handle_global_exception(e):
    print(f"🔥 Unhandled Server Exception: {str(e)}")
    return api_response(error=f"A apărut o eroare pe server: {str(e)}", code=500)


# =========================
# ROUTES
# =========================

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"})


@app.route("/check-cv-memory", methods=["GET", "OPTIONS"])
@cross_origin()
def check_cv_memory():
    if request.method == "OPTIONS":
        return api_response(code=200)
    if MEMORY.get("cv_text") and len(MEMORY["cv_text"].strip()) > 10:
        return api_response(payload={"has_cv": True}, code=200)
    else:
        return api_response(error="No CV in memory", code=404)


@app.route("/clear-memory", methods=["POST", "OPTIONS"])
@cross_origin()
def clear_memory():
    if request.method == "OPTIONS":
        return api_response(code=200)
    MEMORY["cv_text"] = None
    return jsonify({
        "status": "ok",
        "payload": {"message": "Memoria CV a fost ștearsă cu succes"}
    })


@app.route("/analyze-cv-quality", methods=["POST", "OPTIONS"])
@app.route("/api/cv-quality", methods=["POST", "OPTIONS"])
@cross_origin()
def analyze_cv_quality():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_description") or data.get("job_text") or ""

        cv = clean_text(cv_raw)

        if not cv:
            return api_response(error="CV lipsă în request sau memorie", code=400)

        MEMORY["cv_text"] = cv

        context_extra = ""
        if job_raw:
            context_extra = f"\n\nJOB DESCRIPTION / CERINȚE POST:\n{clean_text(job_raw)}"

        chunks = chunk_text(cv, chunk_size=3000)

        clarity_scores, relevance_scores, structure_scores = [], [], []
        concrete_improvements, suggested_rephrasings = [], []

        for chunk in chunks:
            prompt_chunk = f"""
You are a senior hybrid recruiter with 10+ years of experience. Analyze ONLY the CV fragment below in relation to the job description provided.

CRITICAL RULES - MUST FOLLOW EXACTLY:
1. Detect the dominant language of the fragment. ALL output (scores, concrete_improvements, suggested_rephrasings) MUST be written STRICTLY IN THAT LANGUAGE ONLY.
2. Do NOT use numbering, prefixes, "Improvement 1:", "Rephrasing 1:", "1.", or bullet points inside the output array strings.
3. For "suggested_rephrasings" use EXACT format:
   "Original: \"exact original phrase\", Improved: \"better version\""
4. Return ONLY valid JSON — nothing else.

Assign scores 0–10:
- clarity_score: clarity & readability
- relevance_score: attractiveness to recruiters
- structure_score: logical flow & organization

JSON structure (strict):
{{
  "clarity_score": int,
  "relevance_score": int,
  "structure_score": int,
  "concrete_improvements": ["suggestion with example...", ...],
  "suggested_rephrasings": [
    "Original: \"...\", Improved: \"...\"",
    ...
  ]
}}

CV fragment:
{chunk}
{context_extra}
"""
            raw_chunk = gemini_text(prompt_chunk)
            parsed_chunk = safe_json(raw_chunk)

            if not parsed_chunk or not isinstance(parsed_chunk, dict):
                parsed_chunk = {
                    "clarity_score": 7, "relevance_score": 7, "structure_score": 7,
                    "concrete_improvements": [], "suggested_rephrasings": []
                }

            clarity_scores.append(parsed_chunk.get("clarity_score", 7))
            relevance_scores.append(parsed_chunk.get("relevance_score", 7))
            structure_scores.append(parsed_chunk.get("structure_score", 7))

            improvements = parsed_chunk.get("concrete_improvements", [])
            if isinstance(improvements, list):
                concrete_improvements.extend(improvements)

            rephrasings = parsed_chunk.get("suggested_rephrasings", [])
            if isinstance(rephrasings, list):
                suggested_rephrasings.extend(rephrasings)

        final_payload = {
            "clarity_score": int(sum(clarity_scores) / len(clarity_scores)) if clarity_scores else 7,
            "relevance_score": int(sum(relevance_scores) / len(relevance_scores)) if relevance_scores else 7,
            "structure_score": int(sum(structure_scores) / len(structure_scores)) if structure_scores else 7,
            "overall_assessment": "CV analysis completed successfully.",
            "concrete_improvements": concrete_improvements[:10],
            "suggested_rephrasings": suggested_rephrasings[:10]
        }

        return api_response(payload=final_payload)

    except Exception as e:
        print(f"❌ Error inside analyze_cv_quality: {str(e)}")
        return api_response(error=f"Eroare internă server la analiza CV: {str(e)}", code=500)


@app.route("/analyze-cv", methods=["POST", "OPTIONS"])
@cross_origin()
def analyze_cv():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_raw = data.get("cv_text") or MEMORY.get("cv_text") or ""
        job_raw = data.get("job_description") or data.get("job_text") or ""

        if not cv_raw or not job_raw:
            return api_response(error="Date lipsă: CV și Job sunt necesare", code=400)

        cv_clean = clean_text(cv_raw)
        job_clean = clean_text(job_raw)
        MEMORY["cv_text"] = cv_clean

        cv_chunks = chunk_text(cv_clean, chunk_size=3000)
        job_chunks = chunk_text(job_clean, chunk_size=3000)

        chunk_feedbacks, chunk_scores = [], []

        for i, (cv_chunk, job_chunk) in enumerate(zip_longest(cv_chunks, job_chunks, fillvalue="")):
            prompt_chunk = f"""
Ești un recrutor profesionist. Analizează compatibilitatea dintre CV și cerințele postului.
Oferă procentaj realist (0-100) și feedback detaliat.

Returnează NUMAI JSON valid:
{{"compatibility_percent": int, "feedback_markdown": "text curat și profesionist"}}

CV fragment: {cv_chunk}
Job fragment: {job_chunk}
"""
            raw_chunk = gemini_text(prompt_chunk)
            parsed_chunk = safe_json(raw_chunk) or {
                "compatibility_percent": 70,
                "feedback_markdown": "Fragmentul CV-ului are relevanță parțială pentru cerințele acestui segment al jobului."
            }

            chunk_feedbacks.append(parsed_chunk.get("feedback_markdown", ""))
            chunk_scores.append(parsed_chunk.get("compatibility_percent", 70))

        combined_feedback = "\n\n".join(chunk_feedbacks)

        final_prompt = f"""
Rescrie feedback-ul combinat într-un text profesionist, fluent și corect, în limba dominantă a analizei.
Text combinat:
{combined_feedback}
Returnează NUMAI text curat.
"""
        res_final = gemini_text(final_prompt)
        if not res_final.strip():
            res_final = combined_feedback

        final_score = int(sum(chunk_scores) / len(chunk_scores)) if chunk_scores else 75

        return api_response(payload={
            "compatibility_percent": final_score,
            "feedback_markdown": res_final
        })

    except Exception as e:
        return api_response(error=f"Eroare internă: {str(e)}", code=500)


# =========================
# START
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
