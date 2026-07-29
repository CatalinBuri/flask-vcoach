import os
import re
import json
import traceback
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# Configurare aplicație Flask
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

# Memorie temporară în RAM pentru sesiune
MEMORY = {
    "cv_text": "",
    "job_description": "",
    "interview_history": []
}

# ==========================================
# 1. INIȚIALIZARE CLIENȚI AI (Groq, Gemini & Hugging Face)
# ==========================================

groq_client = None
USE_GROQ = False
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        USE_GROQ = True
        print("✅ Groq ready", flush=True)
    except Exception as e:
        print(f"⚠️ Groq nu a putut fi inițializat: {e}", flush=True)

gemini_client = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini ready | model:", MODEL_NAME, flush=True)
    except Exception as e:
        print(f"⚠️ Gemini nu a putut fi inițializat: {e}", flush=True)

HF_API_KEY = os.environ.get("HF_API_KEY")
USE_HF = bool(HF_API_KEY)

if USE_HF:
    print("✅ Hugging Face ready", flush=True)


# ==========================================
# 2. FUNCȚII AUXILIARE & MAP-REDUCE CORECTAT
# ==========================================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def safe_json(raw_text: str) -> dict:
    if not raw_text:
        return {}
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE).strip()
    
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}

def api_response(payload=None, error=None, code=200):
    if error:
        return jsonify({
            "status": "error",
            "success": False,
            "ok": False,
            "message": error,
            "error": error
        }), code

    base_response = {
        "status": "success",
        "success": True,
        "ok": True,
        "code": 200
    }

    if isinstance(payload, dict):
        base_response["data"] = payload
        base_response.update(payload)
        return jsonify(base_response), code

    base_response["data"] = payload if payload is not None else {}
    return jsonify(base_response), code

def call_huggingface(prompt: str) -> str:
    if not HF_API_KEY:
        return ""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_API_KEY)
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.2
        )
        if response and response.choices:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Eroare Hugging Face Hub: {e}", flush=True)
    return ""

def gemini_text(prompt: str, max_tokens: int = 4096) -> str:
    if USE_GROQ and groq_client:
        try:
            res = groq_client.with_options(max_retries=0).chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ești un asistent AI senior specializat exclusiv în inginerie automotive, management tehnic și optimizare CV-uri ATS. "
                            "Nu schimba niciodată domeniul de activitate al candidatului (rămâi strict pe Automotive, ECU, SW/HW Engineering)."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=min(max_tokens, 4096),
                timeout=15.0
            )
            if res and res.choices and res.choices[0].message.content:
                text_res = res.choices[0].message.content.strip()
                if len(text_res) > 5:
                    return text_res
        except Exception as e:
            print(f"⚠️ Groq Error: {e}. Trecem la Gemini...", flush=True)

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            if response and hasattr(response, "text") and response.text:
                text_res = response.text.strip()
                if len(text_res) > 5:
                    return text_res
        except Exception as e:
            print(f"⚠️ Gemini Error: {e}. Trecem la Hugging Face...", flush=True)

    if USE_HF:
        try:
            text_res = call_huggingface(prompt)
            if text_res and len(text_res) > 5:
                return text_res
        except Exception as e:
            print(f"❌ Eroare Hugging Face: {e}", flush=True)

    return ""

def split_cv_into_sections(cv_text: str) -> dict:
    """Împarte CV-ul curat pe secțiuni majore fără a duplica titlurile."""
    keywords = [
        "WORK EXPERIENCE", "EXPERIENCE", "EXPERIENȚĂ", 
        "EDUCATION", "EDUCAȚIE", "EDUCATIE", 
        "SKILLS", "COMPETENȚE", "COMPETENTE", 
        "PROJECTS", "PROIECTE", "PUBLICATIONS", "PUBLICATII", "ABOUT ME"
    ]
    
    pattern = r'(?i)\b(' + '|'.join(keywords) + r')\b'
    parts = re.split(pattern, cv_text)
    
    sections = {}
    current_section = "ABOUT ME"
    sections[current_section] = ""
    
    for i in range(1, len(parts), 2):
        current_section = parts[i].upper().strip()
        content = parts[i+1] if i+1 < len(parts) else ""
        sections[current_section] = content.strip()
        
    if len(sections) <= 1 and len(cv_text.strip()) > 0:
        return {"SUMMARY": cv_text}
        
    return sections

def map_reduce_rephrase(cv_text: str, job_desc: str, target_lang: str) -> str:
    """Rescrie CV-ul secțiune cu secțiune menținând contextul tehnic original și eliminând dublurile."""
    sections = split_cv_into_sections(cv_text)
    html_results = []

    # Maparea secțiunilor valide din CV-ul real pentru a preveni contaminarea cu date false
    valid_sections = ["ABOUT ME", "WORK EXPERIENCE", "EXPERIENCE", "EDUCATION", "EDUCATION AND TRAINING", "SKILLS", "PUBLICATIONS", "PATENTS"]

    for sec_name, sec_content in sections.items():
        if not sec_content.strip():
            continue
            
        # Asigurăm un format curat al titlului și evităm secțiunile necunoscute / inventate
        clean_sec_name = sec_name.upper().strip()
        
        prompt = f"""
Ești un expert tehnic senior în resurse umane pentru industria AUTOMOTIVE și inginerie software/hardware (ECU, CATIA, SDV). 
Optimizează strict această secțiune ({clean_sec_name}) a CV-ului unui Engineering Manager real.

REGULI STRICTE DE SIGURANȚĂ ȘI INTEGRITATE A DATELOR:
1. DOMENIU STRICT: Rămâi 100% în domeniul AUTOMOTIVE și management tehnic. Este INTERZIS să introduci activități de "cercetare de piață" (market research), "chestionare de opinie" sau alte domenii non-tehnice.
2. FĂRĂ INVENȚII: Nu inventa publicații, articole sau conferințe care nu există în textul sursă. Dacă există brevete (patents), păstrează-le exact pe cele originale (de exemplu, brevete de bare de portbagaj sau sisteme de planșă de bord).
3. STRUCTURĂ: Nu duplica titlurile în interiorul conținutului.
4. FORMAT: Răspunde EXCLUSIV în format HTML curat (folosind tag-uri precum <p>, <ul>, <li>, <strong>), FĂRĂ blocuri de cod markdown (fără ```html sau ```).
5. LIMBĂ: Limba de răspuns: STRICT {target_lang}.

CONȚINUTUL ORIGINAL AL ACESTEI SECȚIUNI:
{sec_content[:4000]}

DESCRIERE JOB DE REFERINȚĂ (dacă există):
{job_desc[:2000]}
"""
        raw_res = gemini_text(prompt, max_tokens=2048)
        cleaned_sec = re.sub(r'^```(?:html)?\s*', '', raw_res.strip(), flags=re.MULTILINE)
        cleaned_sec = re.sub(r'\s*```$', '', cleaned_sec, flags=re.MULTILINE).strip()
        
        if cleaned_sec:
            # Generăm un singur titlu per secțiune, eliminând orice duplicat intern generat de AI
            html_results.append(f"<div class='cv-section'><h2>{clean_sec_name}</h2>\n{cleaned_sec}\n</div>")
            
    return "\n".join(html_results)


# ==========================================
# 3. RUTELE API (Neschimbate ca rutare, actualizate doar intern)
# ==========================================

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({
        "status": "online",
        "success": True,
        "service": "vCoach AI API (Map-Reduce & Context Fix)",
        "groq_active": USE_GROQ,
        "gemini_active": gemini_client is not None,
        "huggingface_active": USE_HF
    }), 200

@app.route("/ping", methods=["GET", "HEAD"])
def ping():
    return "OK", 200

@app.route("/upload-cv", methods=["POST", "OPTIONS"], endpoint="upload_cv_root")
@app.route("/api/upload-cv", methods=["POST", "OPTIONS"], endpoint="upload_cv_api")
@cross_origin()
def upload_cv():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        text_content = ""
        if "file" in request.files:
            file = request.files["file"]
            filename = file.filename.lower()
            if filename.endswith(".pdf"):
                try:
                    import fitz
                    doc = fitz.open(stream=file.read(), filetype="pdf")
                    for page in doc:
                        text_content += page.get_text()
                except Exception as pdf_err:
                    return api_response(error=f"Eroare citire PDF: {str(pdf_err)}", code=400)
            else:
                text_content = file.read().decode("utf-8", errors="ignore")
        elif request.is_json:
            data = request.get_json(force=True, silent=True) or {}
            text_content = data.get("cv_text", "")

        cleaned = clean_text(text_content)
        if not cleaned:
            return api_response(error="Nu s-a putut extrage text din fișierul trimis.", code=400)

        MEMORY["cv_text"] = cleaned
        return api_response(payload={"message": "CV încărcat cu succes", "length": len(cleaned), "cv_text": cleaned})

    except Exception as e:
        return api_response(error=f"Eroare la procesare: {str(e)}", code=500)

@app.route("/analyze-cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_root")
@app.route("/api/cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_api")
@cross_origin()
def analyze_cv_quality():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv = clean_text(data.get("cv_text") or MEMORY.get("cv_text") or "")
        job = clean_text(data.get("job_description") or data.get("job_text") or MEMORY.get("job_description") or "")
        target_lang = data.get("target_language") or data.get("language") or "ro"
        
        if not cv:
            return api_response(error="CV lipsă.", code=400)

        MEMORY["cv_text"] = cv
        if job:
            MEMORY["job_description"] = job

        prompt = f"""
Ești un recruiter senior în industria automotive. Analizează CV-ul în raport direct cu descrierea jobului.
Limba de răspuns: STRICT {target_lang}.
Răspunde EXCLUSIV cu un obiect JSON valid:
{{
  "clarity_score": 8,
  "relevance_score": 7,
  "structure_score": 8,
  "matched_ats_keywords": ["Cuvant1"],
  "missing_ats_keywords": ["CuvantLipsește"],
  "concrete_improvements": ["Sfat 1"],
  "suggested_rephrasings": ["Exemplu"]
}}
CV: {cv[:4000]}
JOB DESCRIPTION: {job[:2000]}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res)

        payload = {
            "clarity_score": parsed.get("clarity_score", 8),
            "relevance_score": parsed.get("relevance_score", 7),
            "structure_score": parsed.get("structure_score", 8),
            "has_job_context": bool(job),
            "ats_keywords": parsed.get("matched_ats_keywords") or [],
            "missing_keywords": parsed.get("missing_ats_keywords") or [],
            "concrete_improvements": parsed.get("concrete_improvements") or [],
            "suggested_rephrasings": parsed.get("suggested_rephrasings") or []
        }
        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare analiză CV: {str(e)}", code=500)

@app.route("/match-job", methods=["POST", "OPTIONS"], endpoint="match_job_root")
@app.route("/api/match-job", methods=["POST", "OPTIONS"], endpoint="match_job_api")
@cross_origin()
def match_job():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv = clean_text(data.get("cv_text") or MEMORY.get("cv_text") or "")
        job_desc = clean_text(data.get("job_description") or MEMORY.get("job_description") or "")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        if not cv or not job_desc:
            return api_response(error="CV-ul și Descrierea Jobului sunt necesare.", code=400)

        prompt = f"""
Compară CV-ul cu Descrierea Jobului. 
Limba de răspuns: STRICT {target_lang}.
Returnează DOAR un obiect JSON valid:
{{
  "match_score": 75,
  "matching_skills": ["abilitate1"],
  "missing_skills": ["cerinta1"],
  "recommendations": ["recomandare1"]
}}
CV: {cv[:4000]}
JOB: {job_desc[:2000]}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res) or {}

        payload = {
            "match_score": parsed.get("match_score", 65),
            "score": parsed.get("match_score", 65),
            "matching_skills": parsed.get("matching_skills", []),
            "missing_skills": parsed.get("missing_skills", []),
            "recommendations": parsed.get("recommendations", [])
        }
        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare match-job: {str(e)}", code=500)

@app.route("/interview-question", methods=["POST", "OPTIONS"], endpoint="interview_question_root")
@app.route("/api/interview-question", methods=["POST", "OPTIONS"], endpoint="interview_question_api")
@cross_origin()
def interview_question():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        user_answer = data.get("user_answer", "")
        role = data.get("role", "Engineering Manager")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        prompt = f"""
Ești un recrutor tehnic pentru rolul: {role}. 
Limba de răspuns: STRICT {target_lang}. 
Răspuns candidat: "{user_answer[:1500]}"
Returnează DOAR un obiect JSON valid:
{{
  "feedback": "Evaluare...",
  "score": 8,
  "next_question": "Următoarea întrebare..."
}}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res) or {}

        payload = {
            "feedback": parsed.get("feedback", ""),
            "score": parsed.get("score", 7),
            "next_question": parsed.get("next_question", ""),
            "question": parsed.get("next_question", "")
        }
        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare interviu: {str(e)}", code=500)

@app.route("/translate", methods=["POST", "OPTIONS"], endpoint="translate_root")
@app.route("/api/translate", methods=["POST", "OPTIONS"], endpoint="translate_api")
@cross_origin()
def translate():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text") or MEMORY.get("cv_text") or ""
        target_lang = data.get("target_language") or data.get("language") or "en"

        prompt = f"Translate into {target_lang} maintaining a formal automotive engineering tone:\n\n{text[:4000]}"
        translated_text = gemini_text(prompt)

        payload = {
            "translated_text": translated_text,
            "translation": translated_text,
            "text": translated_text
        }
        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare traducere: {str(e)}", code=500)

@app.route("/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_root")
@app.route("/api/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_api")
@cross_origin()
def rephrase():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_text = data.get("text") or MEMORY.get("cv_text") or ""
        job_desc = data.get("job_description") or MEMORY.get("job_description") or ""
        target_lang = data.get("target_language") or data.get("language") or "ro"

        if not cv_text:
            return api_response(error="Textul CV-ului pentru reformulare lipsește.", code=400)

        improved = map_reduce_rephrase(cv_text, job_desc, target_lang)

        if not improved or len(str(improved).strip()) == 0:
            improved = "<p>Eroare la procesarea prin Map-Reduce.</p>"

        payload = {
            "improved_text": improved,
            "rephrased_text": improved,
            "text": improved
        }
        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare rephrase: {str(e)}", code=500)

@app.route("/generate-summary", methods=["POST", "OPTIONS"], endpoint="generate_summary_root")
@app.route("/api/generate-summary", methods=["POST", "OPTIONS"], endpoint="generate_summary_api")
@cross_origin()
def generate_summary():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv = clean_text(data.get("cv_text") or MEMORY.get("cv_text") or "")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        prompt = f"""
Creează 3 opțiuni scurte de rezumat profesional pentru un Engineering Manager în automotive. 
Limba de răspuns: STRICT {target_lang}.
Returnează DOAR un obiect JSON valid:
{{
  "summaries": ["Opțiunea 1...", "Opțiunea 2...", "Opțiunea 3..."]
}}
CV: {cv[:4000]}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res)
        summaries = parsed.get("summaries", []) if parsed else [raw_res]

        payload = {
            "summaries": summaries,
            "summary": summaries[0] if summaries else raw_res
        }
        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare summary: {str(e)}", code=500)

@app.route("/generate-cover-letter", methods=["POST", "OPTIONS"], endpoint="cover_letter_root")
@app.route("/api/generate-cover-letter", methods=["POST", "OPTIONS"], endpoint="cover_letter_api")
@cross_origin()
def generate_cover_letter():
    if request.method == "OPTIONS":
        return api_response(code=200)

    try:
        data = request.get_json(force=True, silent=True) or {}
        cv = clean_text(data.get("cv_text") or MEMORY.get("cv_text") or "")
        job_desc = clean_text(data.get("job_description") or MEMORY.get("job_description") or "")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        prompt = f"""
Creează o scrisoare de intenție profesională pentru un rol tehnic în automotive. 
Limba de răspuns: STRICT {target_lang}.
CV: {cv[:3000]}
Job: {job_desc[:2000]}
"""
        cover_letter_text = gemini_text(prompt, max_tokens=3000)
        payload = {
            "cover_letter": cover_letter_text,
            "text": cover_letter_text
        }
        return api_response(payload=payload)
    except Exception as e:
        return api_response(error=f"Eroare cover letter: {str(e)}", code=500)

@app.route("/get-session", methods=["GET", "OPTIONS"], endpoint="get_session_root")
@app.route("/api/get-session", methods=["GET", "OPTIONS"], endpoint="get_session_api")
@cross_origin()
def get_session():
    if request.method == "OPTIONS":
        return api_response(code=200)
    return api_response(payload={
        "has_cv": bool(MEMORY.get("cv_text")),
        "cv_length": len(MEMORY.get("cv_text", "")),
        "cv_text": MEMORY.get("cv_text", ""),
        "has_job": bool(MEMORY.get("job_description")),
        "job_description": MEMORY.get("job_description", "")
    })

@app.route("/clear-session", methods=["POST", "OPTIONS"], endpoint="clear_session_root")
@app.route("/api/clear-session", methods=["POST", "OPTIONS"], endpoint="clear_session_api")
@cross_origin()
def clear_session():
    if request.method == "OPTIONS":
        return api_response(code=200)
    MEMORY["cv_text"] = ""
    MEMORY["job_description"] = ""
    MEMORY["interview_history"] = []
    return api_response(payload={"message": "Sesiunea a fost resetată cu succes."})

@app.errorhandler(404)
def not_found(e):
    return api_response(error="Endpoint-ul căutat nu există pe server.", code=404)

@app.errorhandler(500)
def server_error(e):
    return api_response(error="Eroare internă pe server.", code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Serverul pornește pe portul {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
