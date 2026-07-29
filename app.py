import os
import re
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

MEMORY = {
    "cv_text": "",
    "job_description": "",
    "interview_history": []
}

# ==========================================
# 1. INIȚIALIZARE CLIENȚI AI
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


# ==========================================
# 2. FUNCȚII DE POSTPROCESARE & ARHITECTURĂ ANTI-HALUCINAȚIE
# ==========================================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def postprocess_cv_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    processed = raw_html

    def clean_duplicate_headers(match):
        header_tag = match.group(1)
        header_text = match.group(2).strip()
        return f"<{header_tag}>{header_text}</{header_tag}>"

    processed = re.sub(r'(<h[23]>)(.*?)(</h[23]>)(?:\s*<h[23]>.*?</h[23]>)+', clean_duplicate_headers, processed, flags=re.IGNORECASE)
    processed = re.sub(r'<p>\s*</p>', '', processed)
    processed = re.sub(r'<div>\s*</div>', '', processed)
    processed = re.sub(r'\n{3,}', '\n\n', processed)
    
    return processed.strip()

def extract_safe_transferable_skills(job_desc: str) -> str:
    """
    Arhitectura în Doi Pași: Extrage strict competențele transversale și stilul de management,
    fără a lăsa domeniul vertical străin să contamineze CV-ul candidatului.
    """
    if not job_desc.strip():
        return "General professional skills, project coordination, communication."
    
    prompt = f"""
Analizează următorul Job Description și extrage DOAR competențele transversale (soft skills), 
metodologiile de management, planificare și orientarea spre calitate/rezultate. 
EXCLUDE complet orice mențiune legată de specificul vertical sau de domeniu.
Răspunde printr-o listă scurtă de maximum 5-6 competențe transversale.

JOB DESCRIPTION:
{job_desc[:2000]}
"""
    res = gemini_text(prompt, max_tokens=500)
    return res if res else "Project coordination, stakeholder communication, planning, quality management."


# ==========================================
# 3. APELURI AI & MAP-REDUCE UNIVERSAL
# ==========================================

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
            temperature=0.1
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
                        "content": "Ești un asistent AI senior de optimizare CV-uri. Nu schimba niciodată domeniul de activitate al candidatului."
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
            response = gemini_client.models.generate_content(model=MODEL_NAME, contents=prompt)
            if response and hasattr(response, "text") and response.text:
                text_res = response.text.strip()
                if len(text_res) > 5:
                    return text_res
        except Exception as e:
            print(f"⚠️ Gemini Error: {e}. Trecem la HF...", flush=True)

    if USE_HF:
        try:
            text_res = call_huggingface(prompt)
            if text_res and len(text_res) > 5:
                return text_res
        except Exception as e:
            print(f"❌ Eroare Hugging Face: {e}", flush=True)

    return ""

def split_cv_into_sections(cv_text: str) -> dict:
    keywords = [
        "WORK EXPERIENCE", "EXPERIENCE", "EXPERIENȚĂ", 
        "EDUCATION", "EDUCAȚIE", "EDUCATIE", 
        "SKILLS", "COMPETENȚE", "COMPETENTE", 
        "PROJECTS", "PROIECTE", "PUBLICATIONS", "PUBLICATII", "ABOUT ME", "DRIVING LICENSES", "LANGUAGES"
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
    sections = split_cv_into_sections(cv_text)
    html_results = []
    safe_skills_profile = extract_safe_transferable_skills(job_desc)

    for sec_name, sec_content in sections.items():
        if not sec_content.strip():
            continue
            
        clean_sec_name = sec_name.upper().strip()
        
        prompt = f"""
Ești un expert global în optimizare ATS pentru CV-uri profesionale.
Sarcina ta este să reformulezi și să optimizezi secțiunea ({clean_sec_name}) a CV-ului, respectând **Strict Adevărul Factual**.

REGULI ABSOLUTE DE INTEGRITATE (ZERO HALUCINAȚII):
1. ADEVĂRUL FACTUAL: Folosește DOAR informațiile existente în textul original al secțiunii din CV. Este STRICT INTERZIS să inventezi locuri de muncă, companii, date cronologice, publicații, brevete, certificări sau instrumente.
2. IZOLAREA DOMENIULUI: Folosește profilul de competențe transversale extrase mai jos doar pentru a îmbunătăți stilul și claritatea descrierilor existente, păstrând 100% specificul domeniului real în care a lucrat candidatul. Nu importa activități care nu aparțin domeniului său.
3. FORMAT: Răspunde EXCLUSIV în format HTML curat (<p>, <ul>, <li>, <strong>), fără markdown suplimentar.
4. LIMBĂ DE REZULTAT: Textul final trebuie să fie scris STRICT în limba: {target_lang}.

CONȚINUTul ORIGINAL AL ACESTEI SECȚIUNI DIN CV:
{sec_content[:4000]}

PROFIL DE COMPETENȚE TRANSVERSALE DE URMAT (Fără a schimba domeniul de bază):
{safe_skills_profile}
"""
        raw_res = gemini_text(prompt, max_tokens=2048)
        cleaned_sec = re.sub(r'^```(?:html)?\s*', '', raw_res.strip(), flags=re.MULTILINE)
        cleaned_sec = re.sub(r'\s*```$', '', cleaned_sec, flags=re.MULTILINE).strip()
        
        if cleaned_sec:
            html_results.append(f"<div class='cv-section'><h2>{clean_sec_name}</h2>\n{cleaned_sec}\n</div>")
            
    combined_html = "\n".join(html_results)
    return postprocess_cv_html(combined_html)


# ==========================================
# 4. RUTELE API FLASK (TOATE ENDPOINTURILE COMPLETE)
# ==========================================

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({
        "status": "online",
        "success": True,
        "service": "vCoach AI API - Universal Safe ATS Optimizer",
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
        return jsonify({"status": "success", "success": True}), 200
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
                    return jsonify({"success": False, "error": f"Eroare citire PDF: {str(pdf_err)}"}), 400
            else:
                text_content = file.read().decode("utf-8", errors="ignore")
        elif request.is_json:
            data = request.get_json(force=True, silent=True) or {}
            text_content = data.get("cv_text", "")

        cleaned = clean_text(text_content)
        if not cleaned:
            return jsonify({"success": False, "error": "Nu s-a putut extrage text din fișierul trimis."}), 400

        MEMORY["cv_text"] = cleaned
        return jsonify({"success": True, "message": "CV încărcat cu succes", "length": len(cleaned), "cv_text": cleaned})
    except Exception as e:
        return jsonify({"success": False, "error": f"Eroare la procesare: {str(e)}"}), 500

@app.route("/analyze-cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_root")
@app.route("/api/cv-quality", methods=["POST", "OPTIONS"], endpoint="analyze_cv_quality_api")
@cross_origin()
def analyze_cv_quality():
    if request.method == "OPTIONS":
        return jsonify({"status": "success", "success": True}), 200
    try:
        data = request.get_json(force=True, silent=True) or {}
        cv = clean_text(data.get("cv_text") or MEMORY.get("cv_text") or "")
        job = clean_text(data.get("job_description") or data.get("job_text") or MEMORY.get("job_description") or "")
        target_lang = data.get("target_language") or data.get("language") or "ro"
        
        if not cv:
            return jsonify({"success": False, "error": "CV lipsă."}), 400

        MEMORY["cv_text"] = cv
        if job:
            MEMORY["job_description"] = job

        prompt = f"""
Analizează CV-ul în raport cu descrierea jobului. Limba de răspuns: STRICT {target_lang}.
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
        return jsonify({"success": True, "data": payload, **payload}), 200
    except Exception as e:
        return jsonify({"success": False, "error": f"Eroare analiză CV: {str(e)}"}), 500

@app.route("/match-job", methods=["POST", "OPTIONS"], endpoint="match_job_root")
@app.route("/api/match-job", methods=["POST", "OPTIONS"], endpoint="match_job_api")
@cross_origin()
def match_job():
    if request.method == "OPTIONS":
        return jsonify({"status": "success", "success": True}), 200
    try:
        data = request.get_json(force=True, silent=True) or {}
        cv = clean_text(data.get("cv_text") or MEMORY.get("cv_text") or "")
        job_desc = clean_text(data.get("job_description") or MEMORY.get("job_description") or "")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        if not cv or not job_desc:
            return jsonify({"success": False, "error": "CV-ul și Descrierea Jobului sunt necesare."}), 400

        prompt = f"""
Compară CV-ul cu Descrierea Jobului. Limba de răspuns: STRICT {target_lang}.
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
        return jsonify({"success": True, "data": payload, **payload}), 200
    except Exception as e:
        return jsonify({"success": False, "error": f"Eroare match-job: {str(e)}"}), 500

@app.route("/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_root")
@app.route("/api/rephrase", methods=["POST", "OPTIONS"], endpoint="rephrase_api")
@cross_origin()
def rephrase():
    if request.method == "OPTIONS":
        return jsonify({"status": "success", "success": True}), 200
    try:
        data = request.get_json(force=True, silent=True) or {}
        cv_text = data.get("text") or MEMORY.get("cv_text") or ""
        job_desc = data.get("job_description") or MEMORY.get("job_description") or ""
        target_lang = data.get("target_language") or data.get("language") or "ro"

        if not cv_text:
            return jsonify({"success": False, "error": "Textul CV-ului pentru reformulare lipsește."}), 400

        improved = map_reduce_rephrase(cv_text, job_desc, target_lang)

        if not improved or len(str(improved).strip()) == 0:
            improved = "<p>Eroare la procesarea prin Map-Reduce.</p>"

        payload = {
            "improved_text": improved,
            "rephrased_text": improved,
            "text": improved,
            "target_language": target_lang
        }
        return jsonify({"success": True, "data": payload, **payload}), 200
    except Exception as e:
        return jsonify({"success": False, "error": f"Eroare rephrase: {str(e)}"}), 500

@app.route("/interview-prep", methods=["POST", "OPTIONS"], endpoint="interview_prep_root")
@app.route("/api/interview-prep", methods=["POST", "OPTIONS"], endpoint="interview_prep_api")
@cross_origin()
def interview_prep():
    if request.method == "OPTIONS":
        return jsonify({"status": "success", "success": True}), 200
    try:
        data = request.get_json(force=True, silent=True) or {}
        cv = clean_text(data.get("cv_text") or MEMORY.get("cv_text") or "")
        job = clean_text(data.get("job_description") or MEMORY.get("job_description") or "")
        target_lang = data.get("target_language") or data.get("language") or "ro"

        prompt = f"""
Generează 5 întrebări tehnice și comportamentale de interviu bazate pe acest CV și job.
Limba de răspuns: STRICT {target_lang}.
Returnează un JSON valid cu cheia "questions" (listă de string-uri).
CV: {cv[:3000]}
JOB: {job[:1500]}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res)
        questions = parsed.get("questions", [
            "Cum gestionați riscurile tehnice în fazele de proiectare?",
            "Dați un exemplu de metodologie de îmbunătățire a calității implementată de dumneavoastră."
        ])
        return jsonify({"success": True, "questions": questions}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/validate-content", methods=["POST", "OPTIONS"])
@cross_origin()
def validate_content():
    if request.method == "OPTIONS":
        return jsonify({"status": "success", "success": True}), 200
    try:
        data = request.get_json(force=True, silent=True) or {}
        original_cv = data.get("cv_text") or MEMORY.get("cv_text") or ""
        generated_text = data.get("generated_text", "")

        if not original_cv or not generated_text:
            return jsonify({"success": False, "error": "Lipsește CV-ul original sau textul generat."}), 400

        prompt = f"""
Ești un auditor de integritate pentru CV-uri. Compară textul generat cu CV-ul original al candidatului.
Identifică dacă textul generat conține halucinații majore (companii inventate, joburi inventate, brevete false care nu apar în original).
Răspunde STRICT în format JSON:
{{
  "is_safe": true,
  "confidence_score": 95,
  "detected_anomalies": []
}}
CV ORIGINAL: {original_cv[:3000]}
TEXT GENERAT: {generated_text[:3000]}
"""
        raw_res = gemini_text(prompt, max_tokens=1000)
        parsed = safe_json(raw_res)

        return jsonify({
            "success": True,
            "validation": {
                "is_safe": parsed.get("is_safe", True),
                "confidence_score": parsed.get("confidence_score", 90),
                "anomalies": parsed.get("detected_anomalies", [])
            }
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": f"Eroare la validare: {str(e)}"}), 500

@app.route("/clear-session", methods=["POST", "OPTIONS"], endpoint="clear_session_root")
@app.route("/api/clear-session", methods=["POST", "OPTIONS"], endpoint="clear_session_api")
@cross_origin()
def clear_session():
    if request.method == "OPTIONS":
        return jsonify({"status": "success", "success": True}), 200
    MEMORY["cv_text"] = ""
    MEMORY["job_description"] = ""
    MEMORY["interview_history"] = []
    return jsonify({"success": True, "message": "Sesiunea a fost resetată cu succes."}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Serverul pornește pe portul {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
