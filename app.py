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
# 2. FUNCȚII DE POSTPROCESARE & FILTRE DE SECURITATE
# ==========================================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def postprocess_cv_html(raw_html: str) -> str:
    """
    Postprocesează HTML-ul generat pentru a elimina dublurile de titluri,
    halucinațiile nedorite și tag-urile redundante.
    """
    if not raw_html:
        return ""

    processed = raw_html

    # 1. Eliminarea halucinațiilor frecvente legate de "market research" sau "cercetare de piață"
    market_research_patterns = [
        r'<p>[^<]*?(?:market research|cercetare de piață|online research projects)[^<]*?</p>',
        r'<li>[^<]*?(?:market research|cercetare de piață|online research projects)[^<]*?</li>'
    ]
    for pattern in market_research_patterns:
        processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)

    # 2. Curățarea titlurilor duplicate sau a tag-urilor de titlu imbricate
    def clean_duplicate_headers(match):
        header_tag = match.group(1)
        header_text = match.group(2).strip()
        return f"<{header_tag}>{header_text}</{header_tag}>"

    processed = re.sub(r'(<h[23]>)(.*?)(</h[23]>)(?:\s*<h[23]>.*?</h[23]>)+', clean_duplicate_headers, processed, flags=re.IGNORECASE)

    # 3. Eliminarea liniilor goale sau a tag-urilor HTML fără conținut
    processed = re.sub(r'<p>\s*</p>', '', processed)
    processed = re.sub(r'<div>\s*</div>', '', processed)
    processed = re.sub(r'\n{3,}', '\n\n', processed)
    
    return processed.strip()

def validate_and_fix_translations(text: str, target_lang: str) -> str:
    """
    Verifică textul tradus/optimizat și corectează termenii tehnici uzați greșit
    sau amestecați cu alte domenii (asigură acuratețea terminologică).
    """
    if not text:
        return ""

    corrected_text = text

    automotive_glossary_ro = {
        r'\bcercetare de piață\b': 'cercetare de dezvoltare tehnică',
        r'\bdate de cercetare\b': 'date de inginerie / calibrare'
    }

    automotive_glossary_en = {
        r'\bmarket research\b': 'automotive engineering development',
        r'\bonline research projects\b': 'automotive engineering projects',
        r'\bsurvey data\b': 'calibration data'
    }

    glossary = automotive_glossary_en if target_lang.lower() in ['en', 'english'] else automotive_glossary_ro

    for wrong_pattern, correct_term in glossary.items():
        corrected_text = re.sub(wrong_pattern, correct_term, corrected_text, flags=re.IGNORECASE)

    return corrected_text


# ==========================================
# 3. APELURI AI & MAP-REDUCE CORECTAT (Regula Adevărului Tehnic)
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
    sections = split_cv_into_sections(cv_text)
    html_results = []

    for sec_name, sec_content in sections.items():
        if not sec_content.strip():
            continue
            
        clean_sec_name = sec_name.upper().strip()
        
        prompt = f"""
Ești un expert tehnic în resurse umane pentru industria AUTOMOTIVE și inginerie software/hardware (ECU, CATIA, SDV). 
Optimizează strict această secțiune ({clean_sec_name}) a CV-ului unui Engineering Manager real. 

REGULI CRITICE DE SEPARARE ȘI INTEGRITATE (ANTI-CONTAMINARE):
1. REGULA ADEVĂRULUI TEHNIC: Folosește descrierea de job de mai jos DOAR pentru a prelua cuvinte-cheie tehnice și stilul de exprimare. Este STRICT INTERZIS să introduci în CV activități de "Market Research", "cercetare de piață" sau "client service non-tehnic", dacă ele nu există deja în experiența reală de Automotive Engineering a candidatului.
2. DOMENIU STRICT: Candidatul lucrează în Automotive (Renault/Horse, ECU, SDV, SFS, Management de Proiect Tehnic, Brevete oficiale). Păstrează exclusiv acest domeniu.
3. FĂRĂ INVENȚII: Nu inventa publicații sau conferințe. Păstrează brevetele reale (patents).
4. STRUCTURĂ: Nu duplica titlurile în interiorul conținutului.
5. FORMAT: Răspunde EXCLUSIV în format HTML curat (<p>, <ul>, <li>, <strong>), FĂRĂ blocuri de cod markdown.
6. LIMBĂ: STRICT {target_lang}.

CONȚINUTul ORIGINAL AL ACESTEI SECȚIUNI DIN CV:
{sec_content[:4000]}

DESCRIERE JOB DE REFERINȚĂ (Folosește DOAR pentru alinierea termenilor de management/tehnici, FĂ A PRELUA DOMENIUL DE MARKET RESEARCH):
{job_desc[:2000]}
"""
        raw_res = gemini_text(prompt, max_tokens=2048)
        cleaned_sec = re.sub(r'^```(?:html)?\s*', '', raw_res.strip(), flags=re.MULTILINE)
        cleaned_sec = re.sub(r'\s*```$', '', cleaned_sec, flags=re.MULTILINE).strip()
        
        if cleaned_sec:
            html_results.append(f"<div class='cv-section'><h2>{clean_sec_name}</h2>\n{cleaned_sec}\n</div>")
            
    combined_html = "\n".join(html_results)
    
    # Aplicare filtre postprocesare completă
    final_clean_html = postprocess_cv_html(combined_html)
    final_clean_html = validate_and_fix_translations(final_clean_html, target_lang)
    
    return final_clean_html


# ==========================================
# 4. RUTELE API FLASK
# ==========================================

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({
        "status": "online",
        "success": True,
        "service": "vCoach AI API (Map-Reduce, Postprocessing & Safe ATS Optimization)",
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
            "text": improved
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
Generează 5 întrebări tehnice și comportamentale de interviu pentru un Engineering Manager în Automotive bazat pe acest CV și job.
Limba de răspuns: STRICT {target_lang}.
Returnează un JSON valid cu cheia "questions" (listă de string-uri).
CV: {cv[:3000]}
JOB: {job[:1500]}
"""
        raw_res = gemini_text(prompt)
        parsed = safe_json(raw_res)
        questions = parsed.get("questions", [
            "Cum gestionați riscurile tehnice în fazele de proiectare CATIA și prototipare?",
            "Dați un exemplu de metodologie de îmbunătățire a calității implementată de dumneavoastră."
        ])
        return jsonify({"success": True, "questions": questions}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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
