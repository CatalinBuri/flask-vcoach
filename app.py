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
    """Extrage un obiect JSON din text, curățând delimitatorii de cod."""
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

# ✅ CONFIGURARE CORS globală (sigură pentru domeniile tale)
CORS(app,
     origins=[
         "https://www.pixelplayground3d.ro",
         "https://pixelplayground3d.ro"
     ],
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
     expose_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# ✅ Handler global pentru CORS — adăugăm antetele corecte la TOATE răspunsurile
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://www.pixelplayground3d.ro')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# ✅ Răspuns standard pentru cererile OPTIONS (preflight)
@app.route("/", methods=["OPTIONS"])
def handle_root_preflight():
    return jsonify({"message": "Preflight OK"}), 200


API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Lipsește GEMINI_API_KEY din .env!")

gemini_client = genai.Client(api_key=API_KEY)

# -----------------------
# /generate-questions
# -----------------------
@app.route("/generate-questions", methods=["POST", "OPTIONS"])
def generate_questions():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        cv_text = data.get("cv_text", "").strip()
        job_text = data.get("job_text", "").strip()
        if not cv_text or not job_text:
            return jsonify({"error": "CV-ul și Job Description sunt obligatorii."}), 400

        prompt = f"""
        Ești un recrutor AI specializat. Analizează CV-ul și Job Description-ul și generează:
        1. O sinteză scurtă a postului (în română).
        2. 5 întrebări personalizate de interviu (în română).
        Toată ieșirea trebuie să fie în format JSON.
        CV: {cv_text}
        JOB: {job_text}
        Format:
        {{
            "summary": "...",
            "questions": ["...", "...", "...", "...", "..."],
            "question_count": 5
        }}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare întrebări eșuată", "details": str(e)}), 500

# -----------------------
# /analyze-answer
# -----------------------
@app.route("/analyze-answer", methods=["POST", "OPTIONS"])
def analyze_answer():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        question = data.get("question")
        user_answer = data.get("user_answer")
        history = data.get("history", [])

        if not question or not user_answer:
            return jsonify({"error": "Întrebarea și răspunsul sunt obligatorii."}), 400

        history_text = json.dumps(history, ensure_ascii=False, indent=2)
        prompt = f"""
        Ești un recrutor AI. Analizează răspunsul candidatului.
        Întrebare: "{question}"
        Răspuns candidat: "{user_answer}"
        Istoric conversațional: {history_text}

        Răspuns strict JSON:
        {{
            "current_evaluation": {{
                "feedback": "Feedback detaliat în limba română (Markdown permis).",
                "nota_finala": 8,
                "claritate": 8,
                "relevanta": 8,
                "structura": 8
            }}
        }}
        """

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )

        ai_data = safe_json_extract(response.text)

        # ✅ Asigurăm structura exactă așteptată de frontend
        if "current_evaluation" not in ai_data:
            ai_data = {"current_evaluation": ai_data}

        return jsonify(ai_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Analiza răspunsului eșuată", "details": str(e)}), 500

# -----------------------
# /generate-report
# -----------------------
@app.route("/generate-report", methods=["POST", "OPTIONS"])
def generate_report():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        job_summary = data.get("job_summary")
        history = data.get("history")
        if not job_summary or not history:
            return jsonify({"error": "Sinteza jobului și istoricul sunt obligatorii."}), 400

        history_text = json.dumps(history, ensure_ascii=False, indent=2)
        prompt = f"""
        Pe baza sintezei jobului și a istoricului interviului, redactează raportul final în limba română:
        Sinteză job: {job_summary}
        Istoric: {history_text}
        Format JSON:
        {{
            "overall_report_markdown": "...",
            "compatibility_score": 75
        }}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare raport eșuat", "details": str(e)}), 500

# -----------------------
# /process-text
# -----------------------
@app.route("/process-text", methods=["POST", "OPTIONS"])
def process_text():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        job_text = data.get("text", "").strip()
        if not job_text:
            return jsonify({"error": "Job description is required."}), 400
        clean_text = re.sub(r"(?i)\b(bullet\s*icon)\b", "", job_text)
        clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()
        prompt = f"""
        Rezumă în limba română principalele responsabilități și competențe din textul următor:
        {clean_text}
        Răspuns: un text coerent și clar (fără bullet points).
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        summary = response.text.strip()
        return jsonify({"processed_text": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Eroare internă la procesarea textului", "details": str(e)}), 500

# -----------------------
# /generate-beginner-faq
# -----------------------
@app.route("/generate-beginner-faq", methods=["POST", "OPTIONS"])
def generate_beginner_faq():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        cv_text = data.get("cv_text", "").strip()
        prompt = f"""
        Generează 5 întrebări de tip FAQ pentru candidați (entry-level) bazate pe CV:
        {cv_text}
        Răspuns strict JSON:
        {{
            "faq": [
                {{"question": "Întrebare 1", "explanation": "..." }},
                {{"question": "Întrebare 2", "explanation": "..." }},
                {{"question": "Întrebare 3", "explanation": "..." }},
                {{"question": "Întrebare 4", "explanation": "..." }},
                {{"question": "Întrebare 5", "explanation": "..." }}
            ]
        }}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare FAQ eșuată", "details": str(e)}), 500

# -----------------------
# /analyze-faq-answers
# -----------------------
@app.route("/analyze-faq-answers", methods=["POST", "OPTIONS"])
def analyze_faq_answers():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        faq_data = data.get("faq_data", [])
        if not faq_data:
            return jsonify({"error": "faq_data obligatoriu"}), 400
        item = faq_data[0]
        prompt = f"""
        Evaluează răspunsul candidatului: "{item.get('user_answer','')}"
        La întrebarea: "{item.get('question','')}"
        Context: "{item.get('explanation','')}"
        Răspuns strict JSON:
        {{
            "feedback": "...",
            "nota_finala": 8,
            "claritate": 8,
            "relevanta": 8,
            "structura": 8
        }}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify({"analysis_results": [{"evaluation": ai_data}]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Analiză FAQ eșuată", "details": str(e)}), 500

# -----------------------
# /generate-cover-letter
# -----------------------
@app.route("/generate-cover-letter", methods=["POST", "OPTIONS"])
def generate_cover_letter():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        cv_text = data.get("cv_text", "")
        job_text = data.get("job_text", "")
        if not cv_text or not job_text:
            return jsonify({"error": "cv_text și job_text sunt obligatorii"}), 400
        prompt = f"""
        Generează o scrisoare de intenție profesională în română, concisă și personalizată, fara a include confirmari de la modelul AI:
        CV: {cv_text}
        Job: {job_text}
        Format JSON:
        {{
            "cover_letter": "..."
        }}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare Cover Letter eșuată", "details": str(e)}), 500

# -----------------------
# /generate-linkedin-summary
# -----------------------
@app.route("/generate-linkedin-summary", methods=["POST", "OPTIONS"])
def generate_linkedin_summary():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        cv_text = data.get("cv_text", "")
        if not cv_text:
            return jsonify({"error": "cv_text obligatoriu"}), 400
        prompt = f"""
        Creează un sumar profesional LinkedIn bazat pe următorul CV:
        {cv_text}
        Format JSON:
        {{
            "linkedin_summary": "..."
        }}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare LinkedIn Summary eșuată", "details": str(e)}), 500

# -----------------------
# /generate-job-hunt-optimization
# -----------------------
@app.route("/generate-job-hunt-optimization", methods=["POST", "OPTIONS"])
def generate_job_hunt_optimization():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        cv_text = data.get("cv_text", "")
        job_text = data.get("job_text", "")
        prompt = f"""
        Analizează CV-ul și Job Description-ul și oferă sugestii concrete pentru îmbunătățirea șanselor de angajare.
        CV: {cv_text}
        Job: {job_text}
        Format JSON:
        {{
            "optimization_tips": [
                "Sugestie 1",
                "Sugestie 2",
                "Sugestie 3"
            ]
        }}
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare optimizare Job Hunt eșuată", "details": str(e)}), 500

# -----------------------
# PORNIRE SERVER
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

