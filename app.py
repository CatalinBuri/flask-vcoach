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
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Lipsește GEMINI_API_KEY din .env!")

gemini_client = genai.Client(api_key=API_KEY)

# -----------------------
# ENDPOINTURI
# -----------------------

@app.route("/generate-questions", methods=["POST"])
def generate_questions():
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
            model="gemini-2.5-flash",
            contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare întrebări eșuată", "details": str(e)}), 500

@app.route("/analyze-answer", methods=["POST"])
def analyze_answer():
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
            model="gemini-2.5-flash",
            contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        if not ai_data.get("current_evaluation"):
            raise ValueError("Structură JSON neconformă (lipsă current_evaluation).")
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Analiza răspunsului eșuată", "details": str(e)}), 500

@app.route("/generate-report", methods=["POST"])
def generate_report():
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
            model="gemini-2.5-flash",
            contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare raport eșuat", "details": str(e)}), 500

@app.route("/process-text", methods=["POST"])
def process_text():
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
        Răspuns: un text coerent și clar (fără bullet points, fără mențiuni de tip 'bullet icon').
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        summary = response.text.strip()
        return jsonify({"processed_text": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Eroare internă la procesarea textului", "details": str(e)}), 500

# --- Alte endpointuri similare: /generate-beginner-faq, /analyze-faq-answers, /generate-cover-letter, /generate-linkedin-summary, /generate-job-hunt-optimization
# Le poți adăuga la fel cum am scris mai sus, toate folosind safe_json_extract și gemini_client

# -----------------------
# PORNIRE SERVER
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
