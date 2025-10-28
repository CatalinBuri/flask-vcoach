# server_vcoach.py

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

# --------------------------
# Încarcă variabilele de mediu (.env)
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

# --------------------------
# Inițializare Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --------------------------
# Client Gemini
try:
    if not API_KEY:
        print("EROARE: GEMINI_API_KEY lipsește!")
        gemini_client = None
    else:
        gemini_client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Eroare la inițializarea Gemini: {e}")
    gemini_client = None

# --------------------------
# UTILITĂȚI
def safe_json_extract(text):
    if not text:
        raise ValueError("Text gol primit de la AI.")
    full_text = text.strip()
    if full_text.startswith('```json'):
        full_text = full_text.replace('```json', '', 1).strip()
    if full_text.endswith('```'):
        full_text = full_text[:-3].strip()
    try:
        return json.loads(full_text)
    except json.JSONDecodeError:
        try:
            start_index = full_text.index('{')
            end_index = full_text.rindex('}') + 1
            return json.loads(full_text[start_index:end_index])
        except Exception as e:
            raise ValueError(f"Eroare la extragerea JSON: {e}. Text: {full_text[:500]}...")

# --------------------------
# ROUTE: Analiza CV
@app.route('/analyze-cv', methods=['POST', 'OPTIONS'])
def analyze_cv():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()
    job_text = data.get('job_text', '').strip()

    if not cv_text or not job_text:
        return jsonify({"error": "CV și Job Description sunt necesare."}), 400

    prompt = f"""
    Evaluează compatibilitatea CV-ului cu Job-ul următor:
    CV: {cv_text}
    Job Description: {job_text}
    Returnează JSON strict cu:
    {{
      "compatibility_percent": 0-100,
      "feedback_markdown": "Feedback detaliat în Markdown"
    }}
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        result = safe_json_extract(response.text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Analiză CV eșuată", "details": str(e)}), 500

# --------------------------
# ROUTE: Generare Job Queries
@app.route('/generate-job-queries', methods=['POST', 'OPTIONS'])
def generate_job_queries():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    cv_text = request.get_json().get('cv_text', '').strip()
    if not cv_text:
        return jsonify({"error": "CV este necesar."}), 400

    prompt = f"""
    Generează 5-10 interogări optimizate pentru job hunt bazate pe acest CV:
    {cv_text}
    Returnează JSON cu cheia 'queries', fiecare element fiind o interogare text.
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        result = safe_json_extract(response.text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Generare Job Queries eșuată", "details": str(e)}), 500

# --------------------------
# ROUTE: Generare Cover Letter
@app.route('/generate-cover-letter', methods=['POST', 'OPTIONS'])
def generate_cover_letter():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    data = request.get_json()
    cv_text = data.get('cv_text', '')
    job_summary = data.get('job_summary', '')

    prompt = f"""
    Generează o scrisoare de intenție profesionistă:
    CV: {cv_text}
    Job Summary: {job_summary}
    Returnează JSON cu cheia 'cover_letter' și textul scrisorii.
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        result = safe_json_extract(response.text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Generare Cover Letter eșuată", "details": str(e)}), 500

# --------------------------
# ROUTE: Optimize LinkedIn Profile
@app.route('/optimize-linkedin-profile', methods=['POST', 'OPTIONS'])
def optimize_linkedin_profile():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    cv_text = request.get_json().get('cv_text', '')
    prompt = f"""
    Oferă recomandări detaliate pentru optimizarea profilului LinkedIn bazat pe acest CV:
    {cv_text}
    Returnează JSON cu cheia 'linkedin_tips', o listă de sugestii.
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        result = safe_json_extract(response.text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Optimizare LinkedIn eșuată", "details": str(e)}), 500

# --------------------------
# ROUTE: Generare Beginner FAQ
@app.route('/generate-beginner-faq', methods=['POST', 'OPTIONS'])
def generate_beginner_faq():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    cv_text = request.get_json().get('cv_text', '').strip()
    prompt_context = f"Generează 5 întrebări FAQ pentru începători bazate pe CV:\n{cv_text}" if cv_text else "Generează 5 întrebări FAQ standard pentru entry-level."

    prompt = f"""
    Ești un recrutor AI. {prompt_context}
    Returnează DOAR JSON cu cheia "questions", fiecare obiect având:
    {{
      "question": "Întrebarea X?",
      "explanation": "Scurtă explicație a intenției recrutorului (Markdown, max 5 propoziții)"
    }}
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        result = safe_json_extract(response.text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Generare Beginner FAQ eșuată", "details": str(e)}), 500

# --------------------------
# ROUTE: Evaluate Answer (Comparative)
@app.route('/evaluate-answer', methods=['POST', 'OPTIONS'])
def evaluate_answer():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    data = request.get_json()
    question = data.get('question')
    user_answer = data.get('answer')
    previous_answer = data.get('previous_answer')
    previous_evaluation = data.get('previous_evaluation')

    prompt = f"""
    Analizează răspunsul utilizatorului:
    Întrebare: {question}
    Răspuns: {user_answer}
    Istoric răspuns anterior: {previous_answer}
    Evaluare anterioară: {previous_evaluation}
    Returnează JSON strict cu:
    {{
      "current_evaluation": {{
          "nota_finala": 0-10,
          "claritate": 0-10,
          "relevanta": 0-10,
          "structura": 0-10,
          "feedback": "Feedback detaliat în Markdown"
      }},
      "comparative_feedback": "Feedback comparativ cu răspunsul anterior"
    }}
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        result = safe_json_extract(response.text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Evaluare răspuns eșuată", "details": str(e)}), 500

# --------------------------
# ROUTE: Generate Final Report
@app.route('/generate-report', methods=['POST', 'OPTIONS'])
def generate_report():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    data = request.get_json()
    faq_history = data.get('history', [])
    job_summary = data.get('summary', '')
    cv_text = data.get('cv_text', '')

    if not faq_history:
        return jsonify({"error": "Istoricul FAQ este gol"}), 400

    history_text = ""
    for idx, entry in enumerate(faq_history):
        q = entry.get('question', 'N/A')
        a = entry.get('answer', 'N/A')
        note = entry.get('evaluation', {}).get('nota_finala', 'N/A')
        feedback = entry.get('evaluation', {}).get('feedback', 'N/A')
        history_text += f"--- Întrebarea {idx+1} (Nota: {note}/10) ---\nQ: {q}\nA: {a}\nFeedback: {feedback}\n\n"

    prompt = f"""
    Ești un Career Coach AI. Folosește istoricul FAQ pentru a genera un raport final.
    FORMAT JSON STRICT:
    {{
      "final_score": "medie din scoruri",
      "summary": "Sinteză generală în Markdown",
      "key_strengths": ["3 puncte forte cheie"],
      "areas_for_improvement": ["3 arii de îmbunătățire"],
      "next_steps_recommendation": "Recomandări pentru următorii pași"
    }}
    ISTORIC FAQ:\n{history_text}
    JOB SUMMARY:\n{job_summary}
    CV TEXT:\n{cv_text}
    """
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        result = safe_json_extract(response.text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Generare raport final eșuată", "details": str(e)}), 500

# --------------------------
# PORNIRE SERVER
if __name__ == '__main__':
    print("Server Flask pornit pe http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)
