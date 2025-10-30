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
CORS(app, resources={r"/*": {"origins": "https://www.pixelplayground3d.ro"}}, supports_credentials=True)

# --------------------------
# Logging minimal pentru debugging
@app.before_request
def log_request():
    print(f"[{request.method}] {request.path} | from: {request.headers.get('Origin', 'local')}")

# --------------------------
# Inițializare client Gemini
try:
    if not API_KEY:
        print("❌ EROARE: GEMINI_API_KEY lipsește!")
        gemini_client = None
    else:
        gemini_client = genai.Client(api_key=API_KEY)
        print("✅ Conexiune Gemini inițializată corect.")
except Exception as e:
    print(f"❌ Eroare la inițializarea Gemini: {e}")
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
# ROUTE: Procesare descriere job
@app.route('/process-text', methods=['POST'])
def process_text():
    if not gemini_client:
        return jsonify({"error": "Serviciul AI nu este disponibil. Verificați cheia API."}), 503

    data = request.get_json()
    job_text = data.get('text', '').strip()

    if not job_text:
        return jsonify({"error": "Descrierea postului (text) este obligatorie."}), 400

    prompt = (
        f"Analizează această descriere de job: '{job_text}'. "
        "Extrage informațiile cheie (rol, cerințe, responsabilități) și oferă un rezumat "
        "scurt și clar, de maxim 3-4 paragrafe."
    )

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        processed_text = response.text
        return jsonify({"processed_text": processed_text}), 200
    except Exception as e:
        print(f"Eroare la procesarea textului cu Gemini: {e}")
        return jsonify({"error": "Eroare la procesarea textului cu AI."}), 500


# --------------------------
# ROUTE: Generare întrebări interviu
@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    if not gemini_client:
        return jsonify({"error": "Serviciul AI nu este disponibil."}), 503

    data = request.get_json()
    processed_text = data.get('processed_text', '')

    prompt = (
        f"Pe baza acestui rezumat al postului: {processed_text}, "
        "generează 5 întrebări de interviu comportamentale. "
        "Returnează JSON strict: {'questions': ['Întrebarea 1?', 'Întrebarea 2?', ...]}."
    )

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        questions_data = safe_json_extract(response.text)
        return jsonify(questions_data), 200
    except Exception as e:
        print(f"Eroare la generarea întrebărilor cu Gemini: {e}")
        return jsonify({"error": "Eroare la generarea întrebărilor."}), 500


# --------------------------
# ROUTE: Analiză CV vs Job
@app.route('/analyze-cv', methods=['POST'])
def analyze_cv():
    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()
    job_text = data.get('job_text', '').strip()

    if not cv_text or not job_text:
        return jsonify({"error": "CV și Job Description sunt necesare."}), 400

    prompt = f"""
    Evaluează compatibilitatea CV-ului cu Job-ul:
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
# ROUTE: Generare interogări job hunt
@app.route('/generate-job-queries', methods=['POST'])
def generate_job_queries():
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
@app.route('/generate-cover-letter', methods=['POST'])
def generate_cover_letter():
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
# ROUTE: Optimizare profil LinkedIn
@app.route('/optimize-linkedin-profile', methods=['POST'])
def optimize_linkedin_profile():
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
@app.route('/generate-beginner-faq', methods=['POST'])
def generate_beginner_faq():
    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    cv_text = request.get_json().get('cv_text', '').strip()
    prompt_context = f"Generează 5 întrebări FAQ pentru începători bazate pe CV:\n{cv_text}" if cv_text else "Generează 5 întrebări FAQ standard pentru entry-level."

    prompt = f"""
    Ești un recrutor AI. {prompt_context}
    Returnează DOAR JSON cu cheia "questions", fiecare obiect având:
    {{
      "question": "Întrebarea X?",
      "explanation": "Scurtă explicație în Markdown"
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
# ROUTE: Evaluare răspuns utilizator
@app.route('/evaluate-answer', methods=['POST'])
def evaluate_answer():
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
# ROUTE: Generare raport final
@app.route('/generate-report', methods=['POST'])
def generate_report():
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
      "key_strengths": ["3 puncte forte"],
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

@app.route('/coach-results-html', methods=['POST'])
def coach_results_html():
    if gemini_client is None:
        return "<h3>AI indisponibil</h3>", 503

    data = request.get_json()
    history = data.get('history', [])

    if not history:
        return "<h3>Nu există răspunsuri de procesat</h3>", 400

    html_content = """
    <html lang='ro'>
    <head>
        <meta charset='UTF-8'>
        <title>Coach Feedback STAR</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f7f6; padding: 20px; color: #2c3e50; }
            h1 { text-align: center; color: #2980b9; }
            .entry { background: #fff; padding: 15px; margin: 15px 0; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.08); }
            .question { font-weight: bold; color: #34495e; }
            .user-answer, .star-answer { margin-top: 10px; padding: 10px; border-radius: 6px; background: #ecf0f1; white-space: pre-wrap; }
            .star-answer { border-left: 5px solid #2ecc71; background: #e8f6ef; }
        </style>
    </head>
    <body>
        <h1>Rezultate Coach - Versiune STAR</h1>
    """

    for idx, entry in enumerate(history):
        question = entry.get('question', 'Întrebare lipsă')
        user_answer = entry.get('answer', 'Răspuns lipsă')

        # Generare răspuns STAR prin AI
        prompt = f"""
        Întrebarea: {question}
        Răspunsul utilizatorului: {user_answer}
        Te rog să rescrii acest răspuns într-o versiune optimizată STAR (Situation, Task, Action, Result).
        Returnează DOAR textul răspunsului optimizat.
        """
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            star_answer = response.text.strip()
        except Exception as e:
            star_answer = f"Eroare la generarea STAR: {str(e)}"

        html_content += f"""
        <div class='entry'>
            <div class='question'>Întrebarea {idx+1}: {question}</div>
            <div class='user-answer'><strong>Răspunsul tău:</strong>\n{user_answer}</div>
            <div class='star-answer'><strong>Răspuns STAR optimizat:</strong>\n{star_answer}</div>
        </div>
        """

    html_content += "</body></html>"
    return html_content, 200


# --------------------------
# PORNIRE SERVER
if __name__ == '__main__':
    print("🚀 Server Flask pornit pe http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)


