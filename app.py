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
# ROUTE: Generare FAQ
@app.route('/generate-questions', methods=['POST', 'OPTIONS'])
def generate_questions():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()

    if not cv_text:
        prompt_context = "Generează 5 întrebări standard pentru entry-level, fără CV."
    else:
        prompt_context = f"Generează 5 întrebări specifice bazate pe CV-ul următor:\n{cv_text}"

    prompt = f"""
    Ești un recrutor AI. Generează 5 întrebări FAQ pentru interviu.
    {prompt_context}
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
        if "questions" not in result:
            raise ValueError("Răspuns AI nu conține 'questions'.")
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Generare FAQ eșuată", "details": str(e)}), 500

# --------------------------
# ROUTE: Analiză răspuns FAQ
@app.route('/analyze-answer', methods=['POST', 'OPTIONS'])
def analyze_answer():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    try:
        data = request.get_json()
        question = data.get('question')
        user_answer = data.get('user_answer')
        history = data.get('history', [])

        prompt = f"""
        Analizează răspunsul utilizatorului la interviu:
        Întrebare: {question}
        Răspuns: {user_answer}
        Folosește context din istoricul trecut, dacă există.
        Oferă evaluare clară pe o scară 1-10 pentru claritate, relevanță și structură.
        Returnează JSON strict cu cheia 'current_evaluation':
        {{
          "nota_finala": 8,
          "claritate": 8,
          "relevanta": 7,
          "structura": 8,
          "feedback": "Feedback detaliat în Markdown"
        }}
        """

        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        result = safe_json_extract(response.text)
        if "current_evaluation" not in result:
            raise ValueError("Răspuns AI nu conține 'current_evaluation'")
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": "Analiză răspuns eșuată", "details": str(e)}), 500

# --------------------------
# ROUTE: Raport final
@app.route('/generate-report', methods=['POST', 'OPTIONS'])
def generate_report():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if gemini_client is None:
        return jsonify({"error": "AI indisponibil"}), 503

    try:
        data = request.get_json()
        faq_history = data.get('history', [])
        job_summary = data.get('job_summary', '')
        cv_text = data.get('cv_text', '')

        if not faq_history:
            return jsonify({"error": "Istoricul FAQ este gol"}), 400

        history_text = ""
        for idx, entry in enumerate(faq_history):
            q = entry.get('question_data', {}).get('question', 'N/A')
            a = entry.get('user_answer', 'N/A')
            note = entry.get('analysis', {}).get('evaluation', {}).get('nota_finala', 'N/A')
            feedback = entry.get('analysis', {}).get('evaluation', {}).get('feedback', 'N/A')
            history_text += f"--- Întrebarea {idx+1} (Nota: {note}/10) ---\nQ: {q}\nA: {a}\nFeedback: {feedback}\n\n"

        prompt = f"""
        Ești un Career Coach AI. Folosește istoricul FAQ pentru a genera un raport final:
        FORMAT JSON STRICT:
        {{
          "final_score": "medie din cele 5 scoruri",
          "summary": "Sinteză generală în Markdown",
          "key_strengths": ["3 puncte forte cheie"],
          "areas_for_improvement": ["3 arii de îmbunătățire"],
          "next_steps_recommendation": "Recomandări pentru următori pași"
        }}
        ISTORIC FAQ:\n{history_text}
        JOB SUMMARY:\n{job_summary}
        CV TEXT:\n{cv_text}
        """

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
