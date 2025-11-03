# server_vcoach_robust.py
import os
import json
import traceback
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
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------
# Logging minimal pentru debugging
@app.before_request
def log_request():
    print(f"[{request.method}] {request.path} | from: {request.headers.get('Origin', 'local')}")

# --------------------------
# Inițializare client Gemini
gemini_client = None
try:
    if not API_KEY:
        print("❌ EROARE: GEMINI_API_KEY lipsește!")
    else:
        gemini_client = genai.Client(api_key=API_KEY)
        print("✅ Conexiune Gemini inițializată corect.")
except Exception as e:
    print(f"❌ Eroare la inițializarea Gemini: {e}")

# --------------------------
# UTILITĂȚI

# 1. Funcția îmbunătățită pentru extracția JSON (înlocuiește vechiul safe_json_extract)
def safe_json_extract(text):
    if not text:
        raise ValueError("Text gol primit pentru extracția JSON.")
    full_text = text.strip()
    
    # 1. Elimină ```json și ```
    if full_text.startswith('```json'):
        full_text = full_text.replace('```json', '', 1).strip()
    if full_text.endswith('```'):
        full_text = full_text[:-3].strip()
        
    try:
        # 2. Încearcă direct
        return json.loads(full_text)
    except json.JSONDecodeError as e_loads:
        # 3. Încearcă să găsească {...}
        try:
            # Găsește primul '{' și ultimul '}'
            start_index = full_text.index('{')
            end_index = full_text.rindex('}') + 1
            return json.loads(full_text[start_index:end_index])
        except Exception as e_extract:
            # Eroarea finală (include detalii mai bune)
            raise ValueError(f"Eroare la extragerea JSON: {e_extract} (Origine: {e_loads}). Text: {full_text[:500]}...")

# 2. Funcție pentru a obține textul brut de la AI 
def call_gemini_raw(prompt):
    if gemini_client is None:
        return {"error": "Eroare de configurare server", "details": "Clientul AI nu a putut fi inițializat (API Key lipsă/invalidă)."}
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
            # Eliminat: timeout=25
        )
        # Returnează textul brut
        return response.text
    # Aceste exceptii sunt acum definite datorita Pasului 1
    except DeadlineExceededError as e:
        return {"error": "Eroare de comunicare AI (Timeout)", "details": "Serviciul AI a depășit timpul maxim de răspuns (25s). Încercați din nou.", "code": 504}
    except APIError as e:
        # Gestionează alte erori API
        return {"error": "Eroare API Gemini", "details": str(e), "code": 500}
    except Exception as e:
        # Eroare de Rețea sau altceva.
        return {"error": "Eroare de comunicare AI (Necunoscută)", "details": str(e), "code": 500}
# 3. Funcție pentru a obține JSON 
def call_gemini_json(prompt):
    raw_text = call_gemini_raw(prompt)
    
    # Verifică dacă raw_text a returnat o eroare de configurare/comunicare
    if isinstance(raw_text, dict) and "error" in raw_text:
        return raw_text 
    
    try:
        # Încearcă să extragă JSON din textul brut
        return safe_json_extract(raw_text)
    except ValueError as e:
        # Eroare de extracție JSON
        return {"error": "Eroare la extragerea JSON", "details": str(e), "raw_text_received": raw_text[:500]}

# --------------------------
# ROUTE: Procesare descriere job (RAW)
@app.route('/process-text', methods=['POST'])
def process_text():
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
        # 🎯 FOLOSIM: call_gemini_raw
        raw_result = call_gemini_raw(prompt) 
        
        if isinstance(raw_result, dict) and "error" in raw_result:
            return jsonify(raw_result), 500

        # Returnăm textul învelit în JSON
        return jsonify({"processed_text": raw_result}), 200 
    
    except Exception as e:
        traceback.print_exc()
        print("❌ Eroare gravă în /process-text:", str(e))
        return jsonify({"error": "Eroare internă neprevăzută", "details": str(e)}), 500

# --------------------------
# ROUTE: Generare întrebări interviu (JSON)
@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    job_summary = data.get('job_summary', '')
    prompt = (
        f"Ești un recrutor AI. Pe baza acestui rezumat al postului: {job_summary} și CV: {cv_text}, "
        "generează 5 întrebări de interviu comportamentale unice, relevante și de nivel avansat. "
        "Returnează JSON strict: {'questions': [{'question': 'Întrebarea 1?'}, {'question': 'Întrebarea 2?'}, ...]}."
    )
    # 🎯 FOLOSIM: call_gemini_json
    result = call_gemini_json(prompt)
    return jsonify(result), 200 if "error" not in result else 500

# --------------------------
# ROUTE: Analiză CV vs Job (JSON)
@app.route('/analyze-cv', methods=['POST'])
def analyze_cv():
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
    # 🎯 FOLOSIM: call_gemini_json
    result = call_gemini_json(prompt)
    return jsonify(result), 200 if "error" not in result else 500

# --------------------------
# ROUTE: Generare interogări job hunt (JSON)
@app.route('/generate-job-queries', methods=['POST'])
def generate_job_queries():
    cv_text = request.get_json().get('cv_text', '').strip()
    if not cv_text:
        return jsonify({"error": "CV este necesar."}), 400
    prompt = f"""
    Generează 5-10 interogări optimizate pentru job hunt bazate pe acest CV:
    {cv_text}
    Returnează JSON cu cheia 'queries', fiecare element fiind o interogare text.
    """
    # 🎯 FOLOSIM: call_gemini_json
    result = call_gemini_json(prompt)
    return jsonify(result), 200 if "error" not in result else 500

# --------------------------
# ROUTE: Generare Cover Letter (JSON)
@app.route('/generate-cover-letter', methods=['POST'])
def generate_cover_letter():
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    job_summary = data.get('job_summary', '')
    prompt = f"""
    Generează o scrisoare de intenție profesionistă:
    CV: {cv_text}
    Job Summary: {job_summary}
    Returnează JSON cu cheia 'cover_letter' și textul scrisorii.
    """
    # 🎯 FOLOSIM: call_gemini_json
    result = call_gemini_json(prompt)
    return jsonify(result), 200 if "error" not in result else 500

# --------------------------
# ROUTE: Optimizare profil LinkedIn (JSON)
@app.route('/optimize-linkedin-profile', methods=['POST'])
def optimize_linkedin_profile():
    cv_text = request.get_json().get('cv_text', '')
    prompt = f"""
    Oferă recomandări detaliate pentru optimizarea profilului LinkedIn bazat pe acest CV:
    {cv_text}
    Returnează JSON cu cheia 'linkedin_tips', o listă de sugestii.
    """
    # 🎯 FOLOSIM: call_gemini_json
    result = call_gemini_json(prompt)
    return jsonify(result), 200 if "error" not in result else 500

# --------------------------
# ROUTE: Generare Beginner FAQ (JSON)
@app.route('/generate-beginner-faq', methods=['POST'])
def generate_beginner_faq():
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
    # 🎯 FOLOSIM: call_gemini_json
    result = call_gemini_json(prompt)
    return jsonify(result), 200 if "error" not in result else 500

# --------------------------
# ROUTE: Evaluare răspuns utilizator (JSON)
@app.route('/evaluate-answer', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    question = data.get('question')
    user_answer = data.get('answer')
    history = data.get('history', []) # Luăm tot istoricul
    
    # Pregătire context din istoric (opțional, dacă AI-ul îl folosește)
    history_text = "\n".join([f"Q: {h.get('question')}\nA: {h.get('answer')}\n" for h in history])

    # Setează promptul pentru AI
    prompt = f"""
    Evaluează răspunsul utilizatorului la următoarea întrebare.
    CONTEXT INTERVIU (Istoric):
    {history_text}
    
    Întrebare curentă: {question}
    Răspuns utilizator: {user_answer}
    
    Returnează JSON strict cu:
    {{
      "current_evaluation": {{"nota_finala": 0-10,"claritate": 0-10,"relevanta": 0-10,"structura": 0-10,"feedback": "Feedback detaliat în Markdown"}},
      "comparative_feedback": {{"feedback": "Feedback evolutiv, bazat pe istoric (dacă există) în Markdown."}}
    }}
    """
    # 🎯 FOLOSIM: call_gemini_json
    result = call_gemini_json(prompt)
    return jsonify(result), 200 if "error" not in result else 500

# --------------------------
# ROUTE: Generare raport final (JSON)
@app.route('/generate-report', methods=['POST'])
def generate_report():
    data = request.get_json()
    faq_history = data.get('history', [])
    job_summary = data.get('job_summary', '')
    cv_text = data.get('cv_text', '')
    if not faq_history:
        return jsonify({"error": "Istoricul interviului este gol"}), 400

    history_text = ""
    for idx, entry in enumerate(faq_history):
        q = entry.get('question', 'N/A')
        a = entry.get('answer', 'N/A')
        eval_dict = entry.get('evaluation', {})
        note = eval_dict.get('nota_finala', 'N/A')
        feedback = eval_dict.get('feedback', 'N/A')
        history_text += f"--- Întrebarea {idx+1} (Nota: {note}/10) ---\nQ: {q}\nA: {a}\nFeedback: {feedback}\n\n"

    prompt = f"""
    Ești un Career Coach AI. Folosește istoricul pentru a genera un raport final.
    FORMAT JSON STRICT:
    {{
      "final_score": "medie din scoruri",
      "summary": "Sinteză generală în Markdown",
      "key_strengths": ["3 puncte forte"],
      "areas_for_improvement": ["3 arii de îmbunătățire"],
      "next_steps_recommendation": "Recomandări pentru următorii pași"
    }}
    ISTORIC INTERVIU:\n{history_text}
    JOB SUMMARY:\n{job_summary}
    CV TEXT:\n{cv_text}
    """
    # 🎯 FOLOSIM: call_gemini_json
    result = call_gemini_json(prompt)
    return jsonify(result), 200 if "error" not in result else 500

# --------------------------
# ROUTE: Rezultate HTML STAR (RAW - returnează text HTML)
@app.route('/coach-results-html', methods=['POST'])
def coach_results_html():
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
        
        prompt = f"""
        Întrebarea: {question}
        Răspunsul utilizatorului: {user_answer}
        Te rog să rescrii acest răspuns într-o versiune optimizată STAR (Situation, Task, Action, Result).
        Returnează DOAR textul răspunsului optimizat.
        """
        # 🎯 FOLOSIM: call_gemini_raw
        star_answer_result = call_gemini_raw(prompt)
        
        star_answer = star_answer_result if isinstance(star_answer_result, str) else star_answer_result.get("details", "Eroare generare STAR")
        
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
# ROUTE: STAR next (RAW - returnează textul STAR în JSON)
@app.route('/coach-next', methods=['POST'])
def coach_next():
    data = request.get_json()
    question = data.get('question')
    user_answer = data.get('user_answer')
    if not question or not user_answer:
        return jsonify({"error": "Întrebare și răspuns obligatorii"}), 400

    prompt = f"""
    Întrebarea: {question}
    Răspunsul utilizatorului: {user_answer}
    Te rog să rescrii acest răspuns într-o versiune optimizată STAR.
    Returnează DOAR textul răspunsului optimizat.
    """
    # 🎯 FOLOSIM: call_gemini_raw
    star_answer_result = call_gemini_raw(prompt)

    if isinstance(star_answer_result, dict) and "error" in star_answer_result:
        return jsonify(star_answer_result), 500

    star_answer = star_answer_result

    return jsonify({
        "question": question,
        "user_answer": user_answer,
        "star_answer": star_answer
    }), 200

# --------------------------
# PORNIRE SERVER
if __name__ == '__main__':
    print("🚀 Server Flask robust pornit pe [http://0.0.0.0:5000/](http://0.0.0.0:5000/)")
    # Recomandăm să folosești gunicorn sau un alt server WSGI pentru producție.
    # Dacă rulezi local, lasă app.run.
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # Pentru Render, de obicei se folosește un entry point gunicorn, dar lăsăm app pentru testare locală.
    pass



