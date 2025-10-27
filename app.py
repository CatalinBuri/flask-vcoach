import os
import json
import re
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

# -----------------------
# UTILITY
# -----------------------
def safe_json_extract(text):
    """Extrage obiectul JSON încapsulat în ```json...``` sau returnează JSON-ul brut."""
    text = text.strip()
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Încercare de remediere
        json_str_clean = json_str.replace('\\"', '"').replace('\n', '').replace('\t', '')
        try:
            return json.loads(json_str_clean)
        except:
            raise json.JSONDecodeError("Parsare JSON eșuată după multiple încercări.", text, 0)

# -----------------------
# INIȚIALIZARE
# -----------------------
load_dotenv()
app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = None
if API_KEY:
    gemini_client = genai.Client(api_key=API_KEY)
else:
    print("EROARE: GEMINI_API_KEY lipsește!")

# -----------------------
# ENDPOINT: /generate-questions
# -----------------------
@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    if not gemini_client:
        return jsonify({"error": "Clientul Gemini nu este inițializat."}), 500
    try:
        data = request.get_json()
        print("📩 Data primită:", data)
        cv_text = data.get("cv_text")
        job_text = data.get("job_text")
        if not cv_text or not job_text:
            return jsonify({"error": "CV-ul și Job Description sunt obligatorii."}), 400

        prompt = f"""
        Ești un recrutor AI specializat. Analizează CV și Job Description și generează:
        1. O sinteză scurtă a postului.
        2. 5 întrebări personalizate de interviu.
        
        CV:
        {cv_text}
        
        Job Description:
        {job_text}
        
        Format JSON:
        {{
            "summary": "...",
            "questions": ["...", "...", "...", "...", "..."],
            "question_count": 5
        }}
        """
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare întrebări eșuată", "details": str(e)}), 500

# -----------------------
# ENDPOINT: /analyze-answer
# -----------------------
@app.route("/analyze-answer", methods=["POST"])
def analyze_answer():
    if not gemini_client:
        return jsonify({"error": "Clientul Gemini nu este inițializat."}), 500
    try:
        data = request.get_json()
        question = data.get("question")
        user_answer = data.get("user_answer")
        previous_history = data.get("history", [])

        if not question or not user_answer:
            return jsonify({"error": "Întrebarea și răspunsul sunt obligatorii."}), 400

        history_text = json.dumps(previous_history, indent=2) if previous_history else ""
        prompt = f"""
        Ești un recrutor AI. Analizează răspunsul candidatului.
        Întrebare: "{question}"
        Răspuns candidat: "{user_answer}"
        Istoric: {history_text}

        Răspunsul trebuie să fie JSON cu:
        {{
            "current_evaluation": {{
                "feedback": "...",
                "nota_finala": 8,
                "claritate": 9,
                "relevanta": 7,
                "structura": 8
            }}
        }}
        """
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Analiza răspunsului eșuată", "details": str(e)}), 500

# -----------------------
# ENDPOINT: /generate-report
# -----------------------
@app.route("/generate-report", methods=["POST"])
def generate_report():
    if not gemini_client:
        return jsonify({"error": "Clientul Gemini nu este inițializat."}), 500
    try:
        data = request.get_json()
        job_summary = data.get("job_summary")
        history = data.get("history")
        if not job_summary or not history:
            return jsonify({"error": "Sinteza job și istoricul sunt obligatorii."}), 400

        history_text = json.dumps(history, indent=2)
        prompt = f"""
        Ești un Recrutor AI Senior. Pe baza următorului istoric:
        {history_text}
        și sinteza jobului: {job_summary}

        Generează raport final strict JSON cu:
        {{
            "overall_report_markdown": "...",
            "compatibility_score": 75
        }}
        """
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare raport eșuat", "details": str(e)}), 500

# -----------------------
# ENDPOINT: /process-text
# -----------------------
@app.route("/process-text", methods=["POST"])
def process_text():
    try:
        data = request.get_json()
        job_text = data.get("text", "").strip()
        if not job_text:
            return jsonify({"error": "Job description is required."}), 400

        # 🧹 Curățare text — eliminăm "bullet icon" și spațiile duble
        clean_text = re.sub(r"(?i)\b(bullet\s*icon)\b", "", job_text)
        clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()

        # 🧠 Prompt clar pentru sinteză în română
        prompt = f"""
        Rezumă în limba română principalele responsabilități și competențe din următorul text de job description.
        Oferă un text coerent, ușor de înțeles, fără liste cu bullet-uri și fără menționarea expresiei 'bullet icon'.

        Text job:
        {clean_text}
        """

        # ✨ Generare conținut cu Gemini
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        # Extragem sinteza
        summary = response.text.strip()

        return jsonify({"processed_text": summary})

    except Exception as e:
        print(f"Eroare la /process-text: {str(e)}")
        return jsonify({"error": "Eroare internă la procesarea textului."}), 500

# -----------------------
# ENDPOINT: /generate-beginner-faq
# -----------------------
@app.route("/generate-beginner-faq", methods=["POST"])
def generate_beginner_faq():
    if not gemini_client:
        return jsonify({"error": "Gemini nu este inițializat"}), 500
    try:
        data = request.get_json()
        cv_text = data.get("cv_text", "").strip()
        if not cv_text:
            prompt_context = "Generează 5 întrebări frecvente pentru candidați entry-level, fără CV."
        else:
            prompt_context = f"Generează 5 întrebări FAQ personalizate pe baza CV-ului: {cv_text}"

        prompt = f"""
        Ești Recrutor AI. {prompt_context}
        Răspunsul strict JSON: 
        {{
            "faq":[
                {{"question":"Întrebare 1","explanation":"..."}},
                {{"question":"Întrebare 2","explanation":"..."}},
                {{"question":"Întrebare 3","explanation":"..."}},
                {{"question":"Întrebare 4","explanation":"..."}},
                {{"question":"Întrebare 5","explanation":"..."}}
            ]
        }}
        """
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare FAQ eșuată", "details": str(e)}), 500

# -----------------------
# ENDPOINT: /analyze-faq-answers
# -----------------------
@app.route("/analyze-faq-answers", methods=["POST"])
def analyze_faq_answers():
    if not gemini_client:
        return jsonify({"error": "Gemini nu este inițializat"}), 500
    try:
        data = request.get_json()
        faq_data = data.get("faq_data")
        if not faq_data:
            return jsonify({"error": "faq_data obligatoriu"}), 400
        item = faq_data[0]
        prompt = f"""
        Evaluează răspunsul utilizatorului: {item.get('user_answer','')}
        La întrebarea: {item.get('question','')}
        Explicația intenției recrutorului: {item.get('explanation','')}
        Răspunsul strict JSON cu evaluare.
        """
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Analiză FAQ eșuată", "details": str(e)}), 500

# -----------------------
# ENDPOINT: /generate-final-report
# -----------------------
@app.route("/generate-final-report", methods=["POST"])
def generate_final_report():
    if not gemini_client:
        return jsonify({"error": "Gemini nu este inițializat"}), 500
    try:
        data = request.get_json()
        faq_history = data.get("faq_history")
        if not faq_history:
            return jsonify({"error": "faq_history obligatoriu"}), 400

        history_text = ""
        for idx, entry in enumerate(faq_history):
            q = entry.get("question_data", {}).get("question", "N/A")
            a = entry.get("user_answer", "N/A")
            note = entry.get("analysis", {}).get("evaluation", {}).get("nota_finala", "N/A")
            feedback = entry.get("analysis", {}).get("evaluation", {}).get("feedback", "N/A")
            history_text += f"Întrebarea {idx+1} (Nota: {note}/10)\nQ: {q}\nA: {a}\nFeedback: {feedback}\n\n"

        prompt = f"""
        Generează raport final sinteză bazat pe istoricul următor:
        {history_text}
        Răspuns strict JSON cu scor final, puncte forte, arii de îmbunătățire și next steps.
        """
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare raport final eșuat", "details": str(e)}), 500

# -----------------------
# ENDPOINT: /generate-cover-letter
# -----------------------
@app.route("/generate-cover-letter", methods=["POST"])
def generate_cover_letter():
    if not gemini_client:
        return jsonify({"error": "Gemini nu este inițializat"}), 500
    try:
        data = request.get_json()
        cv_text = data.get("cv_text")
        job_text = data.get("job_text")
        if not cv_text or not job_text:
            return jsonify({"error": "cv_text și job_text obligatorii"}), 400

        prompt = f"""
        Generează scrisoare de intenție personalizată pe baza următoarelor:
        CV: {cv_text}
        Job: {job_text}
        Răspuns strict JSON:
        {{
            "cover_letter": "..."
        }}
        """
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare Cover Letter eșuat", "details": str(e)}), 500

# -----------------------
# ENDPOINT: /generate-linkedin-summary
# -----------------------
@app.route("/generate-linkedin-summary", methods=["POST"])
def generate_linkedin_summary():
    if not gemini_client:
        return jsonify({"error": "Gemini nu este inițializat"}), 500
    try:
        data = request.get_json()
        cv_text = data.get("cv_text")
        if not cv_text:
            return jsonify({"error": "cv_text obligatoriu"}), 400

        prompt = f"""
        Generează un sumar LinkedIn profesional bazat pe CV:
        {cv_text}
        Răspuns strict JSON:
        {{
            "linkedin_summary": "..."
        }}
        """
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        ai_data = safe_json_extract(response.text)
        return jsonify(ai_data), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Generare LinkedIn summary eșuat", "details": str(e)}), 500

# -----------------------
# PORNIRE SERVER
# -----------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)



