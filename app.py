from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import json
import re
from datetime import datetime

# ✅ Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Mock Gemini client (înlocuiește cu clientul tău real)
from some_gemini_wrapper import gemini_client  # ← sau importul tău corect

# --------------------------------------------------------
# ✅ Funcții utilitare
# --------------------------------------------------------

def safe_json_extract(text):
    """
    Extrage obiect JSON valid din textul primit de la AI.
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {}

# --------------------------------------------------------
# ✅ ROUTE: Process job text
# --------------------------------------------------------

@app.route("/process-text", methods=["POST", "OPTIONS"])
def process_text():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Textul jobului lipsește."}), 400

        prompt = f"""
        Rezumă acest anunț de job, evidențiind:
        - Competențe esențiale
        - Tehnologii specifice
        - Responsabilități principale
        - Cuvinte cheie utile pentru interviu
        Returnează un text coerent în limba română.
        """
        result = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=f"{prompt}\n\n{text}"
        )
        processed_text = result.text.strip()
        return jsonify({"processed_text": processed_text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Eroare la procesarea textului.", "details": str(e)}), 500

# --------------------------------------------------------
# ✅ ROUTE: Generate Interview Questions
# --------------------------------------------------------

@app.route("/generate-questions", methods=["POST", "OPTIONS"])
def generate_questions():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        data = request.get_json()
        cv_text = data.get("cv_text", "")
        job_text = data.get("job_text", "")
        processed_text = data.get("processed_text", "")

        if not cv_text or not job_text:
            return jsonify({"error": "CV-ul și descrierea jobului sunt obligatorii."}), 400

        prompt = f"""
        Ești un recrutor AI. Creează 8-10 întrebări relevante pentru un interviu bazat pe:
        - CV-ul candidatului
        - Descrierea jobului
        - Rezumatul procesat

        Răspuns strict JSON:
        {{
            "questions": [
                "Întrebare 1 ...",
                "Întrebare 2 ..."
            ]
        }}
        """

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{prompt}\n\nCV:\n{cv_text}\n\nJOB:\n{job_text}\n\nRezumat:\n{processed_text}"
        )

        questions_data = safe_json_extract(response.text)
        questions = questions_data.get("questions", [])

        if not isinstance(questions, list) or not questions:
            return jsonify({"error": "Nu s-au putut genera întrebări valide."}), 500

        return jsonify({"questions": questions})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Eroare la generarea întrebărilor.", "details": str(e)}), 500

# --------------------------------------------------------
# ✅ ROUTE: Analyze candidate's answer (CORE FIX)
# --------------------------------------------------------

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
        Analizează răspunsul candidatului la întrebarea de interviu de mai jos.
        Întrebare: "{question}"
        Răspuns candidat: "{user_answer}"
        Istoric anterior: {history_text}

        Returnează feedback detaliat și o evaluare numerică (1-10).
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

        # ✅ Fallback pentru structura corectă
        if "current_evaluation" not in ai_data:
            ai_data = {"current_evaluation": ai_data}

        # ✅ Default values dacă modelul nu trimite complet
        eval_data = ai_data["current_evaluation"]
        eval_data.setdefault("feedback", "Feedback indisponibil.")
        eval_data.setdefault("nota_finala", 7)
        eval_data.setdefault("claritate", 7)
        eval_data.setdefault("relevanta", 7)
        eval_data.setdefault("structura", 7)

        return jsonify(ai_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Eroare la analizarea răspunsului.", "details": str(e)}), 500

# --------------------------------------------------------
# ✅ ROUTE: Generate final interview report
# --------------------------------------------------------

@app.route("/generate-report", methods=["POST", "OPTIONS"])
def generate_report():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        data = request.get_json()
        history = data.get("history", [])
        job_summary = data.get("job_summary", "")
        cv_text = data.get("cv_text", "")

        avg_score = 0
        count = 0
        for item in history:
            ev = item.get("analysis", {}).get("evaluation", {})
            if "nota_finala" in ev:
                avg_score += ev["nota_finala"]
                count += 1
        avg_score = round(avg_score / count, 2) if count > 0 else 0

        final_feedback = f"""
        **Rezumat general:**
        Scor mediu: {avg_score}/10  
        Candidat: evaluare general pozitivă.  

        **Job Summary:**
        {job_summary[:500]}...

        **CV:**  
        {cv_text[:500]}...
        """

        return jsonify({
            "average_score": avg_score,
            "final_feedback": final_feedback,
            "total_questions": count,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Eroare la generarea raportului final.", "details": str(e)}), 500

# --------------------------------------------------------
# ✅ ROUTE: LinkedIn & Job Hunt Integration
# --------------------------------------------------------

@app.route("/linkedin-optimizer", methods=["POST", "OPTIONS"])
def linkedin_optimizer():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    try:
        data = request.get_json()
        cv = data.get("cv_text", "")
        job = data.get("job_text", "")
        if not cv or not job:
            return jsonify({"error": "CV și Job Description obligatorii."}), 400

        prompt = f"""
        Analizează CV-ul și descrierea jobului.
        Propune optimizări pentru profilul LinkedIn (Headline, About, Skills).
        Răspuns strict JSON:
        {{
            "headline": "...",
            "about_section": "...",
            "recommended_skills": ["...", "..."]
        }}
        """
        result = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{prompt}\n\nCV:\n{cv}\n\nJob:\n{job}"
        )

        data_out = safe_json_extract(result.text)
        return jsonify(data_out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Eroare la optimizarea LinkedIn.", "details": str(e)}), 500

# --------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
