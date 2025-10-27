from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Permite CORS doar pentru domeniul tău
CORS(app, resources={r"/*": {"origins": "https://www.pixelplayground3d.ro"}}, supports_credentials=True)

# -----------------------
# Endpoint: /generate-cover-letter
# -----------------------
@app.route("/generate-cover-letter", methods=["POST"])
def generate_cover_letter():
    try:
        data = request.get_json()
        cv_text = data.get("cv_text")
        job_text = data.get("job_text")

        if not cv_text or not job_text:
            return jsonify({"error": "CV și Job Description sunt obligatorii"}), 400

        # Dummy cover letter
        cover_letter = f"Acesta este un exemplu de scrisoare de intenție pentru jobul tău:\n\n{job_text}\n\nCV:\n{cv_text}"

        return jsonify({"cover_letter": cover_letter})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Endpoint: /optimize-linkedin-profile
# -----------------------
@app.route("/optimize-linkedin-profile", methods=["POST"])
def optimize_linkedin_profile():
    try:
        data = request.get_json()
        cv_text = data.get("cv_text")
        domain = data.get("domain")

        if not cv_text or not domain:
            return jsonify({"error": "CV și domeniu sunt obligatorii"}), 400

        # Dummy headlines și about section
        headlines = [
            f"{domain} Specialist cu experiență",
            f"Profesional în {domain} și AI",
            f"Expert {domain} | Dezvoltator CV"
        ]
        about_section = f"Profil optimizat pentru domeniul {domain} bazat pe CV-ul tău."

        return jsonify({"headlines": headlines, "about_section": about_section})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Endpoint: /generate-job-queries
# -----------------------
@app.route("/generate-job-queries", methods=["POST"])
def generate_job_queries():
    try:
        data = request.get_json()
        cv_text = data.get("cv_text")

        if not cv_text:
            return jsonify({"error": "CV este obligatoriu"}), 400

        # Dummy queries
        queries = [
            "Software Developer Python",
            "AI Engineer",
            "Data Scientist Junior",
            "Backend Developer Remote"
        ]

        return jsonify({"queries": queries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Endpoint: /process-text
# -----------------------
@app.route("/process-text", methods=["POST"])
def process_text():
    try:
        data = request.get_json()
        text = data.get("text")

        if not text:
            return jsonify({"error": "Text obligatoriu"}), 400

        # Dummy processed text
        processed_text = f"Sinteza Jobului: {text[:150]}..."  # primele 150 caractere

        return jsonify({"processed_text": processed_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Endpoint: /generate-questions
# -----------------------
@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    try:
        data = request.get_json()
        processed_text = data.get("processed_text")
        cv_text = data.get("cv_text")

        if not processed_text or not cv_text:
            return jsonify({"error": "Text procesat și CV obligatorii"}), 400

        # Dummy întrebări
        questions = [
            "Povestește-ne despre experiența ta relevantă.",
            "Care sunt punctele tale forte?",
            "De ce vrei să lucrezi la această companie?",
            "Cum abordezi o problemă complexă?",
            "Unde te vezi peste 5 ani?"
        ]

        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Endpoint: /evaluate-answer
# -----------------------
@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    try:
        data = request.get_json()
        question = data.get("question")
        answer = data.get("answer")
        cv_text = data.get("cv_text")
        job_summary = data.get("job_summary")
        previous_answer = data.get("previous_answer")
        previous_evaluation = data.get("previous_evaluation")

        if not question or not answer or not cv_text:
            return jsonify({"error": "question, answer și cv_text sunt obligatorii"}), 400

        # Dummy evaluare
        current_evaluation = {
            "feedback": "Răspuns corect, dar poate fi mai concis.",
            "nota_finala": 8,
            "claritate": 8,
            "relevanta": 8,
            "structura": 8
        }
        comparative_feedback = {"feedback": "Comparativ cu răspunsul anterior, ai progresat."}

        return jsonify({
            "current_evaluation": current_evaluation,
            "comparative_feedback": comparative_feedback
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------
# Endpoint: /generate-report
# -----------------------
@app.route("/generate-report", methods=["POST"])
def generate_report():
    try:
        data = request.get_json()
        summary = data.get("summary")
        history = data.get("history")
        cv_text = data.get("cv_text")

        if not summary or not history or not cv_text:
            return jsonify({"error": "summary, history și cv_text sunt obligatorii"}), 400

        # Dummy raport
        report_text = f"Raport final:\nSinteză job: {summary}\nIstoric interviu: {history[:300]}...\nCV: {cv_text[:150]}..."

        return jsonify({"report_text": report_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# RUN SERVER
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
