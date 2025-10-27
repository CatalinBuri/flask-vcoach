from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
# Permite acces de oriunde (poți restrânge la domeniul tău live)
CORS(app)

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

        processed_text = f"Sinteza Jobului: {text[:150]}..."
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

        # Convertim history la obiect Python dacă e string
        if isinstance(history, str):
            history = json.loads(history)

        report_text = f"Raport final:\nSinteză job: {summary}\nIstoric interviu: {json.dumps(history)[:300]}...\nCV: {cv_text[:150]}..."
        return jsonify({"report_text": report_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
