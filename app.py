import traceback
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

# ------------------------------
# CONFIGURARE APLICAȚIE FLASK
# ------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://www.pixelplayground3d.ro", "http://localhost:3000", "http://127.0.0.1:3000"]}},
     supports_credentials=True)

# ------------------------------
# INIȚIALIZARE GEMINI CLIENT
# ------------------------------
try:
    gemini_client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
except Exception as e:
    gemini_client = None
    print("Eroare inițializare Gemini:", e)

# ------------------------------
# FUNCȚIE UTILITARĂ: Extract JSON sigur
# ------------------------------
def safe_json_extract(response_text):
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end]
        return json.loads(json_str)
    except Exception:
        return {"processed_text": response_text}


# ------------------------------
# ENDPOINT 1: /process-text
# ------------------------------
@app.route("/process-text", methods=["POST"])
def process_text():
    try:
        data = request.get_json()
        text = data.get("text", "").replace("bullet icon", "").strip()
        if not text:
            return jsonify({"error": "Textul este gol"}), 400

        prompt = f"""
        Ești un asistent HR care face sinteze concise și profesioniste în limba română.
        Elimină repetițiile și expresii inutile (ex. „bullet icon”).
        Text de analizat:
        {text}

        Returnează JSON cu cheia "processed_text".
        """

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        ai_data = safe_json_extract(re_
