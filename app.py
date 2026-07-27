import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Permite cereri Cross-Origin dinspre site-ul WordPress

logging.basicConfig(level=logging.INFO)

# Inițializare client OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "online", "message": "AI Career Suite Backend is running."})

@app.route("/api/cv-quality", methods=["POST"])
def analyze_cv_quality():
    try:
        data = request.get_json() or {}
        
        cv_text = data.get("cv_text", "").strip()
        job_description = data.get("job_description", "").strip()
        job_image = data.get("job_image", None)  # Base64 string data:image/png;base64,...

        if not cv_text:
            return jsonify({"error": "Lipseste textul din CV."}), 400

        if not job_description and not job_image:
            return jsonify({"error": "Lipsesc cerințele jobului (text sau captură)."}), 400

        logging.info(f"Procesare request - CV Text Len: {len(cv_text)}, Job Image Present: {bool(job_image)}")

        # Construire mesaj pentru OpenAI (Vision Model GPT-4o)
        content_items = [
            {
                "type": "text",
                "text": f"""Ești un recruiter profesionist și un expert HR. 
Analizează CV-ul candidatului în raport cu cerințele jobului.

=== TEXT CV ===
{cv_text}

=== DESCRIERE JOB (TEXT) ===
{job_description if job_description else 'Consulta imaginea atasata pentru cerintele jobului.'}

Returnează un răspuns exclusiv în format JSON valid, fără formate markdown, după următoarea structură:
{{
    "clarity_score": 8,
    "relevance_score": 7,
    "structure_score": 9,
    "concrete_improvements": [
        "Punctul 1 de îmbunătățire...",
        "Punctul 2 de îmbunătățire..."
    ],
    "suggested_rephrasings": [
        "Reformulare recomandată 1...",
        "Reformulare recomandată 2..."
    ]
}}
"""
            }
        ]

        # Dacă s-a trimis o captură de ecran Base64, o adăugăm în mesajul de Vision
        if job_image:
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": job_image
                }
            })

        if not client:
            # Fallback mock dacă nu este setat OPENAI_API_KEY în mediul Render
            return jsonify({
                "payload": {
                    "clarity_score": 8,
                    "relevance_score": 7,
                    "structure_score": 8,
                    "concrete_improvements": [
                        "Adaugă mai multe cifre și rezultate cuantificabile în experiență.",
                        "Corelează abilitățile din CV cu termenii cheie din captura de ecran a jobului."
                    ],
                    "suggested_rephrasings": [
                        "În loc de 'Am gestionat proiecte', folosește 'Am coordonat 5 proiecte cross-funcționale cu un buget de X...'"
                    ]
                }
            })

        # Apel API către OpenAI GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content_items
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        raw_response = response.choices[0].message.content
        parsed_json = json.loads(raw_response)

        return jsonify({"payload": parsed_json})

    except Exception as e:
        logging.error(f"Eroare procesare: {str(e)}", exc_info=True)
        return jsonify({"error": f"Eroare pe server: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
