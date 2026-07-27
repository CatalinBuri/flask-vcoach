import os
import io
import json
import base64
import logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_compress import Compress
from dotenv import load_dotenv
import orjson

# Încărcare variabile de mediu din .env
load_dotenv()

# Configurare Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_career_suite")

# Inițializare Flask & Extensii
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
Compress(app)

# JSON Fast Serializer cu orjson
def json_response(data, status=200):
    return Response(
        orjson.dumps(data),
        status=status,
        mimetype="application/json"
    )

# --- INIȚIALIZARE CLIENTS AI ---

# 1. Google GenAI Client (Gemini 2.5 Flash / Vision)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
gemini_client = None

if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Clientul Google GenAI a fost inițializat cu succes.")
    except Exception as e:
        logger.error(f"Eroare la inițializarea Google GenAI: {e}")
else:
    logger.warning("GEMINI_API_KEY / GOOGLE_API_KEY nu este setat.")

# 2. Groq Client (pentru chat rapid / LLM secundar)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = None

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Clientul Groq a fost inițializat cu succes.")
    except Exception as e:
        logger.error(f"Eroare la inițializarea Groq: {e}")
else:
    logger.warning("GROQ_API_KEY nu este setat.")


# --- ENDPOINT-URI APLICAȚIE ---

@app.route("/", methods=["GET"])
def health_check():
    """ Endpoint de Health Check pentru Render """
    return json_response({
        "status": "online",
        "service": "AI Career Suite Backend",
        "gemini_active": bool(gemini_client),
        "groq_active": bool(groq_client)
    })


@app.route("/api/cv-quality", methods=["POST"])
def analyze_cv_quality():
    """
    Endpoint pentru Audit & Calitate CV
    Suportă atât text din Job Description, cât și Captură de Ecran (Base64)
    """
    try:
        data = request.get_json(silent=True) or {}
        
        cv_text = data.get("cv_text", "").strip()
        job_description = data.get("job_description", "").strip()
        job_image_base64 = data.get("job_image", None)

        if not cv_text:
            return json_response({"error": "Lipseste textul din CV."}, status=400)

        if not job_description and not job_image_base64:
            return json_response({"error": "Lipsesc cerintele jobului (text sau captura din tab)."}, status=400)

        logger.info(f"[Audit CV] Processing request - CV Len: {len(cv_text)}, Has Image: {bool(job_image_base64)}")

        prompt = f"""
Ești un recruiter profesionist, expert HR și auditor de carieră.
Analizează CV-ul candidatului în raport cu cerințele jobului furnizate (fie prin text, fie prin captura de ecran).

=== TEXT CV CANDIDAT ===
{cv_text}

=== DESCRIERE JOB (TEXT) ===
{job_description if job_description else 'A se analiza captura de ecran atașată pentru detaliile jobului.'}

Generează un raport complet. Răspunsul TĂU TREBUIE SĂ FIE EXCLUSIV UN OBIECT JSON VALID, respectând strict această structură:
{{
    "clarity_score": 8,
    "relevance_score": 7,
    "structure_score": 9,
    "concrete_improvements": [
        "Recomandare 1 de îmbunătățire...",
        "Recomandare 2 de îmbunătățire..."
    ],
    "suggested_rephrasings": [
        "Reformulare sugerată 1...",
        "Reformulare sugerată 2..."
    ]
}}
"""

        # Construire conținut pentru Gemini (Text + Imagine Opțională)
        contents = [prompt]

        if job_image_base64 and gemini_client:
            try:
                from google.genai import types
                mime_type = "image/png"
                if "," in job_image_base64:
                    header, base64_data = job_image_base64.split(",", 1)
                    if "image/jpeg" in header:
                        mime_type = "image/jpeg"
                    elif "image/webp" in header:
                        mime_type = "image/webp"
                else:
                    base64_data = job_image_base64

                image_bytes = base64.b64decode(base64_data)
                
                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
                contents.append(image_part)
            except Exception as img_err:
                logger.error(f"Eroare la decodarea imaginii base64: {img_err}")

        # Execuție procesare prin Gemini sau Fallback Groq / Mock
        if gemini_client:
            from google.genai import types
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            raw_json = response.text.strip()
            parsed_data = json.loads(raw_json)

        elif groq_client and not job_image_base64:
            # Fallback pe Groq dacă nu avem imagine
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a professional HR auditor. Return ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            parsed_data = json.loads(completion.choices[0].message.content)

        else:
            # Structură Mock în caz că nu există API Keys setate pe Render
            parsed_data = {
                "clarity_score": 8,
                "relevance_score": 7,
                "structure_score": 8,
                "concrete_improvements": [
                    "Sortează experiența profesională în ordine invers-cronologică.",
                    "Adaugă indicatori de performanță (KPI) pentru ultimele roluri."
                ],
                "suggested_rephrasings": [
                    "Înlocuiește 'Aprobat facturi' cu 'Am gestionat fluxul de aprobare pentru bugete de peste 50.000 EUR.'"
                ]
            }

        return json_response({"payload": parsed_data})

    except Exception as e:
        logger.error(f"Eroare la /api/cv-quality: {str(e)}", exc_info=True)
        return json_response({"error": f"Eroare internă server: {str(e)}"}, status=500)


@app.route("/api/interview-simulator", methods=["POST"])
def interview_simulator():
    """ Endpoint pentru Simulatorul de Interviu """
    try:
        data = request.get_json(silent=True) or {}
        cv_text = data.get("cv_text", "")
        job_description = data.get("job_description", "")
        chat_history = data.get("history", [])

        if not cv_text:
            return json_response({"error": "Lipseste textul din CV."}, status=400)

        prompt = f"""
Ești un interviewer tehnic și de HR riguros. 
Conduci un interviu simulativ bazat pe CV-ul candidatului și cerințele jobului.
Răspunde scurt, pune o singură întrebare de interviu odată și oferă un scurt feedback la răspunsul anterior.

CV Candidat: {cv_text[:1500]}
Job Description: {job_description[:1000]}
"""

        if groq_client:
            messages = [{"role": "system", "content": prompt}]
            for msg in chat_history:
                messages.append(msg)
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7
            )
            reply = completion.choices[0].message.content
        else:
            reply = "Bună! Sunt gata să începem interviul. Pentru început, spune-mi mai multe despre ultima ta experiență profesională relevantă."

        return json_response({"reply": reply})

    except Exception as e:
        logger.error(f"Eroare la /api/interview-simulator: {str(e)}", exc_info=True)
        return json_response({"error": str(e)}, status=500)


@app.route("/api/reframe-cv", methods=["POST"])
def reframe_cv():
    """ Endpoint pentru Optimizare / Reformulare CV """
    try:
        data = request.get_json(silent=True) or {}
        cv_text = data.get("cv_text", "")
        job_description = data.get("job_description", "")

        if not cv_text:
            return json_response({"error": "Lipseste textul din CV."}, status=400)

        prompt = f"""
Adaptează și reformulează secțiunile din următorul CV pentru a se potrivi perfect cu jobul țintă.

CV Original:
{cv_text}

Cerințe Job Target:
{job_description}

Returnează versiunea optimizată și îmbunătățită a CV-ului în format text clar, structurat pe secțiuni.
"""

        if gemini_client:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt]
            )
            optimized_cv = response.text
        elif groq_client:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            optimized_cv = completion.choices[0].message.content
        else:
            optimized_cv = f"Versiune optimizată generată pentru CV-ul tău:\n\n{cv_text}"

        return json_response({"optimized_cv": optimized_cv})

    except Exception as e:
        logger.error(f"Eroare la /api/reframe-cv: {str(e)}", exc_info=True)
        return json_response({"error": str(e)}, status=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
