import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

# NU importăm Waitress. Folosim serverul Flask standard (app.run)

# ÎNȚIALIZARE
# --------------------------
# Încarcă variabilele de mediu din .env (unde se află GEMINI_API_KEY)
load_dotenv() 

# 1. INIȚIALIZARE APLICAȚIE FLASK ȘI CORS
app = Flask(__name__)
# Permite accesul CORS pentru dezvoltare (ex: http://localhost)
CORS(app) 

# Configurează clientul Gemini.
API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    if not API_KEY:
        print("\n=======================================================")
        print("EROARE CRITICĂ: Variabila de mediu GEMINI_API_KEY lipsește!")
        print("Asigurați-vă că fișierul .env există și conține cheia.")
        print("=======================================================\n")
        gemini_client = None
    else:
        # Se recomandă utilizarea modelului gemini-2.5-flash pentru viteză
        gemini_client = genai.Client(api_key=API_KEY) 
except Exception as e:
    print(f"Eroare la inițializarea clientului Gemini: {e}")
    print("Asigurați-vă că cheia API este setată corect în fișierul .env.")
    gemini_client = None

# --- UTILS ȘI LOGICĂ DE BAZĂ ---

def safe_json_extract(text):
    """
    Extrage și parsează un obiect JSON dintr-un string, gestionând blocuri Markdown
    sau alte artefacte, folosind o metodă robustă.
    """
    if not text:
        raise ValueError("Text gol primit de la AI.")
        
    full_text = text.strip()
    
    # 1. Curățare: Elimină marcajele Markdown (```json)
    if full_text.startswith('```json'):
        full_text = full_text.replace('```json', '', 1).strip()
    if full_text.endswith('```'):
        full_text = full_text[:-3].strip()
        
    try:
        # 2. Parsare directă (dacă e curat)
        return json.loads(full_text)
        
    except json.JSONDecodeError:
        # 3. Metoda de extragere forțată (în caz de artefacte sau text suplimentar)
        try:
            # Găsește prima și ultima acoladă { }
            start_index = full_text.index('{')
            end_index = full_text.rindex('}') + 1
            json_string = full_text[start_index:end_index]
            
            # Reparsare
            return json.loads(json_string)
            
        except ValueError as e:
            # Aruncă eroare dacă indexarea eșuează (nu găsește { sau })
            raise ValueError(f"Nu s-a putut extrage JSON-ul din text. Eroare la indexare: {e}. Text primit (parțial): {full_text[:500]}...")
        except json.JSONDecodeError as e:
            # Aruncă eroare dacă parsarea finală eșuează
            raise json.JSONDecodeError(f"Eroare de parsare JSON la extragere: {e}. String încercat: {json_string[:500]}...", doc=json_string, pos=0)

# --- RUTĂ PENTRU GENERAREA FAQ (ÎNTREBĂRI ȘI EXPLICAȚII) ---

@app.route('/generate-beginner-faq', methods=['POST'])
def generate_beginner_faq():
    if gemini_client is None:
        return jsonify({"error": "Serviciul AI nu este disponibil. Verificați cheia API."}), 503
        
    data = request.get_json()
    cv_text = data.get('cv_text', '').strip()

    # 1. DETECTARE MOD (SPECIFIC sau GENERIC)
    if not cv_text or cv_text == 'GENERIC_FAQ_MODE':
        context_prompt = (
            "Nu ai primit un CV. Generază 5 întrebări standard, frecvente, potrivite "
            "pentru candidații la un rol de începător (entry-level) sau junior."
        )
    else:
        context_prompt = (
            "Generează 5 întrebări FAQ specifice, bazate pe textul CV-ului de mai jos. "
            "Context CV: {cv_text}"
        ).format(cv_text=cv_text)

    # 2. PROMPT-UL PRINCIPAL
    prompt = f"""
    Ești un Recrutor AI. Sarcina ta este să generezi o listă de 5 întrebări frecvente (FAQ) de interviu,
    împreună cu o explicație scurtă a intenției recrutorului pentru fiecare întrebare.
    
    Instrucțiuni:
    - Nu include niciun alt text în afară de JSON.
    - {context_prompt}
    
    Format JSON STRICT:
    {{
      "faq": [
        {{
          "question": "Întrebarea 1?",
          "explanation": "Explicația intenției recrutorului (formatată în Markdown, max 5 propoziții)"
        }},
        {{
          "question": "Întrebarea 2?",
          "explanation": "Explicația intenției recrutorului (formatată în Markdown, max 5 propoziții)"
        }},
        {{
          "question": "Întrebarea 3?",
          "explanation": "Explicația intenției recrutorului (formatată în Markdown, max 5 propoziții)"
        }},
        {{
          "question": "Întrebarea 4?",
          "explanation": "Explicația intenției recrutorului (formatată în Markdown, max 5 propoziții)"
        }},
        {{
          "question": "Întrebarea 5?",
          "explanation": "Explicația intenției recrutorului (formatată în Markdown, max 5 propoziții)"
        }}
      ]
    }}
    """
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
        
    except Exception as e:
        print(f"Eroare în apelul Gemini pentru generarea FAQ: {e}")
        return jsonify({
            "error": "Generarea AI sau parsarea răspunsului pentru FAQ a eșuat.",
            "details": str(e)
        }), 500


# --- RUTĂ PENTRU ANALIZA RĂSPUNSURILOR FAQ (ADĂUGAT SCOR) ---

@app.route('/analyze-faq-answers', methods=['POST'])
def analyze_faq_answers():
    """Analizează un răspuns al utilizatorului la o întrebare FAQ și oferă feedback cu scor."""
    
    if not gemini_client:
        return jsonify({"error": "Clientul Gemini nu este inițializat. Verificați API Key."}), 500

    try:
        data = request.get_json()
        faq_data = data.get('faq_data') 
        
        if not faq_data or not isinstance(faq_data, list) or len(faq_data) == 0:
            return jsonify({"error": "Lipsă sau format invalid pentru 'faq_data'."}), 400
        
        item = faq_data[0]
        analysis_context = f"""
        ÎNTREBAREA: {item.get('question', 'N/A')}
        EXPLICAȚIA INTENȚIEI RECRUTORULUI: {item.get('explanation', 'N/A')}
        RĂSPUNS UTILIZATOR: {item.get('user_answer', 'N/A')}
        """
        
    except Exception as e:
        return jsonify({"error": f"Eroare la preluarea datelor JSON din solicitare: {e}"}), 400

    # PROMPT DE ANALIZĂ
    prompt = f"""
    Ești un antrenor de interviu. Analizează critic RĂSPUNSUL UTILIZATORULUI pe baza ÎNTREBĂRII și a INTENȚIEI RECRUTORULUI.
    
    1. **Evaluează** răspunsul pe o scară de la 1 la 10 pe baza următoarelor criterii: Claritate, Relevanță și Structură (STAR).
    2. Oferă un **feedback** detaliat și constructiv (formatat în Markdown).

    Format JSON STRICT pentru output (asigură-te că folosești cheia 'evaluation'):
    {{
      "analysis_results": [
        {{
          "question": "{item.get('question', 'N/A')}",
          "user_answer": "{item.get('user_answer', 'N/A')}",
          "evaluation": {{
            "nota_finala": 8, // Scorul final între 1 și 10
            "claritate": 9,
            "relevanta": 7,
            "structura": 8,
            "feedback": "Feedback detaliat și constructiv (formatat în Markdown)"
          }}
        }}
      ]
    }}
    
    ---
    CONTEXTUL ANALIZEI:
    {analysis_context}
    """

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        return jsonify(safe_json_extract(response.text)), 200 
        
    except Exception as e:
        print(f"Eroare în apelul Gemini pentru analiza răspunsurilor FAQ: {e}")
        return jsonify({
            "error": "Generarea AI sau parsarea răspunsului pentru analiza FAQ a eșuat.",
            "details": str(e)
        }), 500


# --- NOU: RUTĂ PENTRU GENERAREA RAPORTULUI FINAL (SINTEZĂ) ---

@app.route('/generate-final-report', methods=['POST'])
def generate_final_report():
    """
    Ruta care primește istoricul complet al sesiunii FAQ Coach 
    și generează un raport final agregat de la Gemini.
    """
    if not gemini_client:
        return jsonify({"error": "Gemini API nu este configurat."}), 500
    
    data = request.json
    faq_history = data.get('faq_history', [])
    
    if not faq_history:
        return jsonify({"error": "Istoricul FAQ este gol."}), 400
    
    # Transformăm istoricul într-un format lizibil pentru prompt
    history_text = ""
    for idx, entry in enumerate(faq_history):
        # Utilizăm .get() pentru a preveni erorile în caz că lipsește o cheie
        q = entry.get('question_data', {}).get('question', 'N/A')
        a = entry.get('user_answer', 'N/A')
        note = entry.get('analysis', {}).get('evaluation', {}).get('nota_finala', 'N/A')
        feedback = entry.get('analysis', {}).get('evaluation', {}).get('feedback', 'N/A')
        
        history_text += (
            f"--- Întrebarea {idx+1} (Nota: {note}/10) ---\n"
            f"Întrebare: {q}\n"
            f"Răspuns Utilizator: {a}\n"
            f"Feedback Coach: {feedback}\n\n"
        )

    # Prompt pentru raportul final
    prompt = f"""
    Ești un Expert Coach de Carieră. Ai primit istoricul complet al unei sesiuni de FAQ Coach, care conține 5 întrebări/răspunsuri.
    
    Te rog să generezi un raport de sinteză detaliat.
    
    FORMATUL JSON STRICT pe care trebuie să-l respecți (NU adăuga text suplimentar în afara JSON-ului):
    {{
      "final_score": "O notă medie din cele 5, rotunjită la o zecimală (ex: 7.5).",
      "summary": "O sinteză generală și un comentariu introductiv despre performanța candidatului (formatat în Markdown).",
      "key_strengths": [
        "Identifică cel puțin 3 puncte forte cheie bazate pe răspunsuri și feedback (text simplu)."
      ],
      "areas_for_improvement": [
        "Identifică cel puțin 3 arii specifice care necesită îmbunătățire (text simplu)."
      ],
      "next_steps_recommendation": "Recomandări practice pentru următorii pași în pregătirea interviului (formatat în Markdown)."
    }}
    
    ---\r\n
    ISTORICUL SESIUNII FAQ:\r\n
    {history_text}
    """
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return jsonify(safe_json_extract(response.text)), 200
        
    except Exception as e:
        print(f"Eroare în apelul Gemini pentru generarea raportului final: {e}")
        return jsonify({
            "error": "Generarea AI sau parsarea răspunsului pentru raportul final a eșuat.",
            "details": str(e)
        }), 500


# --- PORNIREA SERVERULUI ---

if __name__ == '__main__':
      print("Server Flask running directly for debug")
      app.run(host='0.0.0.0', port=5000)
