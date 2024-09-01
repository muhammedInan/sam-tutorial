from flask import Flask, request, Response
from flask_cors import CORS
from ai import get_ai_response, transcribe
from elevenlabs import generate, stream, set_api_key, voices, Voice
import key
app = Flask(__name__)

CORS(app)

set_api_key(key.ELEVENLABS_API_KEY)

@app.route("/speak", methods=["POST"])
def speak():
    question = transcribe(request)
    generate_response = get_ai_response(question)
    
    # Supprimez cette ligne :
    # full_text = "".join(generate_response())
    
    # Utilisez directement generate_response :
    full_text = generate_response
    
    # Le reste du code reste inchangé
    voice = Voice(voice_id="hRacV2aLliCx5S1IaxNs")
    
    audio = generate(
        text=full_text,
        voice=voice,
        model="eleven_multilingual_v2"
    )
    
    return Response(audio, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(debug=True)