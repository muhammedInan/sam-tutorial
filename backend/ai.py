import openai
import key
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

conversation = []

openai.api_key = key.openai_key

def lire_tous_les_fichiers(dossier):
    documents = []
    for fichier in os.listdir(dossier):
        if fichier.endswith('.txt'):
            chemin = os.path.join(dossier, fichier)
            with open(chemin, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    return documents

def trouver_document_pertinent(question, documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents + [question])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    index_plus_similaire = cosine_similarities.argmax()
    return documents[index_plus_similaire]

def get_ai_response(question):
    dossier_documents = os.path.join(os.path.dirname(__file__), 'textes')
    documents = lire_tous_les_fichiers(dossier_documents)
    document_pertinent = trouver_document_pertinent(question, documents)
    
    messages = [
        {"role": "system", "content": "Vous êtes un assistant IA spécialisé. Répondez en 4 phrases maximum en utilisant les informations fournies."},
        {"role": "user", "content": f"Contexte : {document_pertinent}\n\nQuestion : {question}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=200,
        messages=messages
    )

    return response.choices[0].message['content'].strip()

def transcribe(request):
  temp_file_name = None
  try:
      # Créer un fichier temporaire
      with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
          temp_file_name = temp_file.name
          
      # Obtenir le contenu audio de la requête
      audio_content = request.files["audio"]
      
      # Écrire le contenu audio dans le fichier temporaire
      with open(temp_file_name, "wb") as f:
          f.write(audio_content.read())
      
      # Ouvrir le fichier audio pour la transcription
      with open(temp_file_name, "rb") as audio_file:
          # Effectuer la transcription
          transcription = openai.Audio.transcribe("whisper-1", audio_file)
      
      return transcription["text"]
  
  finally:
      # Supprimer le fichier temporaire, s'il existe
      if temp_file_name and os.path.exists(temp_file_name):
          try:
              os.remove(temp_file_name)
          except PermissionError:
              print(f"Impossible de supprimer le fichier temporaire : {temp_file_name}")

def lire_fichier_texte(nom_fichier):
    chemin_fichier = os.path.join(os.path.dirname(__file__), 'textes', nom_fichier)
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
            contenu = fichier.read()
        return contenu
    except FileNotFoundError:
        print(f"Le fichier {nom_fichier} n'a pas été trouvé.")
        return ""
    except Exception as e:
        print(f"Une erreur s'est produite lors de la lecture du fichier {nom_fichier}: {e}")
        return ""
