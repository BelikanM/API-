from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.cluster import KMeans
import pdfplumber

app = Flask(__name__)
CORS(app)

# Charger le modèle existant
activity_model = joblib.load('activity_model.joblib')  # Assurez-vous que ce fichier existe

@app.route('/train', methods=['POST'])
def train_model():
    """
    Entraîner le modèle KMeans ou autres avec de nouvelles données GPS.
    """
    gps_data = request.json.get('gps_data')  # Format attendu: liste de listes [[lat, lon], ...]
    gps_data = np.array(gps_data)
    
    # Entraînez un nouveau modèle KMeans
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(gps_data)
    
    joblib.dump(kmeans, 'clustering_model.joblib')  # Met à jour le modèle sauvegardé
    return jsonify({'message': 'Model trained successfully'}), 200

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """
    Traiter le fichier PDF téléversé pour l'analyse de zone.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    pdf_file = request.files['file']
    text = extract_text_from_pdf(pdf_file)
    zones_to_avoid = analyze_pdf_text(text)
    return jsonify({'zones_to_avoid': zones_to_avoid}), 200

def extract_text_from_pdf(pdf_file):
    """
    Extraire du texte d'un fichier PDF en utilisant pdfplumber.
    """
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def analyze_pdf_text(text):
    """
    Analyse du texte pour identifier des zones à éviter.
    """
    keywords = ['danger', 'avoid', 'accident', 'crime', 'unsafe']  # Mots clés à rechercher
    zones = []
    
    for line in text.splitlines():
        for keyword in keywords:
            if keyword in line.lower():
                zones.append(line)
    return zones

if __name__ == '__main__':
    app.run(port=5006, debug=True)

