from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import re  # Importer la bibliothèque de expressions régulières pour l'extraction

app = Flask(__name__)
CORS(app)

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """
    Traiter le fichier PDF téléversé pour l'analyse des zones à éviter.
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
    keywords = ['danger', 'avoid', 'accident', 'crime', 'unsafe']
    zones = []

    for line in text.splitlines():
        for keyword in keywords:
            if keyword in line.lower():
                # Utiliser une expression régulière pour extraire le nom de la zone
                match = re.search(r'(\w+|\w+\s\w+) \s*(?:zone|quartier|ville)', line, re.I)
                if match:
                    zone_name = match.group(0).strip()
                    zones.append({'name': zone_name, 'description': line})

    return zones

if __name__ == '__main__':
    app.run(port=5006, debug=True)

