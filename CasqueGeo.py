#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import urllib.request
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
import io

# === Configuration du fichier cascade ===

# Nom du fichier cascade et URL pour le télécharger s'il manque
CASCADE_FILENAME = "haarcascade_fullbody.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_fullbody.xml"

def ensure_cascade_file():
    """Vérifie si le fichier cascade existe ; sinon, le télécharge."""
    if not os.path.exists(CASCADE_FILENAME):
        print(f"{CASCADE_FILENAME} introuvable. Téléchargement depuis {CASCADE_URL}...")
        try:
            urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILENAME)
            print("Téléchargement terminé.")
        except Exception as e:
            print("Échec du téléchargement du fichier cascade:", e)
            exit(1)

ensure_cascade_file()

# Charger le classificateur pour la détection des personnes
body_cascade = cv2.CascadeClassifier(CASCADE_FILENAME)
if body_cascade.empty():
    print("Échec du chargement du classificateur depuis", CASCADE_FILENAME)
    exit(1)

# === Création de l'API Flask ===
app = Flask(__name__)
CORS(app)

def decode_image(base64_str):
    """
    Décodage d'une image en DataURL base64
    et conversion en objet PIL.Image en RGB.
    """
    header, b64 = base64_str.split(",", 1)
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def analyze_frame(img):
    """
    Convertit l'image (PIL.Image) en niveaux de gris,
    applique le cascade classifier pour détecter les personnes,
    et renvoie un dictionnaire contenant le nombre de personnes détectées et leurs bounding boxes.
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Détection des personnes dans l'image
    bodies = body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    persons = []
    for (x, y, w, h) in bodies:
        persons.append({
            "bbox": [int(x), int(y), int(w), int(h)]
        })
    
    return {
        "people_detected": len(persons),
        "persons": persons
    }

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Endpoint pour l'analyse.
    Attend un JSON avec {"image": "data:image/jpeg;base64,..."}.
    Renvoie le nombre de personnes détectées et leurs positions.
    """
    data = request.get_json(force=True)
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "Aucune image reçue"}), 400
    
    try:
        img = decode_image(img_b64)
        result = analyze_frame(img)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Lancement du serveur Flask sur le port 5007...")
    app.run(host="0.0.0.0", port=5007, debug=True)
