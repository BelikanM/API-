import os
import urllib.request
import cv2
import numpy as np
import joblib
from flask import Flask, jsonify, request
import threading
import time
from mediapipe import solutions

# Initialisation du modèle et de la cascade
app = Flask(__name__)

# Charger les modèles
model = joblib.load('person_detection_model.joblib')  # Modèle pour prédire le nombre de personnes
scaler = joblib.load('scaler.joblib')  # Scaler pour normaliser les données

# Vérification de la présence du fichier haarcascade_fullbody.xml
body_cascade_path = 'haarcascade_fullbody.xml'

# Si le fichier n'est pas présent, on le télécharge depuis une URL
if not os.path.exists(body_cascade_path):
    print(f"Le fichier {body_cascade_path} est manquant. Téléchargement...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_fullbody.xml"
    urllib.request.urlretrieve(url, body_cascade_path)
    print(f"Fichier téléchargé avec succès : {body_cascade_path}")

# Charger la cascade de détection des corps
body_cascade = cv2.CascadeClassifier(body_cascade_path)

# Variables globales
detected_people = 0  # Nombre de personnes détectées
pose_estimation = []

def detect_people(camera_source=0):
    global detected_people, pose_estimation
    # Initialisation de MediaPipe pour la détection de pose
    mp_pose = solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(camera_source)  # Choisir la caméra (0 pour arrière, 1 pour avant)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir l'image en niveau de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détection des personnes avec haarcascade
        bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        detected_people = len(bodies)

        # Convertir l'image pour la détection de pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(frame_rgb)

        if results.pose_landmarks:
            pose_estimation = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            # Dessiner les landmarks sur l'image
            mp_pose.POSE_CONNECTIONS(frame)

        # Afficher le nombre de personnes et les points de la pose
        cv2.putText(frame, f'Personnes détectées: {detected_people}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Affichage des résultats sur l'image
        cv2.imshow('Detection', frame)

        # Si l'utilisateur appuie sur 'q', on arrête la détection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Lance la détection en arrière-plan avec la caméra arrière (0)"""
    detection_thread = threading.Thread(target=detect_people, args=(0,))
    detection_thread.daemon = True
    detection_thread.start()
    return jsonify({"message": "Détection lancée avec la caméra arrière!"})

@app.route('/start_front_detection', methods=['POST'])
def start_front_detection():
    """Lance la détection en arrière-plan avec la caméra avant (1)"""
    detection_thread = threading.Thread(target=detect_people, args=(1,))
    detection_thread.daemon = True
    detection_thread.start()
    return jsonify({"message": "Détection lancée avec la caméra avant!"})

@app.route('/get_detection', methods=['GET'])
def get_detection():
    """Retourne le nombre de personnes détectées et les landmarks de la pose"""
    return jsonify({
        "people_detected": detected_people,
        "pose_estimation": pose_estimation
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
