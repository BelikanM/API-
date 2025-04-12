from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import joblib
import base64
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)
CORS(app)  # Activer CORS pour permettre les requêtes depuis le frontend

# Créer un dossier pour stocker les modèles si nécessaire
os.makedirs('models', exist_ok=True)

# Chemin vers le modèle
MODEL_PATH = 'models/posture_model.pkl'
SCALER_PATH = 'models/posture_scaler.pkl'

# Vérifier si le modèle existe, sinon en créer un simple
if not os.path.exists(MODEL_PATH):
    print("Création d'un modèle d'exemple...")
    # Créer des données d'exemple pour les postures
    # Format: [hauteur/largeur_ratio, angle_du_corps, etc.]
    X_train = np.array([
        [2.1, 170, 0.2],  # Debout
        [2.0, 175, 0.1],  # Debout
        [1.5, 90, 0.5],   # Assis
        [1.4, 100, 0.6],  # Assis
        [0.8, 45, 0.9],   # Courbé
        [0.7, 30, 0.8],   # Courbé
    ])
    y_train = np.array(['debout', 'debout', 'assis', 'assis', 'courbé', 'courbé'])
    
    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Entraîner un modèle simple
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_scaled, y_train)
    
    # Sauvegarder le modèle et le scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Modèle d'exemple créé et sauvegardé!")
else:
    print("Modèle existant trouvé!")

# Charger le modèle et le scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Fonction pour détecter les personnes et analyser leurs postures
def detect_and_analyze_people(image):
    # Convertir l'image en niveaux de gris pour la détection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Utiliser HOG (Histogram of Oriented Gradients) pour la détection de personnes
    # C'est plus précis que les cascades de Haar pour les personnes
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Détecter les personnes dans l'image
    boxes, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(4, 4), scale=1.05)
    
    people_data = []
    
    # Pour chaque personne détectée
    for i, (x, y, w, h) in enumerate(boxes):
        # Extraire la région d'intérêt (ROI) pour cette personne
        person_roi = image[y:y+h, x:x+w]
        
        # Calculer des caractéristiques pour l'analyse de posture
        # Exemple: ratio hauteur/largeur, estimation de l'angle du corps, etc.
        height_width_ratio = h / w
        
        # Simuler une analyse d'angle du corps (dans un cas réel, cela serait plus complexe)
        # Utiliser une pose estimation model serait idéal ici
        # Pour cet exemple, nous générons une valeur aléatoire basée sur le ratio
        body_angle = 180 * height_width_ratio / 3
        
        # Autre caractéristique d'exemple (pourrait être la position relative des épaules, etc.)
        other_feature = w / (h + w)
        
        # Préparer les caractéristiques pour la prédiction
        features = np.array([[height_width_ratio, body_angle, other_feature]])
        features_scaled = scaler.transform(features)
        
        # Prédire la posture
        posture = model.predict(features_scaled)[0]
        
        # Stocker les données de cette personne
        person_data = {
            'id': i,
            'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'posture': posture,
            'confidence': float(weights[i])
        }
        
        people_data.append(person_data)
    
    return people_data

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Récupérer les données de l'image
        data = request.json
        image_data = data.get('image')
        
        # Décoder l'image base64
        image_bytes = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Analyser l'image
        people_data = detect_and_analyze_people(image)
        
        # Compter les postures
        posture_counts = {}
        for person in people_data:
            posture = person['posture']
            posture_counts[posture] = posture_counts.get(posture, 0) + 1
        
        # Préparer la réponse
        response = {
            'timestamp': time.time(),
            'total_people': len(people_data),
            'people': people_data,
            'posture_summary': posture_counts
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Cette route permettrait de réentraîner le modèle avec de nouvelles données
        # Pour cet exemple, nous simulons juste un réentraînement
        
        return jsonify({'message': 'Modèle réentraîné avec succès!', 'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5007)

