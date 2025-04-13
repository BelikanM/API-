from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import numpy as np
import cv2
import joblib
import torch
import os
import base64
from PIL import Image
from io import BytesIO
import time
import threading
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnalysisAPI')

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Chemin des modèles
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Chargement des modèles
def load_models():
    models = {}
    try:
        # Modèle de détection de personnes (MobileNet SSD)
        prototxt_path = os.path.join(MODELS_PATH, "MobileNetSSD_deploy.prototxt.txt")
        model_path = os.path.join(MODELS_PATH, "MobileNetSSD_deploy.caffemodel")
        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            models['person_detector'] = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            logger.info("Modèle de détection de personnes chargé avec succès")
        
        # Modèle de posture
        posture_model_path = os.path.join(MODELS_PATH, "posture_model.pkl")
        posture_scaler_path = os.path.join(MODELS_PATH, "posture_scaler.pkl")
        if os.path.exists(posture_model_path) and os.path.exists(posture_scaler_path):
            models['posture_model'] = joblib.load(posture_model_path)
            models['posture_scaler'] = joblib.load(posture_scaler_path)
            logger.info("Modèle de posture chargé avec succès")
        
        # Modèle d'émotion
        emotion_model_path = os.path.join(MODELS_PATH, "emotion_model.pkl")
        emotion_scaler_path = os.path.join(MODELS_PATH, "emotion_scaler.pkl")
        if os.path.exists(emotion_model_path) and os.path.exists(emotion_scaler_path):
            models['emotion_model'] = joblib.load(emotion_model_path)
            models['emotion_scaler'] = joblib.load(emotion_scaler_path)
            logger.info("Modèle d'émotion chargé avec succès")
        
        # Autres modèles
        accessibility_path = os.path.join(MODELS_PATH, "accessibility.joblib")
        if os.path.exists(accessibility_path):
            models['accessibility'] = joblib.load(accessibility_path)
            logger.info("Modèle d'accessibilité chargé avec succès")
        
        sustainability_path = os.path.join(MODELS_PATH, "sustainability.joblib")
        if os.path.exists(sustainability_path):
            models['sustainability'] = joblib.load(sustainability_path)
            logger.info("Modèle de durabilité chargé avec succès")
        
        urban_density_path = os.path.join(MODELS_PATH, "urban_density.joblib")
        if os.path.exists(urban_density_path):
            models['urban_density'] = joblib.load(urban_density_path)
            logger.info("Modèle de densité urbaine chargé avec succès")
        
        return models
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {str(e)}")
        return {}

# Charger les modèles au démarrage
models = load_models()

# Classes que MobileNet SSD peut détecter
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Fonction pour traiter les images
def process_image(image_data):
    try:
        # Décodage de l'image base64
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Conversion en tableau numpy pour OpenCV
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # Si RGBA
            image_np = image_np[:, :, :3]
        
        return image_np
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
        return None

# Détection de personnes avec MobileNet SSD
def detect_persons(image):
    results = []
    
    if 'person_detector' not in models:
        return {"error": "Modèle de détection de personnes non disponible"}
    
    try:
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, 
                                     (300, 300), 127.5)
        
        # Passer le blob à travers le réseau et obtenir les détections
        models['person_detector'].setInput(blob)
        detections = models['person_detector'].forward()
        
        # Boucle sur les détections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filtre de confiance minimale
            if confidence > 0.5:
                # Extraire l'indice de classe de la détection
                idx = int(detections[0, 0, i, 1])
                
                # Si la classe est 'person' (15 dans la liste CLASSES)
                if CLASSES[idx] == "person":
                    # Calculer les coordonnées de la boîte englobante
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ajouter la détection aux résultats
                    results.append({
                        "type": "person",
                        "confidence": float(confidence),
                        "box": [int(startX), int(startY), int(endX), int(endY)]
                    })
        
        return {"detections": results}
    
    except Exception as e:
        logger.error(f"Erreur lors de la détection de personnes: {str(e)}")
        return {"error": str(e)}

# Analyse de posture
def analyze_posture(image, person_boxes):
    if 'posture_model' not in models or 'posture_scaler' not in models:
        return {"error": "Modèles de posture non disponibles"}
    
    try:
        results = []
        for box in person_boxes:
            startX, startY, endX, endY = box
            person_img = image[startY:endY, startX:endX]
            
            if person_img.size == 0:
                continue
            
            # Redimensionner l'image pour l'analyse
            person_img_resized = cv2.resize(person_img, (64, 128))
            
            # Extraire des caractéristiques (simplifié pour l'exemple)
            # En production, vous utiliseriez un extracteur de caractéristiques plus sophistiqué
            gray = cv2.cvtColor(person_img_resized, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
            
            # Normaliser avec le scaler
            scaled_features = models['posture_scaler'].transform([hist])
            
            # Prédire la posture
            posture = models['posture_model'].predict(scaled_features)[0]
            posture_proba = models['posture_model'].predict_proba(scaled_features)[0].max()
            
            results.append({
                "box": box,
                "posture": str(posture),
                "confidence": float(posture_proba)
            })
        
        return {"posture_analysis": results}
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de posture: {str(e)}")
        return {"error": str(e)}

# Analyse d'émotion
def analyze_emotion(image, person_boxes):
    if 'emotion_model' not in models or 'emotion_scaler' not in models:
        return {"error": "Modèles d'émotion non disponibles"}
    
    try:
        # Charger le détecteur de visage de OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        results = []
        for box in person_boxes:
            startX, startY, endX, endY = box
            person_img = image[startY:endY, startX:endX]
            
            if person_img.size == 0:
                continue
            
            # Détecter les visages dans la portion de personne
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img_resized = cv2.resize(face_img, (48, 48))
                
                # Extraire des caractéristiques du visage
                face_features = face_img_resized.flatten() / 255.0
                
                # Normaliser avec le scaler
                scaled_features = models['emotion_scaler'].transform([face_features])
                
                # Prédire l'émotion
                emotion = models['emotion_model'].predict(scaled_features)[0]
                emotion_proba = models['emotion_model'].predict_proba(scaled_features)[0].max()
                
                # Coordonnées absolues du visage
                abs_x, abs_y = x + startX, y + startY
                
                results.append({
                    "box": [int(abs_x), int(abs_y), int(abs_x + w), int(abs_y + h)],
                    "emotion": str(emotion),
                    "confidence": float(emotion_proba)
                })
        
        return {"emotion_analysis": results}
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse d'émotion: {str(e)}")
        return {"error": str(e)}

# Analyse de l'accessibilité
def analyze_accessibility(image):
    if 'accessibility' not in models:
        return {"error": "Modèle d'accessibilité non disponible"}
    
    try:
        # Réduire la taille de l'image pour l'analyse
        resized = cv2.resize(image, (224, 224))
        
        # Convertir en niveaux de gris pour simplifier
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Extraire quelques caractéristiques simples
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (224 * 224)
        
        # Créer un vecteur de caractéristiques (simplifié)
        features = np.array([[edge_density, np.mean(gray), np.std(gray)]])
        
        # Prédire l'accessibilité
        accessibility_score = models['accessibility'].predict(features)[0]
        
        return {
            "accessibility_score": float(accessibility_score),
            "accessibility_issues": ["stairs", "narrow_path"] if accessibility_score < 0.5 else []
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse d'accessibilité: {str(e)}")
        return {"error": str(e)}

# Analyse de la durabilité
def analyze_sustainability(image):
    if 'sustainability' not in models:
        return {"error": "Modèle de durabilité non disponible"}
    
    try:
        # Réduire la taille de l'image pour l'analyse
        resized = cv2.resize(image, (224, 224))
        
        # Convertir et extraire des caractéristiques de couleur
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 25, 25), (85, 255, 255))
        green_ratio = np.sum(green_mask) / (224 * 224 * 255)
        
        # Créer un vecteur de caractéristiques (simplifié)
        features = np.array([[green_ratio, np.mean(resized[:,:,1]), np.std(resized)]])
        
        # Prédire la durabilité
        sustainability_score = models['sustainability'].predict(features)[0]
        
        return {
            "sustainability_score": float(sustainability_score),
            "sustainable_elements": ["green_space", "solar_panels"] if sustainability_score > 0.6 else []
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de durabilité: {str(e)}")
        return {"error": str(e)}

# Analyse de la densité urbaine
def analyze_urban_density(image):
    if 'urban_density' not in models:
        return {"error": "Modèle de densité urbaine non disponible"}
    
    try:
        # Réduire la taille de l'image pour l'analyse
        resized = cv2.resize(image, (224, 224))
        
        # Détection des contours (bâtiments, structures)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Calculer la densité des contours
        edge_density = np.sum(edges) / (224 * 224)
        
        # Créer un vecteur de caractéristiques (simplifié)
        features = np.array([[edge_density, np.mean(gray), np.std(gray)]])
        
        # Prédire la densité urbaine
        density_score = models['urban_density'].predict(features)[0]
        
        return {
            "urban_density_score": float(density_score),
            "density_category": "high" if density_score > 0.7 else "medium" if density_score > 0.4 else "low"
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de densité urbaine: {str(e)}")
        return {"error": str(e)}

# Route principale d'analyse d'image
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Aucune image fournie"}), 400
        
        # Traiter l'image
        image = process_image(data['image'])
        if image is None:
            return jsonify({"error": "Impossible de traiter l'image"}), 400
        
        # Analyses demandées
        requested_analyses = data.get('analyses', ['persons'])
        
        results = {}
        
        # Détection de personnes (toujours exécutée pour les autres analyses)
        person_result = detect_persons(image)
        if 'persons' in requested_analyses:
            results.update({"persons": person_result})
        
        # Extraire les boîtes englobantes des personnes pour d'autres analyses
        person_boxes = []
        if 'detections' in person_result:
            person_boxes = [d['box'] for d in person_result['detections']]
        
        # Analyse de posture
        if 'posture' in requested_analyses and person_boxes:
            results.update({"posture": analyze_posture(image, person_boxes)})
        
        # Analyse d'émotion
        if 'emotion' in requested_analyses and person_boxes:
            results.update({"emotion": analyze_emotion(image, person_boxes)})
        
        # Analyse d'accessibilité
        if 'accessibility' in requested_analyses:
            results.update({"accessibility": analyze_accessibility(image)})
        
        # Analyse de durabilité
        if 'sustainability' in requested_analyses:
            results.update({"sustainability": analyze_sustainability(image)})
        
        # Analyse de densité urbaine
        if 'urban_density' in requested_analyses:
            results.update({"urban_density": analyze_urban_density(image)})
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Erreur générale lors de l'analyse: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Variables globales pour le streaming
streaming_active = False
streaming_thread = None
last_frame = None

# Démarrage du streaming
@app.route('/stream/start', methods=['POST'])
def start_stream():
    global streaming_active, streaming_thread
    
    if streaming_active:
        return jsonify({"message": "Le streaming est déjà actif"}), 400
    
    # Configuration du streaming
    data = request.json or {}
    interval = data.get('interval', 1)  # Intervalle en secondes entre les analyses
    analyses = data.get('analyses', ['persons', 'posture', 'emotion'])
    
    def stream_analysis():
        global streaming_active, last_frame
        
        while streaming_active:
            try:
                if last_frame is not None:
                    # Convertir le frame en image OpenCV
                    image = cv2.imdecode(np.frombuffer(last_frame, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Réaliser les analyses
                    results = {}
                    
                    # Détection de personnes
                    person_result = detect_persons(image)
                    results.update({"persons": person_result})
                    
                    # Extraire les boîtes englobantes des personnes
                    person_boxes = []
                    if 'detections' in person_result:
                        person_boxes = [d['box'] for d in person_result['detections']]
                    
                    # Autres analyses selon la configuration
                    if 'posture' in analyses and person_boxes:
                        results.update({"posture": analyze_posture(image, person_boxes)})
                    
                    if 'emotion' in analyses and person_boxes:
                        results.update({"emotion": analyze_emotion(image, person_boxes)})
                    
                    if 'accessibility' in analyses:
                        results.update({"accessibility": analyze_accessibility(image)})
                    
                    if 'sustainability' in analyses:
                        results.update({"sustainability": analyze_sustainability(image)})
                    
                    if 'urban_density' in analyses:
                        results.update({"urban_density": analyze_urban_density(image)})
                    
                    # Émettre les résultats via Socket.IO
                    socketio.emit('analysis_results', results)
            
            except Exception as e:
                logger.error(f"Erreur dans le thread de streaming: {str(e)}")
            
            # Attendre l'intervalle spécifié
            time.sleep(interval)
    
    # Démarrer le thread de streaming
    streaming_active = True
    streaming_thread = threading.Thread(target=stream_analysis)
    streaming_thread.daemon = True
    streaming_thread.start()
    
    return jsonify({"message": "Streaming démarré avec succès"})

# Arrêt du streaming
@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    global streaming_active, streaming_thread
    
    if not streaming_active:
        return jsonify({"message": "Aucun streaming n'est actif"}), 400
    
    streaming_active = False
    if streaming_thread:
        streaming_thread.join(timeout=1.0)
    
    return jsonify({"message": "Streaming arrêté avec succès"})

# Recevoir un frame pour le streaming
@socketio.on('frame')
def receive_frame(frame_data):
    global last_frame
    
    # Traiter la frame base64
    if isinstance(frame_data, str) and frame_data.startswith('data:image'):
        frame_data = frame_data.split(',')[1]
    
    last_frame = base64.b64decode(frame_data)

# Point de terminaison pour vérifier les modèles disponibles
@app.route('/models', methods=['GET'])
def get_models():
    available_models = {
        'person_detector': 'person_detector' in models,
        'posture': 'posture_model' in models and 'posture_scaler' in models,
        'emotion': 'emotion_model' in models and 'emotion_scaler' in models,
        'accessibility': 'accessibility' in models,
        'sustainability': 'sustainability' in models,
        'urban_density': 'urban_density' in models
    }
    
    return jsonify({
        "available_models": available_models,
        "total_models": sum(available_models.values())
    })

# Route pour tester l'API
@app.route('/test', methods=['GET'])
def test_api():
    return jsonify({
        "status": "ok",
        "message": "API d'analyse en temps réel opérationnelle",
        "version": "1.0.0"
    })

if __name__ == '__main__':
    logger.info("Démarrage de l'API d'analyse en temps réel...")
    socketio.run(app, host='0.0.0.0', port=5007, debug=True, allow_unsafe_werkzeug=True)


