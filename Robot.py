from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline
import numpy as np
import os
import json
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Charge le modèle de détection d'objets (utilise Hugging Face)
try:
    # Modèle plus léger pour éviter les problèmes de mémoire
    object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    print("Modèle de détection d'objets chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    # Fallback plus léger si nécessaire
    try:
        object_detector = pipeline("image-classification", model="google/vit-base-patch16-224")
        print("Modèle de classification d'images chargé avec succès (fallback)")
    except Exception as e:
        print(f"Erreur de fallback: {e}")
        object_detector = None

# Fonction pour analyser l'intention et la trajectoire à partir de la position des objets
def analyze_movement(objects, sensor_data):
    # Logique simplifiée pour déterminer l'intention et la trajectoire
    # Dans un système réel, utilisez les données des capteurs et l'IA pour une analyse plus précise
    
    if not objects:
        return {"intention": None, "trajectory": None, "height": None}
    
    # Estimer la trajectoire en fonction des objets détectés
    if any(obj['label'] in ['person', 'human'] for obj in objects):
        intention = "Mouvement humain détecté"
        
        # Utiliser les données du gyroscope pour estimer la trajectoire
        if 'gyroscope' in sensor_data:
            gyro = sensor_data['gyroscope']
            if abs(float(gyro.get('beta', 0))) > 10:
                trajectory = "Déplacement vertical"
            elif abs(float(gyro.get('gamma', 0))) > 10:
                trajectory = "Déplacement latéral"
            else:
                trajectory = "Stationnaire"
        else:
            trajectory = "Indéterminé"
        
        # Estimation grossière de la hauteur basée sur la taille relative des objets
        # Dans un système réel, utilisez la vision 3D/stéréo ou des capteurs LiDAR
        person_objects = [obj for obj in objects if obj['label'] in ['person', 'human']]
        if person_objects:
            height_px = person_objects[0]['box']['ymax'] - person_objects[0]['box']['ymin']
            # Conversion arbitraire pixels->cm (à calibrer dans un système réel)
            height = int(height_px * 0.5)
        else:
            height = None
    else:
        intention = "Objet inanimé"
        trajectory = "Stationnaire"
        height = None
        
    return {
        "intention": intention,
        "trajectory": trajectory,
        "height": height
    }

@app.route('/process/image', methods=['POST'])
def process_image():
    try:
        # Récupérer l'image
        file = request.files.get('image')
        if not file:
            return jsonify({"error": "No image provided"}), 400
        
        img = Image.open(file.stream)
        
        # Récupérer les données des capteurs
        sensor_data = {}
        for sensor in ['gps', 'accelerometer', 'gyroscope']:
            if sensor in request.form:
                try:
                    sensor_data[sensor] = json.loads(request.form[sensor])
                except Exception as e:
                    print(f"Erreur lors du traitement des données {sensor}: {e}")
        
        # Détecter les objets
        objects = []
        if object_detector:
            try:
                results = object_detector(img)
                
                # Adapter le format de la réponse au modèle utilisé
                if isinstance(results, list) and results and 'box' in results[0]:
                    # Format pour object-detection
                    objects = [
                        {
                            "label": item["label"],
                            "confidence": item["score"],
                            "box": item["box"]
                        } 
                        for item in results
                    ]
                else:
                    # Format pour image-classification
                    objects = [
                        {
                            "label": item["label"],
                            "confidence": item["score"],
                            "box": None
                        } 
                        for item in results
                    ]
            except Exception as e:
                print(f"Erreur lors de la détection d'objets: {e}")
        
        # Analyser le mouvement
        analysis = analyze_movement(objects, sensor_data)
        
        # Construire la réponse
        response = {
            "objects": objects,
            "intention": analysis["intention"],
            "trajectory": analysis["trajectory"],
            "height": analysis["height"],
            "sensor_data": sensor_data
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Erreur serveur: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)

