from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline
import numpy as np
import os
import json
from PIL import Image
import io
import time
import threading
import geopy.distance

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Variables globales pour le suivi en temps réel
latest_location = {"latitude": None, "longitude": None, "timestamp": None}
active_camera = "back"  # 'back' ou 'front'
sensor_history = {
    'accelerometer': [],
    'gyroscope': [],
    'gps': []
}
MAX_HISTORY = 100  # Nombre maximal d'entrées d'historique à conserver

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
        if person_objects and 'box' in person_objects[0] and person_objects[0]['box'] is not None:
            height_px = person_objects[0]['box']['ymax'] - person_objects[0]['box']['ymin']
            # Conversion arbitraire pixels->cm (à calibrer dans un système réel)
            height = int(height_px * 0.5)  # Conversion sécurisée
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

def calculate_speed(gps_data, prev_gps_data):
    """Calcule la vitesse basée sur deux points GPS"""
    if not gps_data or not prev_gps_data:
        return 0
    
    try:
        # Calculer la distance entre deux points
        coords_1 = (prev_gps_data.get('latitude'), prev_gps_data.get('longitude'))
        coords_2 = (gps_data.get('latitude'), gps_data.get('longitude'))
        
        # Vérifier que les coordonnées sont valides
        if None in coords_1 or None in coords_2:
            return 0
            
        distance = geopy.distance.geodesic(coords_1, coords_2).meters
        
        # Calculer le temps écoulé en secondes
        time_diff = (gps_data.get('timestamp', time.time()) - 
                    prev_gps_data.get('timestamp', time.time()))
        
        if time_diff > 0:
            speed = distance / time_diff  # m/s
            return round(speed * 3.6, 2)  # km/h
        return 0
    except Exception as e:
        print(f"Erreur de calcul de vitesse: {e}")
        return 0

def update_sensor_history(sensor_type, data):
    """Met à jour l'historique des capteurs avec horodatage"""
    global sensor_history
    
    if sensor_type not in sensor_history:
        sensor_history[sensor_type] = []
        
    # Ajouter un horodatage si non présent
    if 'timestamp' not in data:
        data['timestamp'] = time.time()
        
    sensor_history[sensor_type].append(data)
    
    # Limiter la taille de l'historique
    if len(sensor_history[sensor_type]) > MAX_HISTORY:
        sensor_history[sensor_type].pop(0)

@app.route('/switch/camera', methods=['POST'])
def switch_camera():
    """Permet de basculer entre caméra avant et arrière"""
    global active_camera
    
    data = request.json
    if 'camera' in data and data['camera'] in ['front', 'back']:
        active_camera = data['camera']
        return jsonify({"status": "success", "active_camera": active_camera})
    
    return jsonify({"status": "error", "message": "Camera type must be 'front' or 'back'"}), 400

@app.route('/sensor/current', methods=['GET'])
def get_current_sensor_data():
    """Renvoie les dernières données des capteurs"""
    current_data = {
        'gps': sensor_history['gps'][-1] if sensor_history['gps'] else None,
        'accelerometer': sensor_history['accelerometer'][-1] if sensor_history['accelerometer'] else None,
        'gyroscope': sensor_history['gyroscope'][-1] if sensor_history['gyroscope'] else None,
        'active_camera': active_camera
    }
    
    # Calculer la vitesse si possible
    if len(sensor_history['gps']) >= 2:
        current_data['speed'] = calculate_speed(
            sensor_history['gps'][-1],
            sensor_history['gps'][-2]
        )
    else:
        current_data['speed'] = 0
        
    return jsonify(current_data)

@app.route('/process/image', methods=['POST'])
def process_image():
    try:
        # Récupérer l'image
        file = request.files.get('image')
        if not file:
            return jsonify({"error": "No image provided"}), 400
        
        img = Image.open(file.stream)
        
        # Récupérer les données des capteurs et information sur la caméra
        camera_info = request.form.get('camera', 'back')
        global active_camera
        active_camera = camera_info  # Mettre à jour l'état de la caméra active
        
        sensor_data = {}
        for sensor in ['gps', 'accelerometer', 'gyroscope']:
            if sensor in request.form:
                try:
                    data = json.loads(request.form[sensor])
                    sensor_data[sensor] = data
                    
                    # Mettre à jour l'historique des capteurs
                    update_sensor_history(sensor, data)
                    
                    # Mettre à jour la position GPS globale
                    if sensor == 'gps' and 'latitude' in data and 'longitude' in data:
                        global latest_location
                        latest_location = {
                            'latitude': data['latitude'],
                            'longitude': data['longitude'],
                            'timestamp': time.time()
                        }
                except Exception as e:
                    print(f"Erreur lors du traitement des données {sensor}: {e}")
        
        # Détecter les objets
        objects = []
        if object_detector:
            try:
                results = object_detector(img)
                
                # Adapter le format de la réponse au modèle utilisé
                if isinstance(results, list) and results and isinstance(results[0], dict) and 'box' in results[0]:
                    # Format pour object-detection
                    objects = [
                        {
                            "label": item["label"],
                            "confidence": float(item["score"]),  # Assurer que c'est un float
                            "box": {
                                "xmin": float(item["box"]["xmin"]),
                                "ymin": float(item["box"]["ymin"]),
                                "xmax": float(item["box"]["xmax"]),
                                "ymax": float(item["box"]["ymax"])
                            }
                        } 
                        for item in results
                    ]
                else:
                    # Format pour image-classification
                    objects = [
                        {
                            "label": item["label"],
                            "confidence": float(item["score"]),  # Assurer que c'est un float
                            "box": None
                        } 
                        for item in results
                    ]
            except Exception as e:
                print(f"Erreur lors de la détection d'objets: {e}")
                objects = []  # Assurer que objects est toujours une liste
        
        # Analyser le mouvement
        analysis = analyze_movement(objects, sensor_data)
        
        # Calculer la vitesse basée sur les données GPS
        speed = None
        if 'gps' in sensor_data and len(sensor_history['gps']) >= 2:
            speed = calculate_speed(
                sensor_history['gps'][-1],
                sensor_history['gps'][-2]
            )
        
        # Construire la réponse
        response = {
            "objects": objects,
            "intention": analysis["intention"],
            "trajectory": analysis["trajectory"],
            "height": analysis["height"],
            "sensor_data": sensor_data,
            "active_camera": active_camera,
            "speed": speed,
            "timestamp": time.time()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Erreur serveur: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Installer geopy si nécessaire
    try:
        import geopy
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "geopy"])
        import geopy
    
    # Démarrer le serveur
    app.run(host='0.0.0.0', port=5007, threaded=True)

