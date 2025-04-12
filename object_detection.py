import cv2
import numpy as np
from typing import List, Dict, Any

# Charger le modèle YOLOv3 pré-entraîné
# En production, ces chemins seraient à adapter
MODEL_CONFIG = "yolov3.cfg"
MODEL_WEIGHTS = "yolov3.weights"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Classes que YOLO peut détecter (format COCO)
OBJECT_CLASSES = [
    "personne", "vélo", "voiture", "moto", "avion", "bus", "train", "camion", 
    "bateau", "feu de circulation", "borne incendie", "panneau stop", "parcmètre", 
    "banc", "oiseau", "chat", "chien", "cheval", "mouton", "vache", "éléphant", 
    "ours", "zèbre", "girafe", "sac à dos", "parapluie", "sac à main", "cravate", 
    "valise", "frisbee", "skis", "snowboard", "ballon de sport", "cerf-volant", 
    "batte de baseball", "gant de baseball", "skateboard", "planche de surf", 
    "raquette de tennis", "bouteille", "verre à vin", "tasse", "fourchette", 
    "couteau", "cuillère", "bol", "banane", "pomme", "sandwich", "orange", 
    "brocoli", "carotte", "hot-dog", "pizza", "donut", "gâteau", "chaise", 
    "canapé", "plante en pot", "lit", "table à manger", "toilettes", "télévision", 
    "ordinateur portable", "souris", "télécommande", "clavier", "téléphone portable", 
    "four micro-ondes", "four", "grille-pain", "évier", "réfrigérateur", "livre", 
    "horloge", "vase", "ciseaux", "ours en peluche", "sèche-cheveux", "brosse à dents"
]

# Mapping pour la classification personnalisée
OBJECT_TYPE_MAPPING = {
    'person': [0],  # Indice "personne"
    'car': [2, 5, 6, 7],  # Indices "voiture", "bus", "train", "camion"
    'chair': [56],  # Indice "chaise"
    'bottle': [39],  # Indice "bouteille"
    'laptop': [63]  # Indice "ordinateur portable"
}

# Fonction pour initialiser le modèle de détection
def initialize_model():
    try:
        net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
        
        # Utiliser CUDA si disponible
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # Obtenir les noms des couches de sortie
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        return net, output_layers
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle: {e}")
        # En cas d'erreur avec CUDA, essayer avec CPU
        try:
            net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            
            return net, output_layers
        except Exception as e2:
            print(f"Erreur lors de l'initialisation du modèle sur CPU: {e2}")
            return None, None

# Variable pour stocker le modèle chargé
model = None
output_layers = None

def detect_objects_in_frame(frame, object_type='all'):
    """
    Détecte les objets dans une image.
    
    Args:
        frame: Image à analyser (numpy array)
        object_type: Type d'objet à détecter ('all' ou un type spécifique)
    
    Returns:
        Liste des objets détectés avec leurs coordonnées et confiance
    """
    global model, output_layers
    
    # Initialiser le modèle si ce n'est pas déjà fait
    if model is None:
        model, output_layers = initialize_model()
        if model is None:
            return []
    
    try:
        height, width, channels = frame.shape
        
        # Préparer l'image pour le réseau
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        model.setInput(blob)
        
        # Obtenir les détections
        outputs = model.forward(output_layers)
        
        # Préparer les conteneurs pour les résultats
        class_ids = []
        confidences = []
        boxes = []
        
        # Analyser les détections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    # Calculer les coordonnées de la boîte
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Coordonnées des coins de la boîte
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Filtrer par type d'objet si spécifié
                    if object_type != 'all':
                        if class_id not in OBJECT_TYPE_MAPPING.get(object_type, []):
                            continue
                    
                    # Ajouter aux listes
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Appliquer la suppression non-maximale
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        # Préparer les résultats
        detected_objects = []
        if len(indexes) > 0:
            indexes = indexes.flatten()
            for i in indexes:
                box = boxes[i]
                x, y, w, h = box
                
                # Créer un dictionnaire pour l'objet détecté
                object_data = {
                    'class_id': class_ids[i],
                    'class_name': OBJECT_CLASSES[class_ids[i]],
                    'confidence': confidences[i],
                    'box': {
                        'xmin': x,
                        'ymin': y,
                        'xmax': x + w,
                        'ymax': y + h,
                        'width': w,
                        'height': h
                    }
                }
                detected_objects.append(object_data)
        
        return detected_objects
        
    except Exception as e:
        print(f"Erreur lors de la détection d'objets: {e}")
        return []

