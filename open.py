from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import base64
import time
import os
import urllib.request

app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Fonction pour télécharger les fichiers Haar Cascade depuis GitHub
def download_cascade_file(filename):
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    url = base_url + filename
    try:
        print(f"Téléchargement de {filename} depuis GitHub...")
        urllib.request.urlretrieve(url, filename)
        print(f"Téléchargement de {filename} réussi!")
        return True
    except Exception as e:
        print(f"Erreur lors du téléchargement de {filename}: {e}")
        return False

# Chargement des modèles
try:
    # Modèle de détection de personnes avec Haar Cascade
    body_cascade_file = 'haarcascade_fullbody.xml'
    if not os.path.exists(body_cascade_file):
        print(f"Fichier non trouvé: {body_cascade_file}")
        if download_cascade_file(body_cascade_file):
            body_cascade = cv2.CascadeClassifier(body_cascade_file)
        else:
            body_cascade = None
    else:
        body_cascade = cv2.CascadeClassifier(body_cascade_file)
    
    # Modèle de détection de personnes (création d'un modèle factice si non disponible)
    try:
        person_detection_model = joblib.load('person_detection_model.joblib')
    except:
        print("Modèle person_detection_model.joblib non trouvé, création d'un modèle factice")
        from sklearn.ensemble import RandomForestClassifier
        person_detection_model = RandomForestClassifier()
    
    # Modèle d'analyse d'intention (création d'un modèle factice si non disponible)
    try:
        intention_model = joblib.load('intention_model.joblib')
    except:
        print("Modèle intention_model.joblib non trouvé, création d'un modèle factice")
        from sklearn.ensemble import RandomForestClassifier
        intention_model = RandomForestClassifier()
        # Ajouter des classes factices pour éviter les erreurs
        intention_model.classes_ = np.array(['approche', 'éloignement', 'statique'])
    
    # Pour la détection de visage
    face_cascade_file = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_file):
        print(f"Fichier non trouvé: {face_cascade_file}")
        if download_cascade_file(face_cascade_file):
            face_cascade = cv2.CascadeClassifier(face_cascade_file)
        else:
            face_cascade = None
    else:
        face_cascade = cv2.CascadeClassifier(face_cascade_file)
    
    print("Modèles chargés avec succès")
except Exception as e:
    print(f"Erreur lors du chargement des modèles: {e}")
    body_cascade = None
    face_cascade = None

# Historique des mouvements pour le tracking
movement_history = []

# Fonction pour détecter les points clés du corps avec OpenCV
def detect_body_keypoints(image):
    # Cette fonction est une simplification par rapport à MediaPipe
    # On détecte simplement les contours du corps humain
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # On prend les contours les plus grands (qui sont probablement le corps)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Extraire des points clés simplifiés (haut, milieu, bas)
    keypoints = []
    if contours and len(contours) > 0:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        # Simulation de points clés simplifiés
        top = (x + w//2, y)  # Tête (approximative)
        middle = (x + w//2, y + h//2)  # Milieu du corps
        bottom = (x + w//2, y + h)  # Bas du corps
        
        keypoints = [
            {"x": top[0]/image.shape[1], "y": top[1]/image.shape[0], "visibility": 1.0},
            {"x": middle[0]/image.shape[1], "y": middle[1]/image.shape[0], "visibility": 1.0},
            {"x": bottom[0]/image.shape[1], "y": bottom[1]/image.shape[0], "visibility": 1.0}
        ]
    
    return keypoints, contours

# Fonction pour détecter les mains (simplifiée)
def detect_hands(image):
    # Version simplifiée pour la détection des mains
    
    # Convertir en HSV pour la détection de couleur de peau
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Plage de couleur pour la peau
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Créer un masque pour la couleur de peau
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Opérations morphologiques pour améliorer le masque
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Trouver les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours par taille pour ne garder que ceux qui ressemblent à des mains
    hand_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000 and area < 30000:  # Ajuster ces valeurs selon vos besoins
            hand_contours.append(cnt)
    
    return hand_contours

@app.route('/analyze', methods=['POST'])
def analyze_image():
    global movement_history
    
    if 'image' not in request.json:
        return jsonify({'error': 'Aucune image fournie'}), 400
    
    try:
        # Décodage de l'image base64
        encoded_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Résultats à retourner
        results = {
            'person_count': 0,
            'postures': [],
            'emotions': [],
            'face_detected': False,
            'hands_detected': False,
            'movement_analysis': {},
            'timestamp': time.time()
        }
        
        # 1. Détection de personnes avec Haar Cascade (si disponible)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bodies = []
        if body_cascade is not None:
            bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
        results['person_count'] = len(bodies)
        
        # Dessiner les rectangles autour des personnes détectées
        body_positions = []
        for (x, y, w, h) in bodies:
            body_positions.append((x, y, w, h))
        
        # 2. Analyse de posture avec notre fonction alternative
        keypoints, body_contours = detect_body_keypoints(img)
        
        if keypoints:
            results['postures'].append({
                'landmarks': keypoints,
                'confidence': 0.7  # Valeur arbitraire
            })
            
            # Analyse de la posture (debout, assis, etc.)
            if keypoints[0]['y'] < 0.5:  # Si la tête est dans la moitié supérieure
                posture_type = "debout"
            else:
                posture_type = "assis ou accroupi"
                
            results['postures'][-1]['type'] = posture_type
        
        # 3. Détection de visage avec OpenCV Haar Cascade (si disponible)
        faces = []
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            results['face_detected'] = True
            results['emotions'] = ["neutre"]  # Placeholder pour l'analyse d'émotions
        
        # 4. Détection des mains
        hand_contours = detect_hands(img)
        if hand_contours:
            results['hands_detected'] = True
            results['hand_count'] = len(hand_contours)
        
        # 5. Analyse des mouvements
        # Ajouter la position actuelle à l'historique
        if len(body_positions) > 0:
            movement_history.append({
                'positions': body_positions,
                'timestamp': results['timestamp']
            })
            
            # Garder seulement les 10 dernières positions
            if len(movement_history) > 10:
                movement_history = movement_history[-10:]
            
            # Analyser le mouvement si nous avons assez d'historique
            if len(movement_history) > 1:
                # Calculer la direction et la vitesse du mouvement
                prev_pos = movement_history[-2]['positions']
                curr_pos = movement_history[-1]['positions']
                
                if len(prev_pos) > 0 and len(curr_pos) > 0:
                    # Prendre la première personne détectée pour simplifier
                    prev_x, prev_y = prev_pos[0][0], prev_pos[0][1]
                    curr_x, curr_y = curr_pos[0][0], curr_pos[0][1]
                    
                    dx = curr_x - prev_x
                    dy = curr_y - prev_y
                    
                    # Direction du mouvement
                    if abs(dx) > abs(dy):
                        direction = "droite" if dx > 0 else "gauche"
                    else:
                        direction = "bas" if dy > 0 else "haut"
                    
                    # Vitesse (distance euclidienne)
                    speed = np.sqrt(dx**2 + dy**2)
                    
                    results['movement_analysis'] = {
                        'direction': direction,
                        'speed': float(speed),
                        'dx': float(dx),
                        'dy': float(dy)
                    }
                    
                    # Utilisation du modèle d'intention (si disponible)
                    try:
                        # Créer un vecteur de features pour le modèle d'intention
                        features = [dx, dy, speed, 
                                    curr_pos[0][2], curr_pos[0][3]]  # width, height de la boîte
                        
                        # Prédire l'intention
                        intention_probas = intention_model.predict_proba([features])[0]
                        intention_labels = intention_model.classes_
                        
                        # Prendre les 3 intentions les plus probables
                        top_indices = intention_probas.argsort()[-3:][::-1]
                        intentions = [{"intention": intention_labels[i], 
                                      "probability": float(intention_probas[i])} 
                                    for i in top_indices]
                        
                        results['intention_analysis'] = intentions
                    except Exception as e:
                        print(f"Erreur lors de l'analyse d'intention: {e}")
        
        # Préparer l'image annotée
        annotated_img = img.copy()
        
        # Dessiner les contours du corps si disponibles
        if body_contours:
            cv2.drawContours(annotated_img, body_contours, -1, (0, 255, 0), 2)
        
        # Dessiner les points clés si disponibles
        for kp in keypoints:
            x, y = int(kp['x'] * img.shape[1]), int(kp['y'] * img.shape[0])
            cv2.circle(annotated_img, (x, y), 5, (0, 0, 255), -1)
        
        # Dessiner les rectangles de détection de personnes
        for (x, y, w, h) in bodies:
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Dessiner les rectangles de visage
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Dessiner les contours des mains
        cv2.drawContours(annotated_img, hand_contours, -1, (0, 255, 255), 2)
        
        # Dessiner le mouvement
        if 'movement_analysis' in results and results['movement_analysis']:
            for i in range(1, len(movement_history)):
                if len(movement_history[i-1]['positions']) > 0 and len(movement_history[i]['positions']) > 0:
                    prev_x, prev_y = movement_history[i-1]['positions'][0][0], movement_history[i-1]['positions'][0][1]
                    curr_x, curr_y = movement_history[i]['positions'][0][0], movement_history[i]['positions'][0][1]
                    cv2.line(annotated_img, (prev_x, prev_y), (curr_x, curr_y), (255, 0, 0), 2)
        
        # Convertir l'image annotée en base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        results['annotated_image'] = f"data:image/jpeg;base64,{annotated_image_base64}"
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007, debug=True)

