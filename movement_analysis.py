import cv2
import numpy as np
import math
from typing import Dict, List, Optional, Any, Union

# Variables pour suivre la position précédente
prev_positions = {}
prev_time = None

def analyze_movement(frame: np.ndarray, person_objects: List[Dict[str, Any]]) -> Dict[str, Union[float, str, int, None]]:
    """
    Analyse le mouvement des personnes détectées dans l'image.
    
    Args:
        frame: Image courante (numpy array)
        person_objects: Liste des personnes détectées avec leurs coordonnées
    
    Returns:
        Dictionnaire contenant:
            - speed: Vitesse estimée en cm/s
            - direction: Direction du mouvement (gauche, droite, haut, bas)
            - person_height: Taille estimée de la personne en cm
    """
    global prev_positions, prev_time
    
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    height, width = frame.shape[:2]
    
    # Valeurs par défaut
    result = {
        'speed': None,
        'direction': 'inconnu',
        'person_height': None
    }
    
    try:
        # Calculer la taille de la personne (si une personne est détectée)
        if person_objects and 'box' in person_objects[0] and person_objects[0]['box'] is not None:
            # Vérifier que la box est un dictionnaire avec les bonnes clés
            box = person_objects[0]['box']
            if isinstance(box, dict) and all(key in box for key in ['ymin', 'ymax']):
                try:
                    height_px = float(box['ymax']) - float(box['ymin'])
                    # Conversion arbitraire pixels->cm (à calibrer selon votre caméra)
                    result['person_height'] = int(height_px * 0.5)
                except (TypeError, ValueError):
                    result['person_height'] = None
        
        # Calculer la vitesse et la direction (si une personne est détectée)
        if person_objects and 'box' in person_objects[0] and person_objects[0]['box'] is not None:
            box = person_objects[0]['box']
            
            if isinstance(box, dict) and all(key in box for key in ['xmin', 'ymin', 'xmax', 'ymax']):
                # Calculer le centre de la boîte
                try:
                    center_x = (float(box['xmin']) + float(box['xmax'])) / 2
                    center_y = (float(box['ymin']) + float(box['ymax'])) / 2
                    
                    person_id = 0  # Pour simplifier, on utilise un seul ID
                    
                    # Si nous avons une position précédente pour cette personne
                    if prev_time is not None and person_id in prev_positions:
                        prev_pos = prev_positions[person_id]
                        time_diff = current_time - prev_time
                        
                        if time_diff > 0:
                            # Calculer la distance en pixels
                            dx = center_x - prev_pos[0]
                            dy = center_y - prev_pos[1]
                            distance_px = math.sqrt(dx**2 + dy**2)
                            
                            # Convertir la vitesse en cm/s (facteur arbitraire, à calibrer)
                            result['speed'] = float(distance_px * 0.5 / time_diff)
                            
                            # Déterminer la direction
                            if abs(dx) > abs(dy):  # Mouvement horizontal dominant
                                if dx > 0:
                                    result['direction'] = 'droite'
                                else:
                                    result['direction'] = 'gauche'
                            else:  # Mouvement vertical dominant
                                if dy > 0:
                                    result['direction'] = 'bas'
                                else:
                                    result['direction'] = 'haut'
                    
                    # Mettre à jour la position précédente
                    prev_positions[person_id] = (center_x, center_y)
                except (TypeError, ValueError):
                    # En cas d'erreur, on laisse les valeurs par défaut
                    pass
        
        # Mettre à jour le temps précédent
        prev_time = current_time
        
    except Exception as e:
        print(f"Erreur dans analyze_movement: {e}")
    
    return result

