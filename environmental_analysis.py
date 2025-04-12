import cv2
import numpy as np
from typing import Dict, Union

def analyze_environment(frame: np.ndarray) -> Dict[str, Union[float, None]]:
    """
    Analyse l'environnement à partir de l'image.
    
    Args:
        frame: Image courante (numpy array)
    
    Returns:
        Dictionnaire contenant:
            - brightness: Luminosité moyenne (0-100%)
            - estimated_temperature: Température estimée (simulation)
    """
    try:
        # Convertir en niveaux de gris pour calculer la luminosité
        if frame is None or len(frame.shape) < 2:
            return {
                'brightness': None,
                'estimated_temperature': None
            }
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculer la luminosité moyenne (0-255) et normaliser à 0-100%
        avg_brightness = np.mean(gray)
        brightness_percent = (avg_brightness / 255.0) * 100.0
        
        # Simuler une température basée sur la teinte de l'image
        # Ceci est une simulation simpliste et ne reflète pas la température réelle
        # Dans un vrai robot, nous utiliserions un capteur de température
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        
        # Simuler une température entre 15°C et 30°C basée sur la teinte
        # Les tons bleus sont plus "froids", les tons rouges/jaunes plus "chauds"
        estimated_temp = 15.0 + (avg_hue / 179.0) * 15.0
        
        return {
            'brightness': float(brightness_percent),
            'estimated_temperature': float(estimated_temp)
        }
    except Exception as e:
        print(f"Erreur lors de l'analyse environnementale: {e}")
        return {
            'brightness': None,
            'estimated_temperature': None
        }

