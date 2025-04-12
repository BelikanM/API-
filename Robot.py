# Robot.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
import cv2
import time
from datetime import datetime
import threading
import json

app = Flask(__name__)
CORS(app)

# Chargement des modèles
try:
    motion_model = load('motion_model.joblib')
    activity_model = load('activity_model.joblib')
    trajectory_model = load('trajectory_model.joblib')
except Exception as e:
    print(f"Erreur lors du chargement des modèles: {e}")

class VideoAnalyzer:
    def __init__(self):
        self.is_running = False
        self.frame_buffer = []
        self.analysis_results = {}
        
    def start_analysis(self, video_source=0):
        self.is_running = True
        self.cap = cv2.VideoCapture(video_source)
        threading.Thread(target=self._analyze_stream).start()
        
    def stop_analysis(self):
        self.is_running = False
        if hasattr(self, 'cap'):
            self.cap.release()
            
    def _analyze_stream(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Analyse du mouvement
            motion_data = self._analyze_motion(frame)
            
            # Analyse de l'activité
            activity_data = self._analyze_activity(frame)
            
            # Analyse de la trajectoire
            trajectory_data = self._analyze_trajectory(frame)
            
            # Stockage des résultats
            timestamp = datetime.now().isoformat()
            self.analysis_results[timestamp] = {
                'motion': motion_data,
                'activity': activity_data,
                'trajectory': trajectory_data
            }
            
            time.sleep(0.1)  # Pause pour éviter une surcharge
            
    def _analyze_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = np.array([
            np.mean(gray),
            np.std(gray),
            np.max(gray) - np.min(gray)
        ]).reshape(1, -1)
        
        return float(motion_model.predict(features)[0])
    
    def _analyze_activity(self, frame):
        features = self._extract_activity_features(frame)
        return float(activity_model.predict(features.reshape(1, -1))[0])
    
    def _analyze_trajectory(self, frame):
        features = self._extract_trajectory_features(frame)
        return float(trajectory_model.predict(features.reshape(1, -1))[0])
    
    def _extract_activity_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.array([
            np.mean(gray),
            np.std(gray),
            np.percentile(gray, 75),
            np.percentile(gray, 25)
        ])
    
    def _extract_trajectory_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.array([
            np.sum(edges),
            np.mean(edges),
            np.std(edges)
        ])

# Instance globale de l'analyseur vidéo
video_analyzer = VideoAnalyzer()

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    try:
        video_analyzer.start_analysis()
        return jsonify({'status': 'success', 'message': 'Analyse vidéo démarrée'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    try:
        video_analyzer.stop_analysis()
        return jsonify({'status': 'success', 'message': 'Analyse vidéo arrêtée'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_analysis', methods=['GET'])
def get_analysis():
    try:
        return jsonify({'status': 'success', 'data': video_analyzer.analysis_results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/clear_analysis', methods=['POST'])
def clear_analysis():
    try:
        video_analyzer.analysis_results.clear()
        return jsonify({'status': 'success', 'message': 'Données d\'analyse effacées'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007, debug=True)

