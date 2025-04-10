from flask import Flask, request, jsonify, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import joblib
import pandas as pd
import numpy as np
import pdfplumber
import time
import threading
from werkzeug.utils import secure_filename
import logging
import json
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_CACHE = 'model_cache'
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_CACHE, exist_ok=True)

# Setup logger
logging.basicConfig(
    filename='mapIA.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mapIA')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'secret!'  # Change this in production
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Try to load existing models
try:
    activity_model = joblib.load('activity_model.joblib')
    height_model = joblib.load('height_model.joblib')
    danger_zone_model = joblib.load(os.path.join(MODEL_CACHE, 'danger_zone_model.joblib'))
    logger.info("Models loaded successfully")
except Exception as e:
    logger.warning(f"Could not load models: {e}")
    activity_model = None
    height_model = None
    danger_zone_model = None

# Cache for city data to avoid repeated processing
city_data_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
    return text

def parse_city_data(text):
    # Simple parsing logic - in a real app, this would be more sophisticated
    data = {
        'dangerous_areas': [],
        'city_name': '',
        'coordinates': {'lat': 0, 'lon': 0}
    }
    
    lines = text.split('\n')
    for line in lines:
        if 'CITY:' in line:
            data['city_name'] = line.split('CITY:')[1].strip()
        elif 'COORDINATES:' in line:
            coords = line.split('COORDINATES:')[1].strip().split(',')
            if len(coords) >= 2:
                try:
                    data['coordinates']['lat'] = float(coords[0])
                    data['coordinates']['lon'] = float(coords[1])
                except ValueError:
                    pass
        elif 'DANGER ZONE:' in line:
            danger_info = line.split('DANGER ZONE:')[1].strip()
            # Format should be "name;latitude;longitude;radius;level;description"
            parts = danger_info.split(';')
            if len(parts) >= 6:
                try:
                    area = {
                        'name': parts[0],
                        'lat': float(parts[1]),
                        'lon': float(parts[2]),
                        'radius': float(parts[3]),
                        'level': parts[4],
                        'description': parts[5]
                    }
                    data['dangerous_areas'].append(area)
                except (ValueError, IndexError):
                    pass
    
    return data

def predict_danger_zones(city_name, lat, lon):
    # If we have a trained model, use it
    if danger_zone_model is not None:
        # This is a simplified example - in a real application,
        # you would need actual features of the area
        features = np.array([[lat, lon]])
        prediction = danger_zone_model.predict_proba(features)[0][1]
        
        if prediction > 0.5:
            return [{
                'lat': lat,
                'lon': lon,
                'radius': 500,  # meters
                'level': 'Medium',
                'description': f"Predicted danger zone in {city_name}"
            }]
    
    # Fallback to simulated data if no model or prediction
    # In a real app, you would have more sophisticated logic here
    return [{
        'lat': lat + np.random.random() * 0.01 - 0.005,
        'lon': lon + np.random.random() * 0.01 - 0.005,
        'radius': np.random.randint(200, 800),
        'level': np.random.choice(['Low', 'Medium', 'High']),
        'description': f"Simulated danger zone in {city_name}"
    } for _ in range(3)]

def train_model_on_files(files):
    global danger_zone_model
    
    features = []
    labels = []
    
    for file_path in files:
        text = extract_text_from_pdf(file_path)
        city_data = parse_city_data(text)
        
        # Extract base coordinates
        base_lat = city_data['coordinates']['lat']
        base_lon = city_data['coordinates']['lon']
        
        # Create a grid of points around the city center
        grid_size = 0.01  # roughly 1km
        grid_points = 20
        
        for i in range(-grid_points, grid_points + 1):
            for j in range(-grid_points, grid_points + 1):
                point_lat = base_lat + i * grid_size / grid_points
                point_lon = base_lon + j * grid_size / grid_points
                
                # Determine if point is in a danger zone
                in_danger_zone = False
                for area in city_data['dangerous_areas']:
                    dlat = point_lat - area['lat']
                    dlon = point_lon - area['lon']
                    distance_squared = dlat**2 + dlon**2
                    radius_degrees = area['radius'] / 111000  # approximate conversion from meters to degrees
                    if distance_squared <= radius_degrees**2:
                        in_danger_zone = True
                        break
                
                # Add feature and label
                features.append([point_lat, point_lon])
                labels.append(1 if in_danger_zone else 0)
    
    if not features:
        logger.warning("No features extracted for training")
        return False
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, os.path.join(MODEL_CACHE, 'danger_zone_model.joblib'))
    
    # Update the global model
    danger_zone_model = model
    
    accuracy = model.score(X_test, y_test)
    logger.info(f"Model trained with accuracy: {accuracy}")
    
    return True

def simulate_training_progress(file_paths):
    """Simulates training progress updates via WebSocket."""
    try:
        total_steps = 10
        for i in range(total_steps + 1):
            progress = int(i * 100 / total_steps)
            socketio.emit('training_progress', {'progress': progress})
            time.sleep(1)  # Simulate processing time
        
        # Actual training (could be long, so we do it after showing UI progress)
        success = train_model_on_files(file_paths)
        
        if success:
            # Send a completed message
            socketio.emit('training_progress', {'progress': 100, 'message': 'Training completed successfully!'})
            
            # Detect new danger zones and broadcast to clients
            # For demo, we'll just generate random zones
            zones = []
            for _ in range(3):
                zones.append({
                    'lat': 48.8566 + np.random.random() * 0.02 - 0.01,  # Around Paris
                    'lon': 2.3522 + np.random.random() * 0.02 - 0.01,
                    'radius': np.random.randint(200, 500),
                    'level': np.random.choice(['Low', 'Medium', 'High']),
                    'description': 'Newly detected danger zone'
                })
            socketio.emit('danger_zones_update', {'zones': zones})
        else:
            socketio.emit('training_progress', {
                'progress': 100, 
                'error': True,
                'message': 'Training failed. Check server logs.'
            })
            
    except Exception as e:
        logger.error(f"Error in training thread: {e}")
        socketio.emit('training_progress', {
            'progress': 100, 
            'error': True,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/search_city', methods=['POST'])
def search_city():
    data = request.json
    city_name = data.get('city', '')
    
    if not city_name:
        return jsonify({'error': 'City name is required'}), 400
    
    # Check if we have cached data for this city
    if city_name in city_data_cache:
        return jsonify(city_data_cache[city_name])
    
    # Simplified city coordinates lookup - in a real app, use a geocoding service
    city_coords = {
        'paris': {'lat': 48.8566, 'lon': 2.3522},
        'london': {'lat': 51.5074, 'lon': -0.1278},
        'new york': {'lat': 40.7128, 'lon': -74.0060},
        'tokyo': {'lat': 35.6762, 'lon': 139.6503},
        'berlin': {'lat': 52.5200, 'lon': 13.4050},
        'moscow': {'lat': 55.7558, 'lon': 37.6173},
        'beijing': {'lat': 39.9042, 'lon': 116.4074},
    }
    
    city_name_lower = city_name.lower()
    if city_name_lower in city_coords:
        lat = city_coords[city_name_lower]['lat']
        lon = city_coords[city_name_lower]['lon']
    else:
        # Default to a random location if city not found
        lat = np.random.uniform(35, 55)
        lon = np.random.uniform(-10, 20)
    
    # Get danger zones for this city
    danger_zones = predict_danger_zones(city_name, lat, lon)
    
    response_data = {
        'city': city_name,
        'lat': lat,
        'lon': lon,
        'danger_zones': danger_zones
    }
    
    # Cache the response
    city_data_cache[city_name] = response_data
    
    return jsonify(response_data)

@app.route('/api/upload_training_data', methods=['POST'])
def upload_training_data():
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'No files part in the request'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': 'No files selected'}), 400
    
    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_files.append(file_path)
    
    if not saved_files:
        return jsonify({
            'success': False, 
            'message': 'No valid files uploaded. Please upload PDF files only.'
        }), 400
    
    return jsonify({
        'success': True, 
        'message': f'Successfully uploaded {len(saved_files)} file(s)', 
        'files': saved_files
    })

@app.route('/api/train_model', methods=['POST'])
def train_model_endpoint():
    data = request.json
    files = data.get('files', [])
    
    if not files:
        return jsonify({'success': False, 'message': 'No files provided for training'}), 400
    
    # Start training in a separate thread to not block the main thread
    threading.Thread(target=simulate_training_progress, args=(files,)).start()
    
    return jsonify({
        'success': True, 
        'message': 'Training started. You will receive progress updates.'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Please provide a message'}), 400
    
    # Simple NLP to detect city names - in a real app, use a more sophisticated approach
    common_cities = [
        'paris', 'london', 'new york', 'tokyo', 'berlin', 'moscow', 'beijing',
        'madrid', 'rome', 'sydney', 'cairo', 'mumbai', 'rio', 'delhi', 'istanbul'
    ]
    
    city_found = None
    for city in common_cities:
        if city.lower() in user_message.lower():
            city_found = city
            break
    
    if "danger" in user_message.lower() and "zone" in user_message.lower():
        response = "Je peux vous aider à identifier les zones dangereuses dans une ville. Veuillez préciser la ville qui vous intéresse."
    elif "train" in user_message.lower() or "entraîner" in user_message.lower() or "pdf" in user_message.lower():
        response = "Pour entraîner le modèle, veuillez télécharger des fichiers PDF contenant des données sur les zones dangereuses de la ville."
    elif "help" in user_message.lower() or "aide" in user_message.lower() or "comment" in user_message.lower():
        response = "Vous pouvez rechercher une ville en tapant son nom, télécharger des PDF pour entraîner le modèle, ou me poser des questions sur les zones dangereuses."
    elif city_found:
        response = f"Je recherche les informations pour {city_found}. Veuillez patienter pendant que j'analyse les zones dangereuses."
    else:
        response = "Je ne suis pas sûr de comprendre. Vous pouvez me demander d'identifier les zones dangereuses d'une ville spécifique ou télécharger des PDF pour entraîner mon modèle."
    
    return jsonify({'response': response, 'city': city_found})

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

if __name__ == '__main__':
    logger.info("Starting MapIA service")
    socketio.run(app, host='0.0.0.0', port=5006, debug=True)

