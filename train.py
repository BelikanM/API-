import os
import sys
import joblib
import numpy as np
import pandas as pd
import pdfplumber
import logging
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MapIA-Train')

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data', 'uploads', 'model_cache', 'reports']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    logger.info(f"Directories created: {dirs}")

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def parse_city_data(text):
    """Parse extracted text to identify city data and danger zones."""
    data = {
        'city_name': 'Unknown',
        'coordinates': {'lat': 0, 'lon': 0},
        'dangerous_areas': []
    }
    
    # Simple parsing logic - in a real app, this would use more robust NLP
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        if "CITY:" in line:
            data['city_name'] = line.split("CITY:")[1].strip()
            
        elif "COORDINATES:" in line:
            try:
                coords = line.split("COORDINATES:")[1].strip().split(',')
                data['coordinates']['lat'] = float(coords[0])
                data['coordinates']['lon'] = float(coords[1])
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse coordinates: {line}, error: {e}")
                
        elif "DANGER ZONE:" in line:
            try:
                # Expected format: "DANGER ZONE: name;lat;lon;radius;level;description"
                parts = line.split("DANGER ZONE:")[1].strip().split(';')
                if len(parts) >= 6:
                    area = {
                        'name': parts[0].strip(),
                        'lat': float(parts[1]),
                        'lon': float(parts[2]),
                        'radius': float(parts[3]),
                        'level': parts[4].strip(),
                        'description': parts                        [5].strip()
                    }
                    data['dangerous_areas'].append(area)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse danger zone info: {line}, error: {e}")
    
    return data

def simulate_danger_zone_training(city_name, lat, lon):
    """Simulates training a danger zone model for a city."""
    features = []
    labels = []
    
    # Create a grid of points around the city center
    grid_size = 0.01  # roughly 1km
    grid_points = 20
    
    for i in range(-grid_points, grid_points + 1):
        for j in range(-grid_points, grid_points + 1):
            point_lat = lat + i * grid_size / grid_points
            point_lon = lon + j * grid_size / grid_points
            
            # Randomly mark some points as in a danger zone
            in_danger_zone = np.random.rand() > 0.7
            
            # Add feature and label
            features.append([point_lat, point_lon])
            labels.append(1 if in_danger_zone else 0)
    
    if not features:
        logger.warning(f"No features extracted for training {city_name}")
        return False
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join('model_cache', f'danger_zone_model_{city_name.lower()}.joblib')
    joblib.dump(model, model_path)
    
    # Validate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model trained for {city_name} with accuracy: {accuracy}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return True

def train(file_paths):
    """Train models on the provided file paths."""
    for file_path in file_paths:
        text = extract_text_from_pdf(file_path)
        city_data = parse_city_data(text)
        
        if city_data['city_name'] != 'Unknown':
            logger.info(f"Training danger zone model for {city_data['city_name']}")
            success = simulate_danger_zone_training(
                city_data['city_name'],
                city_data['coordinates']['lat'],
                city_data['coordinates']['lon']
            )
            
            if success:
                logger.info(f"Successfully trained danger zone model for {city_data['city_name']}")
            else:
                logger.error(f"Failed to train danger zone model for {city_data['city_name']}")
        else:
            logger.warning(f"No valid city data found in {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train danger zone models using PDF files.')
    parser.add_argument(
        'files',
        nargs='+',
        help='PDF files containing city data'
    )
    
    args = parser.parse_args()
    
    setup_directories()
    
    logger.info("Starting training process...")
    train(args.files)
    logger.info("Training process completed.")

