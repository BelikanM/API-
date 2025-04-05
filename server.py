#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, abort, make_response
import numpy as np
import math
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Modèles préentraînés (simulés)
URBAN_DENSITY_MODEL = None
ACCESSIBILITY_MODEL = None

def initialize_models():
    """Initialisation des modèles d'IA (simulation)"""
    global URBAN_DENSITY_MODEL, ACCESSIBILITY_MODEL
    
    X_dummy = np.random.rand(100, 5)  # 5 caractéristiques simulées
    y_dummy = np.random.randint(0, 10, 100)  # Scores de densité
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)
    
    URBAN_DENSITY_MODEL = RandomForestClassifier(n_estimators=10, random_state=42)
    URBAN_DENSITY_MODEL.fit(X_train, y_train)
    
    ACCESSIBILITY_MODEL = RandomForestClassifier(n_estimators=10, random_state=42)
    ACCESSIBILITY_MODEL.fit(X_train, y_train)
    
    logger.info("Modèles d'IA initialisés")

def calculate_area(coords):
    """Calcule l'aire d'un polygone"""
    if len(coords) < 3:
        return 0

    # Calcul de l'aire en utilisant la formule du polygone
    area = 0.0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0

def identify_urban_patterns(buildings, roads, water, landuse):
    patterns = []
    
    building_count = len(buildings)
    
    if building_count > 50:
        patterns.append("Zone densément bâtie")
    elif building_count > 20:
        patterns.append("Zone moyennement bâtie")
    else:
        patterns.append("Zone peu bâtie")
    
    building_types = {}
    for building in buildings:
        b_type = building['info'].get('type', 'unknown')
        building_types[b_type] = building_types.get(b_type, 0) + 1
    
    residential_count = sum(building_types.get(t, 0) for t in ['residential', 'apartments', 'house', 'detached'])
    if residential_count > building_count * 0.6:
        patterns.append("Quartier principalement résidentiel")
    
    commercial_count = sum(1 for b in buildings if b['info'].get('amenity') in ['shop', 'restaurant', 'cafe', 'bar', 'supermarket'])
    if commercial_count > building_count * 0.3:
        patterns.append("Zone commerciale importante")
    
    if len(roads) > 20:
        patterns.append("Réseau routier dense")
    elif len(roads) > 7:
        patterns.append("Réseau routier standard")
    else:
        patterns.append("Zone peu desservie par les routes")
    
    if len(water) > 0:
        patterns.append("Proximité de plans d'eau")
    
    landuse_types = {}
    for area in landuse:
        lu_type = area['info'].get('type', 'unknown')
        landuse_types[lu_type] = landuse_types.get(lu_type, 0) + 1
    
    if landuse_types.get('forest', 0) > 0 or landuse_types.get('park', 0) > 0:
        patterns.append("Présence d'espaces verts")

    return patterns

def generate_recommendations(buildings, roads, water, landuse, patterns):
    recommendations = []
    
    building_count = len(buildings)
    road_count = len(roads)
    
    if building_count > 30 and road_count < 10:
        recommendations.append("Cette zone pourrait bénéficier d'une amélioration des infrastructures routières pour la mobilité.")
    
    if "Zone peu desservie par les routes" in patterns:
        recommendations.append("Envisager de développer le réseau de transport pour améliorer l'accessibilité.")
    
    has_green_space = any(area['info'].get('type') in ['forest', 'park', 'grass'] for area in landuse)
    if not has_green_space and building_count > 20:
        recommendations.append("Cette zone urbaine pourrait bénéficier de davantage d'espaces verts.")
    
    amenities = set()
    for building in buildings:
        if building['info'].get('amenity'):
            amenities.add(building['info'].get('amenity'))
    
    if len(amenities) < 3 and building_count > 20:
        recommendations.append("Diversifier les services et équipements pourrait améliorer la qualité de vie dans ce quartier.")
    
    if len(water) > 0:
        recommendations.append("Valoriser les accès aux plans d'eau pour améliorer l'attrait et la qualité de vie.")
    
    if len(recommendations) < 2:
        recommendations.append("Effectuer une analyse plus détaillée de la mobilité urbaine pourrait révéler d'autres opportunités d'amélioration.")
    if len(recommendations) < 3 and "Zone densément bâtie" in patterns:
        recommendations.append("Considérer des initiatives de verdissement urbain pour contrebalancer la densité bâtie.")
    
    return recommendations[:4]

def create_heatmap(buildings, roads, center, radius=800):
    center_lat, center_lng = center
    points = []
    step = 50
    steps_lat = radius / 111320
    steps_lng = radius / (111320 * math.cos(math.radians(center_lat)))
    
    for i in range(-20, 21):
        for j in range(-20, 21):
            lat = center_lat + (i * steps_lat/20)
            lng = center_lng + (j * steps_lng/20)
            
            density_value = 0
            
            for building in buildings:
                building_center = building['info']['center']
                dx = (lat - building_center[0]) * 111320
                dy = (lng - building_center[1]) * (111320 * math.cos(math.radians(center_lat)))
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 200:
                    density_value += 10 * (1 / (1 + distance/50))
            
            for road in roads:
                road_center = road['info']['center']
                dx = (lat - road_center[0]) * 111320
                dy = (lng - road_center[1]) * (111320 * math.cos(math.radians(center_lat)))
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 100:
                    density_value += 5 * (1 / (1 + distance/30))
            
            density_value = min(10, density_value)
            
            points.append([lat, lng, density_value])
    
    return points

def analyze_urban_data(lat, lon, buildings, roads, water, landuse):
    results = {
        "timestamp": datetime.now().isoformat(),
        "location": [lat, lon],
        "stats": {},
        "patterns": [],
        "recommendations": [],
        "density_heatmap": []
    }
    
    results["stats"]["building_count"] = len(buildings)
    results["stats"]["road_count"] = len(roads)
    results["stats"]["water_count"] = len(water)
    results["stats"]["landuse_count"] = len(landuse)
    
    total_building_area = sum(calculate_area(building['coords']) for building in buildings)
    results["stats"]["total_building_area_m2"] = round(total_building_area, 2)
    
    patterns = identify_urban_patterns(buildings, roads, water, landuse)
    results["patterns"] = patterns
    
    recommendations = generate_recommendations(buildings, roads, water, landuse, patterns)
    results["recommendations"] = recommendations
    
    if len(buildings) > 100:
        urban_density_score = min(10, len(buildings) / 20)
    else:
        urban_density_score = len(buildings) / 20
    results["stats"]["urban_density_score"] = round(urban_density_score, 1)
    
    if len(roads) > 0:
        accessibility_score = min(10, len(roads) / 3)
    else:
        accessibility_score = 0
    results["stats"]["accessibility_score"] = round(accessibility_score, 1)
    
    radius = 800
    heatmap_data = create_heatmap(buildings, roads, [lat, lon], radius)
    results["density_heatmap"] = heatmap_data
    
    return results

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method != 'POST':
        return jsonify({'error': 'Méthode non autorisée'}), 405
    
    try:
        content = request.json
        if not content:
            raise ValueError("Aucune donnée envoyée")
        
        location = content.get('location', [0, 0])
        buildings = content.get('buildings', [])
        roads = content.get('roads', [])
        water = content.get('water', [])
        landuse = content.get('landUse', [])
        
        if not location or not isinstance(location, list) or len(location) != 2:
            raise ValueError("Paramètre 'location' invalide ou manquant")
        
        analysis_results = analyze_urban_data(location[0], location[1], buildings, roads, water, landuse)
        
        return jsonify(analysis_results), 200
    
    except ValueError as ve:
        logger.error(f"Erreur de validation des données: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Erreur interne du serveur: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response

if __name__ == '__main__':
    initialize_models()
    app.run(host='0.0.0.0', port=5000, debug=True)

