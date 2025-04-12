import joblib
from flask import Flask, request, jsonify
from geopy.distance import geodesic

app = Flask(__name__)

# Fonction pour charger les modèles en toute sécurité
def load_model(filename):
    try:
        return joblib.load(filename)
    except EOFError:
        print(f"Error loading model from {filename}. File may be corrupted.")
        return None

# Charger les modèles de machine learning
activity_model = load_model('activity_model.joblib')
intention_model = load_model('intention_model.joblib')
trajectory_model = load_model('trajectory_model.joblib')

@app.route('/activity/analyze', methods=['POST'])
def analyze_activity():
    """
    Analyse des données fournies pour suivre une activité physique (distance, activité, etc.).
    """
    data = request.json
    
    # Récupérer les données fournies
    start = data.get('start')  # {'lat': ..., 'lon': ...}
    end = data.get('end')      # {'lat': ..., 'lon': ...}
    motion_data = data.get('motion')  # Données du capteur de mouvement

    if not start or not end or not motion_data:
        return jsonify({'error': 'Invalid data'}), 400

    # Calcul de la distance entre les deux points
    start_point = (start['lat'], start['lon'])
    end_point = (end['lat'], end['lon'])
    distance = geodesic(start_point, end_point).kilometers

    # Analyse via les modèles (si disponibles)
    activity_analysis = (
        activity_model.predict([[start['lat'], start['lon'], end['lat'], end['lon']]]).tolist()
        if activity_model else 'Model unavailable'
    )
    intention_analysis = (
        intention_model.predict([[start['lat'], start['lon'], end['lat'], end['lon']]]).tolist()
        if intention_model else 'Model unavailable'
    )
    trajectory_analysis = (
        trajectory_model.predict([[start['lat'], start['lon'], end['lat'], end['lon']]]).tolist()
        if trajectory_model else 'Model unavailable'
    )

    # Structuration des résultats
    result = {
        'distance_km': distance,
        'activity_analysis': activity_analysis,
        'trajectory_analysis': trajectory_analysis,
        'intention_analysis': intention_analysis,
        'message': 'Activity analysis complete.'
    }

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True, port=5006)

