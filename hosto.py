from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

try:
    model = joblib.load("models/hospital_model.joblib")
except FileNotFoundError:
    print("Le fichier 'hospital_model.joblib' n'a pas été trouvé. Assurez-vous qu'il est correctement situé dans le répertoire 'models/'.")

@app.route("/predict_hospital", methods=["POST"])
def predict_hospital():
    try:
        data = request.json["hospitals"]
        locations = np.array([(hospital['latitude'], hospital['longitude']) for hospital in data])
        predictions = model.predict(locations)
        return jsonify({"clusters": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5006)

