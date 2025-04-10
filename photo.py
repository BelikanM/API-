# -*- coding: utf-8 -*-
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Initialiser Flask
app = Flask(__name__)
CORS(app)

# Charger le modèle Résolut pour reconnaître les plantes
model = models.resnet18(pretrained=True)  # Exemple avec ResNet18
model.eval()

# Transformation nécessaire pour l'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def process_image(image_data):
    """Transforme l'image base64 en un tensore lisible par le modèle."""
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = transform(image).unsqueeze(0)  # Ajouter une dimension
    return image

def analyze_image(image_data):
    try:
        # Charger et transformer l'image
        image_tensor = process_image(image_data)

        # Modèle PyTorch pour prédiction
        with torch.no_grad():
            output = model(image_tensor)
        
        # Obtenir les prédictions
        _, predicted_idx = torch.max(output, 1)
        predicted_idx = predicted_idx.item()

        # Exemple de mappage de prédictions
        # Ajoutez votre propre mappage si vous avez des catégories spécifiques aux plantes
        classes = ["Classe 0", "Classe 1", "Classe Plante Medicinale"]
        plant_name = classes[predicted_idx]
        details = f"Identifié comme {plant_name}."

        return {
            "isPlant": True, 
            "plantName": plant_name, 
            "details": details
        }
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'image : {e}")
        return {"isPlant": False}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json.get("imageBase64", "")
        image_data = base64.b64decode(data)
        result = analyze_image(image_data)
        return jsonify(result)
    except Exception as error:
        print(f"Erreur serveur : {error}")
        return jsonify({"error": "Erreur serveur"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)

