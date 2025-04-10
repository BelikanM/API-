import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Supposons que nous ayons des données fictives
X = np.random.rand(100, 5)  # Exemples de caractéristiques (e.g., GPS, capteurs, etc.)
y = np.random.choice(['Walking', 'Running', 'Driving'], 100)  # Exemple de classes

# Séparation en ensembles de train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarder le modèle
joblib.dump(model, 'activity_model.joblib')  # Assurez-vous que ce fichier sera accessible par vision.py

# Tester le modèle
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

