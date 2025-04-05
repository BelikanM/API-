from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Utiliser SQLite comme base de données
# Stockera les données dans un fichier nommé sql_app.db
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'sql_app.db')}"

# Créer le moteur SQLAlchemy
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Création d'une classe de base pour les modèles
Base = declarative_base()

# Création d'une classe Session pour gérer les sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Fonction pour obtenir une session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

