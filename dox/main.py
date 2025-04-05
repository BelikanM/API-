# -*- coding: utf-8 -*-
from database import engine, get_db
import models
import crud

# Creation de toutes les tables definies dans les modeles
models.Base.metadata.create_all(bind=engine)

def main():
    # Exemple d'utilisation
    db = next(get_db())
    
    try:
        # Creer un utilisateur
        new_user = crud.create_user(
            db=db, 
            username="utilisateur_test", 
            email="test@example.com", 
            password="motdepasse123"
        )
        print(f"Nouvel utilisateur cree: {new_user.username}")
        
        # Recuperer tous les utilisateurs
        users = crud.get_users(db)
        print(f"Nombre d'utilisateurs: {len(users)}")
        
        # Autres operations...
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()

