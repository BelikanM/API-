ls
clear
ls
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
# Installer les dépendances principales
pip install django
pip install channels       # Pour WebSockets (chat et mise à jour temps réel)
pip install django-rest-framework  # Pour les API
pip install daphne        # Serveur ASGI pour WebSockets
pip install pillow        # Pour traitement d'images
pip install leaflet       # Pour intégration avec Leaflet.js
pip install pandas numpy scikit-learn  # Pour les fonctionnalités d'IA
pip install psycopg2-binary  # Si vous utilisez PostgreSQL (recommandé)
pip install redis         # Pour le cache et les channels layers
# Créer le projet
django-admin startproject config .
apt update
apt install python3-pip
clear
ls
pip3 --version
pip3 install django djangorestframework daphne pillow pandas numpy scikit-learn psycopg2-binary redis
pip3 install --upgrade setuptools
clear
ls
pip3 show setuptools
pip list
pip3 install django djangorestframework daphne pillow pandas numpy scikit-learn psycopg2-binary redis
clear
ls
pip list
python3 manage.py runserver 0.0.0.0:8080
pwd
clear
ls
django-admin startproject geonetwork
cd geonetwork
clear
ls
# Création du dossier apps et ses sous-dossiers
mkdir -p apps/accounts apps/geotracking apps/chat apps/ai_pathfinding
# Création des dossiers statiques
mkdir -p static/js static/css static/media
# Création des dossiers templates
mkdir -p templates/accounts templates/map templates/chat
clear
ls
cd apps
ls
cd ..
ls
cd geonetwork
ls
nano settings.py
apt install
apt install nano
clear
ls
nano settings.py
pip install django channels channels-redis djangorestframework pillow
clear
ls
cd ..
nano requirements.txt
ls
nano requirements.txt
cd geonetwork
ls
clear
ls
nano asgi.py
cd ..
python manage.py runserver 0.0.0.0:8080
python3 manage.py runserver 0.0.0.0:8080
python3 manage.py runserver 0.0.0.0:8000
clear
python3 manage.py migrate
clear
ls
python3 manage.py runserver 0.0.0.0:8000
clear
ls
cd geonetwork
ls
nano settings.py
cd ..
clear
ls
python3 manage.py runserver 0.0.0.0:8000
clear
ls
cd ..
ls
clear
clear
ls
cd geonetwork
ls
clear
ls
python3 manage.py createsuperuser
python3 manage.py startapp users       # Pour la gestion des utilisateurs
python3 manage.py startapp geolocation # Pour la géolocalisation
python3 manage.py startapp posts       # Pour les publications/partages
python3 manage.py startapp chat        # Pour la messagerie
clear
ls
cd geonetwork
ls
nano settings.py
cd ..
clear
ls
cd users
ls
nano models.py
ls
cd ..
clear
ls
cd geolocation
ls
nano models.py
cd ..
clear
ls
cd posts
ls
nano models.py
cd ..
clear
ls
cd chat
ls
nano models.py
cd ..
clear
ls
cd geonetwork
ls
nano settings.py
nano urls.py
cd ..
clear
ls
cd users
ls
cd apps
clear
ls
cd ..
ls
cd apps
ls
cd ..
ls
cd geolocation
ls
nano urls.py
cd ..
clear
ls
cd users
ls
nano urls.py
cd .
cd ..
clear
ls
cd posts
ls
nano urls.py
cd ..
cd chat
ls
nano urls.py
clear
ls
cd ..
ls
cd geonetwork
ls
nano asgi.py
cd ..
clear
ls
cd chat
ls
nano routing.py
ls
nano consumers.py
cd ..
clear
ls
cd geolocation
nano routine.py
nano consumers.py
clear
ls
mkdir templates
ls
cd templates
ls
mkdir geolocation
ls
cd geolocation
ls
nano map.html
clear
ls
cd ..
ls
cd ..
ls
nano views.py
nano serializers.py
clear
ls
cd ..
python3 manage.py makemigrations
python3 manage.py migrate
clear
ls
cd ..
ls
cd geonetwork
ls
clear
ls
cd users
ls
clear
ls
rm -f views.py
nano views.py
cd ..
ls
python3 manage.py makemigrations
python3 manage.py migrate
clear
ls
cd users
ls
rm -f models.py
ls
nano models.py
cd ..
ls
python manage.py makemigrations
python manage.py migrate
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver
clear
ls
cd users
ls
rm -f models.py
nano models.py
rm -f views.py
nano views.py
cd ..
clear
ls
python manage.py makemigrations users
python manage.py migrate
# Supprimer la base de données SQLite (si vous utilisez SQLite)
rm db.sqlite3
# Recréer les migrations depuis zéro
python3 manage.py makemigrations
python3 manage.py migrate
# Créer un superutilisateur
python3 manage.py createsuperuser
clear
ls
cd users
ls
rm -f models.py
nano models.py
rm -f views.py
nano views.py
rm -f admin.py
nano admin.py
nano forms.py 
ls
rm -f urls.py
nano urls.py
cd ..
clear
ls
# Créer les migrations pour tous les modèles, y compris le nouveau modèle UserLocation
python3 manage.py makemigrations
# Appliquer les migrations
python3 manage.py migrate
# Créer un superutilisateur pour accéder à l'interface d'administration
python3 manage.py createsuperuser
clear
ls
cd users
ls
rm -f views.py
nano views.py
# Créer les migrations pour tous les modèles, y compris le nouveau modèle UserLocation
python3 manage.py makemigrations
# Appliquer les migrations
python3 manage.py migrate
# Créer un superutilisateur pour accéder à l'interface d'administration
python3 manage.py createsuperuser
cd ..
clear
ls
python3 manage.py createsuperuser
clear
ls
python3 manage.py runserver 0.0.0.0:8000
cleae
clear
ls
cd users
ls
rm -f views.py
nano views.py
rm -f forms.py
nano forms.py
cd ..
python3 manage.py runserver 0.0.0.0:8000
clear
ls
cd users
ls
rm -f models.py
nano models.py
cd ..
python3 manage.py makemigrations
python3 manage.py migrate
clear
ls
cd users
ls
rm models.py
nano models
nano models.py
cd ..
clear
python3 manage.py makemigrations
python3 manage.py migrate
clear
ls
cd ..
clear
ls
rm -r geonetwork
ls
pip list
python -m django --version
python3 -m django --version
django-admin startproject mysite
clear
ls
cd mysite
ls
cd mysite
ls
python manage.py runserver
apt install python
clear
ls
python manage.py runserver
cd ...
ls
clear
ls
python manage.py runserver
ls
cd ..
ls
python manage.py runserver
nano manage.py
python manage.py runserver
clear
ls
python3 manage.py runserver
clear
ls
python3 manage.py startapp chat
clear
ls
cd chat
clear
ls
nano views.py
nano urls.py
clear
ls
nano NavFooter.py
mkdir -p chat/templates/chat/components
clear
ls
rm -f NavFooter.py
ls
clear
ls
cd chat
ls
cd ..
clear
ls
rm -r chat
ls
cd ..
clear
ls
mkdir -p chat/templates/chat/components
clear
ls
cd chat
ls
clear
ls
nano NavFooter.py
cd templates
ls
cd chat
ls
cd components
ls
nano nav_footer.html
cd ..
cd 
.
cd ..
clear
ls
cd sdcard
ls
cd ..
clear
ls
cd home
ls
cd ..
clear
ls
cd mysite
ls
clear
ls
cd chat
ls
clear
ls
rm urls.py
nano urls.py
nano views.py
cd templates
clear
ls
cd chat
ls
nano base.html
cd ..
ls
cleat
clear
ls
mkdir -p chat/templatetags
touch chat/templatetags/__init__.py
clear
ls
cd chat
ls
cd cd templatetags
cd templatetags
ls
nano nav_footer.py
cd ..
cldar
clear
ls
cd ..
clear
ls
cd chat
ls
clear
ls
cd ..
clear
ls
cd .
cd ..
ls
cd ..
ls
python3 manage.py runserver
clear
ls
cd chat
ls
python3 apps.py
ls
cd ..
clear
cd ..
clear
ls
rm -r mysite
ls
clear
ls
django-admin startproject Gab
cd Gabls
clear
ls
cd Gab
ls
Gab
cd Gab
ls
cd ..
clear
ls
cd ..
ls
rm -r Gab
ls
django-admin startproject geogab .
ls
clear
ls
cd geogab
cd ..
clear
ls
cd ..
clear
ls
clear
ls
rm -r geogab
rm -r manage.py
clear
ls
pip install flask flask-sqlalchemy flask-wtf
clear
ls
pip install sqlalchemy
pip install mysqlclient  # Pour Python 3
# OU
pip install mysql-connector-python  # Alternative
clear
ls
pip list
clear
ls
nano main.py
ls
rm -f main.py
mkdir dox
ls
cd dox
ls
clear
ls
nano main.py
nano crud.py
nano models.py
nano database.py
clear
ls
python main.py
python3 main.py
clear
ls
pip3 install mysqlclient
# OU si cela échoue
apt-get install python3-dev default-libmysqlclient-dev build-essential
pip3 install pymysql
clear
ls
rm -f database.py
nano database.py
rm -f main.py
nano main.py
# Vérifier si MySQL est en cours d'exécution
systemctl status mysql
# Tester la connexion
mysql -u Belikan -p -h 127.0.0.1
# Entrez votre mot de passe lorsque vous y êtes invité
# Puis dans le client MySQL
mysql> USE MSDOS;
mysql> SHOW TABLES;
mysql> exit;
clear
ls
rm -f database.py
nano database.py
rm -f main.py
nano main.py
rm -f models.py
nano models.py
python3 main.py
clear
ls
python3 main.py
clear
