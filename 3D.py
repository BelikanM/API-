from flask import Flask
from flask_socketio import SocketIO
import base64
import io
from PIL import Image, ImageDraw
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # tu peux utiliser yolov8s.pt ou m/l selon la précision souhaitée

@socketio.on('frame')
def handle_frame(data):
    try:
        # Décoder l'image base64
        img_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(img_data)).convert('RGB')

        # Prédiction avec YOLOv8
        results = model(image)

        # Dessiner les boîtes avec PIL
        draw = ImageDraw.Draw(image)
        objets_detectes = set()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                objets_detectes.add(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                draw.text((x1, y1 - 10), label, fill='red')

        # Convertir l’image annotée en base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        socketio.emit('result', {
            'objects': list(objets_detectes),
            'image': encoded_image
        })

    except Exception as e:
        print(f"Erreur : {e}")
        socketio.emit('result', {'objects': ['Erreur traitement']})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5005)
