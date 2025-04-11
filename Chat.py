# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit

app = Flask(__name__)

# Configuration pour le serveur MySQL local
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://Belikan:Dieu19961991??!@127.0.0.1:3306/MSDOS'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

socketio = SocketIO(app)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    content = db.Column(db.String(200), nullable=False)
    media_url = db.Column(db.String(200))
    audio_url = db.Column(db.String(200))

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    message = Message(
        username=data['username'],
        content=data['content'],
        media_url=data.get('media_url'),
        audio_url=data.get('audio_url')
    )
    db.session.add(message)
    db.session.commit()
    socketio.emit('message', {
        'username': message.username,
        'content': message.content,
        'media_url': message.media_url,
        'audio_url': message.audio_url
    })
    return jsonify({'status': 'Message sent!'})

@socketio.on('message')
def handle_message(data):
    emit('message', data, broadcast=True)

if __name__ == '__main__':
    db.create_all()
    socketio.run(app, host='0.0.0.0', port=6000, debug=True)

