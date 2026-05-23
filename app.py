#!/usr/bin/env python3
"""
Audible Transcriber Web UI
Dark-themed Flask app with real-time progress via WebSocket
"""
import os
import sys
import json
import subprocess
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import dotenv
import whisper
from audible import Authenticator, Client

dotenv.load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'audible-transcriber-secret'
socketio = SocketIO(app, async_mode='eventlet')

AUTH_FILE = os.getenv("AUDIBLE_AUTH_FILE", "auth.json")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./transcripts")
ACTIVATION_BYTES = os.getenv("AUDIBLE_ACTIVATION_BYTES")

# Store job status
jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth', methods=['POST'])
def auth():
    username = request.json.get('username')
    password = request.json.get('password')
    locale = request.json.get('locale', 'us')
    
    try:
        auth = Authenticator.from_login(
            username=username,
            password=password,
            locale=locale
        )
        auth.to_file(AUTH_FILE)
        
        # Get activation bytes
        from audible.auth import get_activation_bytes
        activation_bytes = get_activation_bytes(auth)
        
        with open('.env', 'a') as f:
            f.write(f'\nAUDIBLE_ACTIVATION_BYTES={activation_bytes}\n')
        
        # Reload env
        dotenv.load_dotenv()
        
        socketio.emit('auth_status', {'status': 'success', 'message': 'Authenticated!'})
        return jsonify({'success': True})
    except Exception as e:
        socketio.emit('auth_status', {'status': 'error', 'message': str(e)})
        return jsonify({'success': False, 'error': str(e)})

@app.route('/list-books')
def list_books():
    if not os.path.exists(AUTH_FILE):
        return jsonify({'error': 'Not authenticated'})
    
    try:
        auth = Authenticator.from_file(AUTH_FILE)
        client = Client(auth=auth)
        
        books = []
        page = 1
        while len(books) < 50:  # Limit to 50 for UI
            response = client.get("library/books", params={"page": page, "per_page": 50})
            items = response.get("items", [])
            if not items:
                break
            for book in items:
                books.append({
                    'asin': book.get('asin'),
                    'title': book.get('title'),
                    'authors': ', '.join([a.get('name') for a in book.get('authors', [])])
                })
            page += 1
        
        return jsonify({'books': books})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    asin = request.json.get('asin')
    model_size = request.json.get('model', 'base')
    fmt = request.json.get('format', 'txt')
    
    if not asin:
        return jsonify({'error': 'ASIN required'})
    
    job_id = f"job_{len(jobs)+1}"
    jobs[job_id] = {'status': 'running', 'progress': 0}
    
    def run_pipeline():
        try:
            socketio.emit('progress', {'job_id': job_id, 'progress': 10, 'message': 'Initializing...'})
            
            auth = Authenticator.from_file(AUTH_FILE)
            client = Client(auth=auth)
            
            # Download
            socketio.emit('progress', {'job_id': job_id, 'progress': 20, 'message': 'Downloading AAX...'})
            aax_path = f"./temp_{asin}.aax"
            product = client.get_product(asin)
            content_url = product.get("content_url")
            
            import requests
            response = requests.get(content_url, stream=True)
            response.raise_for_status()
            with open(aax_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Decrypt
            socketio.emit('progress', {'job_id': job_id, 'progress': 40, 'message': 'Decrypting...'})
            wav_path = f"./temp_{asin}.wav"
            cmd = ["ffmpeg", "-y", "-activation_bytes", ACTIVATION_BYTES,
                   "-i", aax_path, "-acodec", "pcm_s16le", "-ar", "16000", wav_path]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Transcribe
            socketio.emit('progress', {'job_id': job_id, 'progress': 60, 'message': 'Loading Whisper model...'})
            model = whisper.load_model(model_size)
            socketio.emit('progress', {'job_id': job_id, 'progress': 70, 'message': 'Transcribing...'})
            result = model.transcribe(wav_path, fp16=False)
            
            # Save
            socketio.emit('progress', {'job_id': job_id, 'progress': 90, 'message': 'Saving transcript...'})
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            output_base = os.path.join(OUTPUT_DIR, asin)
            
            if fmt == 'txt':
                output_file = Path(output_base).with_suffix('.txt')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
            elif fmt == 'json':
                output_file = Path(output_base).with_suffix('.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Cleanup
            os.remove(aax_path)
            os.remove(wav_path)
            
            socketio.emit('progress', {'job_id': job_id, 'progress': 100, 'message': 'Done!', 'output': str(output_file)})
            jobs[job_id]['status'] = 'complete'
            
        except Exception as e:
            socketio.emit('progress', {'job_id': job_id, 'progress': -1, 'message': f'Error: {str(e)}'})
            jobs[job_id]['status'] = 'error'
    
    thread = threading.Thread(target=run_pipeline)
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/transcripts/<path:filename>')
def download_transcript(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
