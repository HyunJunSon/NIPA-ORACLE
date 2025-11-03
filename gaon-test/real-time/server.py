# 비동기 방식

import io
import base64
import speech_recognition as sr
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import logging
import wave
import tempfile
import os
from pydub import AudioSegment


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Use gevent for async support (better compatibility than eventlet)
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")

@app.route("/")
def index():
    logger.info("Serving real-time.html")
    return render_template("real-time.html")

# 음성 데이터 수신
@socketio.on("audio_stream")
def handle_audio(data):
    try:
        logger.info(f"Received audio data from user: {data.get('user', 'Unknown')}")
        audio_bytes = base64.b64decode(data["audio"])
        
        # Create a temporary file to handle the audio processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(audio_bytes)
            temp_filename = temp_file.name
        
        try:
            # Convert the audio to WAV format using pydub
            audio = AudioSegment.from_file(temp_filename, format="webm")
            wav_filename = temp_filename.replace(".webm", ".wav")
            audio.export(wav_filename, format="wav")
            
            # Use speech recognition on the converted file
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_filename) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data, language="ko-KR")
                    logger.info(f"Recognized text: {text}")
                    # 실시간으로 텍스트 결과 전송
                    emit("text_result", {"user": data["user"], "text": text}, broadcast=True)
                except sr.UnknownValueError:
                    # 음성 인식 실패 시 처리
                    logger.warning("Could not understand audio")
                    emit("text_result", {"user": data["user"], "text": "[음성 인식 실패]"}, broadcast=True)
                except sr.RequestError as e:
                    # 인터넷 연결 문제 등 처리
                    logger.error(f"Request error: {e}")
                    emit("text_result", {"user": data["user"], "text": f"[요청 오류: {e}]"}, broadcast=True)
        finally:
            # Clean up temporary files
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            wav_filename = temp_filename.replace(".webm", ".wav")
            if os.path.exists(wav_filename):
                os.remove(wav_filename)
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        emit("text_result", {"user": data.get("user", "Unknown"), "text": f"[처리 오류: {e}]"}, broadcast=True)

# Handle connection events
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

if __name__ == "__main__":
    # For development only - use Gunicorn with eventlet workers in production
    logger.info("Starting async speech recognition app on port 5")
    socketio.run(app, host="0.0.0.0", port=5005, debug=False)