# Naver Clova Speech API 기반 음성 인식 서버

import io
import base64
import json
import requests
import tempfile
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import logging
from pydub import AudioSegment


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")

# Naver API 설정 - 실제 사용시 환경변수나 config 파일에서 관리해야 합니다
# 네이버 클로바 스피치 API 사용을 위해 실제 client_id와 client_secret이 필요합니다
# 이 정보는 네이버 클라우드 플랫폼에서 애플리케이션 등록 후 확인 가능합니다
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_CLIENT_SECRET")

@app.route("/")
def index():
    logger.info("Serving real-time-naver.html")
    return render_template("real-time-naver.html")

@app.route("/real-time-naver")
def index_naver():
    logger.info("Serving real-time-naver.html")
    return render_template("real-time-naver.html")

def naver_stt(audio_bytes):
    """Naver Clova Speech API를 사용한 음성 인식"""
    
    # API 엔드포인트
    url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"  # 한국어
    
    # 헤더 설정
    headers = {
        "Content-Type": "application/octet-stream",
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    
    try:
        # API 호출
        response = requests.post(url, data=audio_bytes, headers=headers)
        res_code = response.status_code
        
        if res_code == 200:
            # 성공적인 응답
            result = response.json()
            return result.get("text", "")
        else:
            logger.error(f"Naver API Error: {res_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Naver API Request Error: {e}")
        return None

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
            # Convert the audio to FLAC format (Naver API가 권장하는 포맷)
            audio = AudioSegment.from_file(temp_filename, format="webm")
            logger.info(f"Original audio: duration={len(audio)}ms, frame_rate={audio.frame_rate}, channels={audio.channels}")
            
            # Naver API 권장 포맷으로 변환: 16kHz, 단일 채널, FLAC
            # 하지만 FLAC은 브라우저에서 잘 지원되지 않기 때문에 WAV로 변환
            audio = audio.set_frame_rate(16000).set_channels(1)  # Naver API 권장: 16kHz, 단일 채널
            wav_filename = temp_filename.replace(".webm", ".wav")
            audio.export(wav_filename, format="wav")
            
            # WAV 파일을 바이트로 읽기
            with open(wav_filename, "rb") as f:
                audio_data = f.read()
            
            # Naver STT API 호출
            recognized_text = naver_stt(audio_data)
            
            if recognized_text is not None:
                logger.info(f"Recognized text: {recognized_text}")
                emit("text_result", {"user": data["user"], "text": recognized_text}, broadcast=True)
            else:
                logger.warning("Could not recognize audio with Naver API")
                emit("text_result", {"user": data["user"], "text": "[음성 인식 실패 - 네이버 API 오류]"}, broadcast=True)
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
    logger.info("Starting Naver Clova Speech API based speech recognition app on port 5007")
    socketio.run(app, host="0.0.0.0", port=5007, debug=False)