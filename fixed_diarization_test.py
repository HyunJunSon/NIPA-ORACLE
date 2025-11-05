import os
from google.cloud import speech

# Google Cloud 인증 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/hyunjunson/study/python/nipa-oracle/gaon-service-account.json"

client = speech.SpeechClient()

# GCS URI 사용
audio = speech.RecognitionAudio(uri="gs://gaon-cloud-data/sound-raw-data/sample/1040.mp3")

# 발화자 구분 옵션 (더 넓은 범위로 설정)
diarization_config = speech.SpeakerDiarizationConfig(
    enable_speaker_diarization=True,
    min_speaker_count=1,  # 최소값을 1로 낮춤
    max_speaker_count=6,  # 최대값을 높여서 더 많은 화자 감지 가능
)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.MP3,
    # sample_rate_hertz 제거 - 자동 감지하도록 함
    language_code="ko-KR",
    diarization_config=diarization_config,
    enable_automatic_punctuation=True,
    # 채널 관련 설정 제거 (화자 분할과 충돌)
)

print("Waiting for operation to complete...")
operation = client.long_running_recognize(config=config, audio=audio)
response = operation.result(timeout=300)

# 결과 처리
result = response.results[-1]
words_info = result.alternatives[0].words

# 화자별로 그룹화해서 출력
output_file = "transcription_output_fixed.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    current_speaker = None
    current_sentence = []
    
    for word_info in words_info:
        if current_speaker != word_info.speaker_tag:
            # 이전 화자의 문장 출력
            if current_sentence:
                f.write(f"화자 {current_speaker}: {' '.join(current_sentence)}\n")
            
            # 새 화자 시작
            current_speaker = word_info.speaker_tag
            current_sentence = [word_info.word]
        else:
            current_sentence.append(word_info.word)
    
    # 마지막 문장 출력
    if current_sentence:
        f.write(f"화자 {current_speaker}: {' '.join(current_sentence)}\n")

print(f"Transcription saved to {output_file}")

# 화자 수 확인
unique_speakers = set(word.speaker_tag for word in words_info)
print(f"감지된 화자 수: {len(unique_speakers)}")
print(f"화자 태그: {sorted(unique_speakers)}")
