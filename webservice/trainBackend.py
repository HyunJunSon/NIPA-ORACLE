# -*- coding: utf-8 -*-
from fastapi import FastAPI,Request
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import sys

# GPU(CUDA) 사용 가능 여부를 확인하고 장치를 설정합니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 모델 아키텍처를 정의합니다. 학습할 때 사용했던 모델 구조와 동일해야 합니다.
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        return self.linear(x)

# 모델을 생성하고 GPU로 이동시킵니다.
model = LinearRegressionModel().to(device)

# --- 학습된 모델의 상태(가중치와 편향)를 불러옵니다. ---
try:
    checkpoint = torch.load('data/saved_model.pt', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("모델 'saved_model.pt'가 성공적으로 불러와졌습니다.")
except FileNotFoundError:
    print("오류: 'saved_model.pt' 파일을 찾을 수 없습니다. 학습된 모델 파일이 현재 디렉터리에 있는지 확인해 주세요.")
    sys.exit()

# 모델을 평가 모드로 전환합니다. (Dropout 등 비활성화)
model.eval()

# FastAPI 애플리케이션을 초기화합니다.
app = FastAPI()

# 요청 본문을 위한 Pydantic 모델 정의
class TextRequest(BaseModel):
    avg_temp: float
    min_temp: float
    max_temp: float
    rain_fall: float

# 응답 본문을 위한 Pydantic 모델 정의
class PredictResponse(BaseModel):
    predict:float

# --- 메인 페이지를 정의합니다. ---
@app.post('/predict',response_model=PredictResponse)
def home(request: TextRequest):
    # predict_val = 0.0
    # 폼 데이터를 가져와 float으로 변환합니다.
    avg_temp = float( request.avg_temp)
    min_temp = float( request.min_temp)
    max_temp = float( request.max_temp)
    rain_fall = float( request.rain_fall)
    # 입력값을 PyTorch 텐서로 변환합니다.
    input_data = torch.tensor([[avg_temp, min_temp, max_temp, rain_fall]], 
                                dtype=torch.float32, 
                                device=device)								   
	# --- 예측을 수행합니다. ---
    with torch.no_grad():
        prediction = model(input_data)
        predicted_price = f"{prediction.item():.2f}"
    return {"predict": predicted_price }
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("trainBackend:app",host="0.0.0.0",port=5000,reload=True)

