from flask import Flask, request, render_template
import requests
import sys
# Flask 웹 애플리케이션을 초기화합니다.
app = Flask(__name__)

url="http://localhost:5000/predict"
# --- 메인 페이지를 정의합니다. ---
@app.route('/', methods=['GET', 'POST'])
def home():
    # POST 요청(폼 제출)인 경우
    if request.method == 'POST':
        try:
            # 폼 데이터를 가져와 float으로 변환합니다.
            avg_temp = float( request.form['avg_temp'])
            min_temp = float( request.form['min_temp'])
            max_temp = float( request.form['max_temp'])
            rain_fall = float( request.form['rain_fall'])
            
            dicData = {
            "avg_temp": avg_temp,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "rain_fall":rain_fall
            }
            print(dicData)
            # requests.post() 함수를 사용하여 POST 요청 보내기
            response = requests.post(url, json=dicData)

            # JSON 응답을 파이썬 딕셔너리로 변환
            predict_result = response.json()
            print(predict_result)
            predicted_price = f"{predict_result.get('predict'):.2f}"
            # 예측 결과를 포함한 HTML을 반환합니다.
            return render_template("train_predict.html", prediction=predicted_price)
        
        except (ValueError, KeyError):
            print(ValueError)
            print(KeyError)
            return "유효한 숫자를 입력해 주세요. <a href='/'>다시 시도</a>"
    return render_template("train_predict.html", prediction=None)

if __name__ == '__main__':
    # 웹 서버를 5000번 포트로 실행합니다.
    app.run(host='localhost', port=5500,debug=True)

