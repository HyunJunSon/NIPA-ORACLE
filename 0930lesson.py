#!/Users/hyunjunson/Project/green-hat/backend/ml-pipline/yes/envs/langCh-env/bin/python

import openai
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI 

load_dotenv()  

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_ID = "gemini-2.0-flash-lite"

client = OpenAI(
    base_url=GEMINI_API_URL,
    api_key=GEMINI_API_KEY
)

def ai_chat(messages: list):
    # openai api package를 통해 gemini api를 호출합니다.
    # print(f"GEMINI API 호출, MODEL={LLM_ID}")
    response = client.chat.completions.create(
        model=LLM_ID,
        messages=messages,
    )
    return response

if __name__ == '__main__':
    messages = []
    if len(sys.argv) < 2:   
        print(f"Usage: {sys.argv[0]} '질문내용'")
        sys.exit()
    
    # 명령줄 인자 전체를 합쳐서 질문으로 사용
    question = ' '.join(sys.argv[1:])
    messages.append({'role': 'user', 'content': question})
    
    response = ai_chat(messages=messages)
    print(response.choices[0].message.content)