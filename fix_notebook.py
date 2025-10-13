import json

# 원본 노트북 파일 읽기
with open('1013lesson.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 마지막 셀의 새로운 코드
new_code = '''import os
import psycopg2
from dotenv import load_dotenv
import requests

load_dotenv()

# Postgres 접속정보
db_config = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "port": os.getenv("DB_PORT"),
}

# 임베딩 생성 (로컬 서버 사용)
def get_embedding(text: str) -> list:
    url = "http://140.238.1.184:80/embed"
    
    data = {
        "sentences": [text]
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result["embeddings"][0]  # 첫 번째 문장의 임베딩 벡터 반환
    except requests.exceptions.RequestException as e:
        print(f"임베딩 서버 오류: {e}")
        return None

# 가장 유사한 문장 검색
def find_most_similar(text_vector: list) -> str:
    if not text_vector:
        return ""
    
    # pgvector 리터럴: [v1,v2,...] 형태
    vector_str = "[" + ",".join(map(str, text_vector)) + "]"

    sql = """
        SELECT title, vector_title <-> %s::vector AS distance
        FROM doc_vector_list
        ORDER BY distance ASC
        LIMIT 3;
    """

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    try:
        cur.execute(sql, (vector_str,))
        rows = cur.fetchall()
        return "\\n".join(row[0] for row in rows)
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return ""
    finally:
        cur.close()
        conn.close()

# 로컬 벡터 DB 기반 응답 생성
def ask_local_vectorDB(context: str, question: str) -> str:
    if not context:
        return "관련된 정보를 찾을 수 없습니다."
    
    # 간단한 키워드 매칭 기반 응답 생성
    response = f"""
질문: {question}

참고 정보를 바탕으로 한 답변:
{context}

위 정보들이 귀하의 질문과 관련이 있습니다.
"""
    return response

if __name__ == "__main__":
    user_question = input("질문을 입력하세요: ").strip()
    
    # 사용자 질문 임베딩
    question_vector = get_embedding(user_question)
    
    if question_vector:
        # 가장 유사한 문장 검색
        context_text = find_most_similar(question_vector)
        
        if context_text:
            answer = ask_local_vectorDB(context_text, user_question)
            print(f"\\n로컬 벡터DB의 답변:\\n{answer}")
        else:
            print("유사한 문장을 찾을 수 없습니다.")
    else:
        print("임베딩 생성에 실패했습니다.")'''

# 마지막 셀 찾기 및 수정
for i in range(len(notebook['cells']) - 1, -1, -1):
    cell = notebook['cells'][i]
    if cell['cell_type'] == 'code' and 'get_embedding' in ''.join(cell['source']):
        # 마지막 코드 셀 수정
        notebook['cells'][i]['source'] = new_code.split('\n')
        break

# 수정된 노트북 저장
with open('1013lesson.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("1013lesson.ipynb 파일이 성공적으로 수정되었습니다.")
print("주요 변경사항:")
print("1. 임베딩 서버 URL을 http://140.238.1.184:80/embed로 변경")
print("2. OpenAI GPT 대신 로컬 벡터DB 기반 응답 생성으로 변경")
print("3. 에러 처리 및 응답 형식 개선")
