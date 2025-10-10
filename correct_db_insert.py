import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import re

load_dotenv()

db_config = {
    'host': os.getenv("DB_HOST"),
    'port': 5432,
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS'),
}

conn = None
cursor = None

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    df = pd.read_excel("data/naver_fin_more.xlsx", sheet_name='결과')
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)

    # 타입 변환
    df["stock_name"] = df["stock_name"].astype(str)   
    df["stock_price"] = df["stock_price"].astype(str).str.replace(',', '').astype(int)
    df["up_down"] = df["up_down"].astype(str)
    
    # up_down_price에서 숫자만 추출
    def extract_number(text):
        numbers = re.findall(r'\d+', str(text).replace(',', ''))
        return int(''.join(numbers)) if numbers else 0
    
    df["up_down_price"] = df["up_down_price"].apply(extract_number)
    
    # 기존 데이터 삭제
    cursor.execute("DELETE FROM naver_stock")
    
    insert_sql = """
        INSERT INTO naver_stock (
            stock_name, stock_price, up_down, up_down_price
        )
        VALUES (%s, %s, %s, %s)
    """

    rows_to_insert = [
        (r["stock_name"], r["stock_price"], r["up_down"], r["up_down_price"])
        for _, r in df.iterrows()
    ]

    execute_batch(cursor, insert_sql, rows_to_insert)
    conn.commit()

    print(f"{len(rows_to_insert)}개 데이터 정상 삽입 완료")

except Exception as e:
    print("에러 발생:", e)

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("DB 연결이 종료되었습니다.")
