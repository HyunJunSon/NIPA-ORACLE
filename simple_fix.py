import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

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

    df = pd.read_excel("data/naver_fin01.xlsx", sheet_name='결과')
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)

    # 간단한 타입 변환
    df["stock_name"] = df["stock_name"].astype(str)   
    df["stock_price"] = df["stock_price"].astype(str).str.replace(',', '').astype(int)
    df["up_down"] = df["up_down"].astype(str)
    df["up_down_price"] = df["up_down_price"].astype(str).str.replace(r'[^\d]', '', regex=True).astype(int)
    
    insert_sql = """
        INSERT INTO naver_stock (
            stock_name, stock_price, up_down, up_down_price
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """

    rows_to_insert = [
        (r["stock_name"], r["stock_price"], r["up_down"], r["up_down_price"])
        for _, r in df.iterrows()
    ]

    execute_batch(cursor, insert_sql, rows_to_insert, page_size=100)
    conn.commit()

    cursor.execute("""
        SELECT id, stock_name, stock_price, up_down, up_down_price
        FROM naver_stock
        ORDER BY id DESC
        LIMIT 10
    """)
    recent = cursor.fetchall()

    print("최근 10개 데이터")
    for row in recent:
        print(row)

except Exception as e:
    print("에러 발생:", e)

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("DB 연결이 종료되었습니다.")
