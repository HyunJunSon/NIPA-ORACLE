import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

db_config = {
    'host': os.getenv("DB_HOST"),
    'port': 5432,
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS'),
}

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS usa_health_data (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        age INTEGER,
        gender VARCHAR(10),
        blood_type VARCHAR(5),
        medical_condition VARCHAR(255),
        date_of_admission DATE,
        doctor VARCHAR(255),
        hospital VARCHAR(255),
        insurance_provider VARCHAR(255),
        billing_amount DECIMAL(10,2),
        room_number VARCHAR(10),
        admission_type VARCHAR(50),
        discharge_date DATE,
        medication VARCHAR(255),
        test_results VARCHAR(50),
        UNIQUE(name, date_of_admission, doctor)
    );
    """
    
    cursor.execute(create_table_sql)
    conn.commit()
    print("테이블이 성공적으로 생성되었습니다.")
    
except Exception as e:
    print(f"에러 발생: {e}")
    
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
