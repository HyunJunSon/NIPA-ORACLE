-- Active: 1760334103894@@140.238.1.184@5432
create table if not EXISTS test_post(
  id SERIAL PRIMARY key,
  title varchar(200) not null,
  content text not null,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  -- Check if you have proper permissions
  GRANT ALL PRIVILEGES ON DATABASE your_database TO your_user;
);

INSERT INTO test_post (title, content)
VALUES
('첫 번째 게시글', '이것은 첫 번째 게시글의 내용입니다.'),
('두 번째 게시글', 'Flask와 PostgreSQL을 연동한 예시입니다.'),
('세 번째 게시글', '템플릿에서 데이터 반복 출력 테스트 중입니다.');

CREATE TABLE web_user_info (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) NOT NULL,
    password_hash VARCHAR(120) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS web_documents_list (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER REFERENCES web_user_info(id) ON DELETE CASCADE
);

select * from web_user_info;