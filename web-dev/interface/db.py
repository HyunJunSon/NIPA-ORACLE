import psycopg2
import hashlib
from datetime import datetime
from flask_login import UserMixin


# ===============================
# Database Manager
# ===============================
class DatabaseManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string

    def get_connection(self):
        return psycopg2.connect(self.connection_string)

    def create_test_user(self):
        conn = self.get_connection()
        cur = conn.cursor()

        # 테스트 사용자 존재 여부 확인
        cur.execute("SELECT id FROM web_user_info WHERE username = 'test'")
        if not cur.fetchone():
            # 테스트 사용자 추가
            password_hash = hashlib.sha256('test1234'.encode()).hexdigest()
            cur.execute(
                "INSERT INTO web_user_info (username, password_hash) VALUES (%s, %s)",
                ('test', password_hash)
            )
            conn.commit()

        cur.close()
        conn.close()


# ===============================
# User Model
# ===============================
class User(UserMixin):
    def __init__(self, id, username, password_hash, created_at=None):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.created_at = created_at

    def get_id(self):
        return str(self.id)

    def check_password(self, password):
        """비밀번호 검증"""
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def get_by_id(db_manager, user_id):
        conn = db_manager.get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash, created_at FROM web_user_info WHERE id = %s",
            (user_id,)
        )
        user_data = cur.fetchone()
        cur.close()
        conn.close()

        if not user_data:
            return None
        return User(user_data[0], user_data[1], user_data[2], user_data[3])

    @staticmethod
    def get_by_username(db_manager, username):
        conn = db_manager.get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash, created_at FROM web_user_info WHERE username = %s",
            (username,)
        )
        user_data = cur.fetchone()
        cur.close()
        conn.close()

        if not user_data:
            return None
        return User(user_data[0], user_data[1], user_data[2], user_data[3])


# ===============================
# Document Model
# ===============================
class Document:
    def __init__(self, id, title, content, created_at, updated_at, user_id):
        self.id = id
        self.title = title
        self.content = content
        self.created_at = created_at
        self.updated_at = updated_at
        self.user_id = user_id

    @staticmethod
    def create(db_manager, title, content, user_id):
        conn = db_manager.get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO web_documents_list (title, content, user_id) VALUES (%s, %s, %s) RETURNING id",
            (title, content, user_id)
        )
        document_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return document_id

    @staticmethod
    def get_by_id(db_manager, document_id):
        conn = db_manager.get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, content, created_at, updated_at, user_id FROM web_documents_list WHERE id = %s",
            (document_id,)
        )
        doc_data = cur.fetchone()
        cur.close()
        conn.close()

        if not doc_data:
            return None
        return Document(doc_data[0], doc_data[1], doc_data[2], doc_data[3], doc_data[4], doc_data[5])

    @staticmethod
    def get_by_user_id(db_manager, user_id):
        conn = db_manager.get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, content, created_at, updated_at, user_id FROM web_documents_list WHERE user_id = %s",
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [Document(*row) for row in rows]