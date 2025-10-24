from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from interface.db import DatabaseManager, User, Document
import os

"""
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
"""

app = Flask(__name__)
app.config['SECRET_KEY'] = 'edu013'  # seed key 용도

# 데이터베이스 매니저 초기화
connection_string = f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASS', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'testdb')}"
db_manager = DatabaseManager(connection_string)

# 로그인 관리자 설정
login_manager = LoginManager()
login_manager.init_app(app)
# login_manager.login_view = 'login'  # 로그인 필요 시 이동할 페이지 설정

# 사용자 로더
@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(db_manager, int(user_id))


# 루트 페이지
@app.route('/')
def index():
    return redirect(url_for('login'))


# 로그인 페이지
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.get_by_username(db_manager, username)

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('document_list'))
        else:
            flash('Invalid username or password')

    return render_template('login.html')


# 로그아웃
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# 문서 목록 페이지
@app.route('/documents')
@login_required
def document_list():
    documents = Document.get_by_user_id(db_manager, current_user.id)
    return render_template('document_list.html', documents=documents)

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(db_manager, int(user_id))

if __name__ == '__main__':
    # Create test user if not exists
    db_manager.create_test_user()
    app.run(host="0.0.0.0")