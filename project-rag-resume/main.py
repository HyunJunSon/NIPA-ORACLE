# app.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_model import RAGBot
import json
import os
from pathlib import Path

# --- 페이지 설정 ---
st.set_page_config(
    page_title="이력서 작성 도우미",
    page_icon="📝",
    layout="wide"
)

# --- 초기화 ---
st.title("📝 이력서 작성 도우미 챗봇")
st.markdown("개인 데이터베이스에서 이력서 항목을 추출하여 제공합니다. 각 항목은 바로 복사할 수 있습니다.")

# --- data 디렉토리의 모든 파일 로드 ---
def load_data_files(data_dir="data"):
    """data 디렉토리의 모든 파일 경로를 리스트로 반환"""
    if not os.path.exists(data_dir):
        st.error(f"❌ '{data_dir}' 디렉토리가 존재하지 않습니다.")
        return []
    
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.txt', '.docx', '.doc']
    file_paths = []
    
    for file in Path(data_dir).iterdir():
        if file.is_file() and file.suffix.lower() in supported_extensions:
            file_paths.append(str(file))
    
    return file_paths

# --- RAGBot 객체 로드 (한 번만 초기화) ---
if "rag_bot" not in st.session_state:
    data_files = load_data_files("data")
    
    if not data_files:
        st.warning("⚠️ 'data' 디렉토리에 파일이 없습니다. PDF, 이미지 등의 파일을 추가해주세요.")
        st.session_state.rag_bot = None
    else:
        with st.spinner(f"📂 {len(data_files)}개의 파일을 로드하는 중..."):
            try:
                # 파일 리스트를 RAGBot에 전달
                st.session_state.rag_bot = RAGBot(data_files)
                st.success(f"✅ {len(data_files)}개의 파일이 성공적으로 로드되었습니다!")
            except Exception as e:
                st.error(f"❌ 파일 로드 중 오류 발생: {str(e)}")
                st.session_state.rag_bot = None

# --- 로드된 파일 정보 저장 ---
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = load_data_files("data")

# --- 대화 기록 관리 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 이력서 항목 추출 여부 ---
if "resume_items" not in st.session_state:
    st.session_state.resume_items = []

# --- CSS 스타일 추가 ---
st.markdown("""
<style>
    .resume-item {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .item-title {
        font-weight: bold;
        color: #1f77b4;
        font-size: 1.1em;
        margin-bottom: 8px;
    }
    .item-content {
        color: #333;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .copy-success {
        color: #28a745;
        font-size: 0.9em;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 복사 버튼 함수 ---
def create_copy_button(content, key):
    """클립보드 복사 버튼 생성"""
    if st.button("📋 복사", key=key, help="클립보드에 복사"):
        # JavaScript를 사용하여 클립보드에 복사
        st.components.v1.html(
            f"""
            <script>
                const text = {json.dumps(content)};
                navigator.clipboard.writeText(text).then(function() {{
                    console.log('복사 완료!');
                }}, function(err) {{
                    console.error('복사 실패:', err);
                }});
            </script>
            """,
            height=0
        )
        st.success("✅ 복사되었습니다!", icon="✅")

# --- 이력서 항목 파싱 함수 ---
def parse_resume_items(response):
    """
    RAG 응답에서 이력서 항목들을 추출
    응답 형식: "### 항목명\n내용\n\n### 항목명2\n내용2"
    """
    items = []
    if "###" in response:
        sections = response.split("###")
        for section in sections[1:]:  # 첫 번째는 빈 문자열
            lines = section.strip().split("\n", 1)
            if len(lines) >= 2:
                title = lines[0].strip()
                content = lines[1].strip()
                items.append({"title": title, "content": content})
            elif len(lines) == 1:
                title = lines[0].strip()
                items.append({"title": title, "content": ""})
    return items

# --- 사이드바: 이력서 항목 가이드 ---
with st.sidebar:
    st.header("📌 이력서 항목 가이드")
    st.markdown("""
    **추천 질문 예시:**
    - "이력서 작성에 필요한 모든 정보를 알려줘"
    - "내 학력 정보를 알려줘"
    - "경력 사항을 정리해줘"
    - "보유 기술을 나열해줘"
    - "프로젝트 경험을 알려줘"
    - "자격증 목록을 보여줘"
    
    **사용 방법:**
    1. 질문을 입력하면 AI가 DB에서 정보를 추출합니다
    2. 각 항목별로 복사 버튼이 제공됩니다
    3. 버튼을 클릭하면 클립보드에 복사됩니다
    """)
    
    # 로드된 파일 목록 표시
    st.markdown("---")
    st.subheader("📂 로드된 파일")
    if st.session_state.loaded_files:
        for file_path in st.session_state.loaded_files:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].upper()
            
            # 파일 타입별 아이콘
            icon = "📄"
            if file_ext in ['.PDF']:
                icon = "📕"
            elif file_ext in ['.JPG', '.JPEG', '.PNG', '.GIF', '.BMP']:
                icon = "🖼️"
            elif file_ext in ['.TXT']:
                icon = "📝"
            elif file_ext in ['.DOCX', '.DOC']:
                icon = "📘"
            
            st.text(f"{icon} {file_name}")
    else:
        st.info("파일이 없습니다.")
    
    st.markdown("---")
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.session_state.resume_items = []
        st.rerun()
    
    if st.button("🔄 파일 새로고침"):
        st.session_state.loaded_files = load_data_files("data")
        if "rag_bot" in st.session_state:
            del st.session_state.rag_bot
        st.rerun()

# --- 이전 대화 출력 ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg.role):
        st.markdown(msg.content)
        
        # AI 응답인 경우 항목별 복사 버튼 추가
        if msg.role == "assistant" and hasattr(msg, 'items') and msg.items:
            st.markdown("---")
            st.markdown("**📋 복사 가능한 항목:**")
            
            for item_idx, item in enumerate(msg.items):
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    st.markdown(f'<div class="resume-item">'
                              f'<div class="item-title">{item["title"]}</div>'
                              f'<div class="item-content">{item["content"]}</div>'
                              f'</div>', unsafe_allow_html=True)
                
                with col2:
                    create_copy_button(item["content"], f"copy_{idx}_{item_idx}")

# --- 사용자 입력 ---
if prompt := st.chat_input("이력서 항목에 대해 질문하세요 (예: 학력 정보를 알려줘)"):
    # RAGBot이 초기화되지 않은 경우 처리
    if st.session_state.rag_bot is None:
        st.error("❌ RAG 시스템이 초기화되지 않았습니다. 'data' 디렉토리에 파일을 추가하고 새로고침하세요.")
    else:
        # 사용자 메시지 추가
        st.session_state.messages.append(HumanMessage(content=prompt, role="user"))
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("정보를 검색하고 있습니다..."):
                try:
                    response = st.session_state.rag_bot.ask(prompt)
                    st.markdown(response)
                    
                    # 이력서 항목 파싱
                    items = parse_resume_items(response)
                    
                    if items:
                        st.markdown("---")
                        st.markdown("**📋 복사 가능한 항목:**")
                        
                        # 각 항목별로 복사 버튼 제공
                        for item_idx, item in enumerate(items):
                            col1, col2 = st.columns([5, 1])
                            
                            with col1:
                                st.markdown(f'<div class="resume-item">'
                                          f'<div class="item-title">{item["title"]}</div>'
                                          f'<div class="item-content">{item["content"]}</div>'
                                          f'</div>', unsafe_allow_html=True)
                            
                            with col2:
                                create_copy_button(
                                    item["content"], 
                                    f"copy_new_{item_idx}"
                                )
                    
                    # AI 응답을 메시지에 추가 (항목 정보 포함)
                    ai_message = AIMessage(content=response, role="assistant")
                    ai_message.items = items  # 항목 정보 저장
                    st.session_state.messages.append(ai_message)
                    
                except Exception as e:
                    st.error(f"❌ 오류가 발생했습니다: {str(e)}")

# --- 푸터 ---
st.markdown("---")
st.caption("💡 Tip: RAG 기술을 사용하여 개인 데이터베이스에서 정보를 추출합니다. 'data' 디렉토리에 PDF, 이미지 등을 추가하세요.")