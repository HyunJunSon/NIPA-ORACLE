# app.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_model import RAGBot

# --- 초기화 ---
st.title("🧠 생성형 인공지능 의료기기 RAG 챗봇")

# --- RAGBot 객체 로드 (한 번만 초기화) ---
if "rag_bot" not in st.session_state:
    st.session_state.rag_bot = RAGBot("data/생성형+인공지능+의료기기+허가·심사+가이드라인.pdf")

# --- 대화 기록 관리 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 이전 대화 출력 ---
for msg in st.session_state.messages:
    with st.chat_message(msg.role):
        st.markdown(msg.content)

# --- 사용자 입력 ---
if prompt := st.chat_input("무엇이든 물어보세요."):
    st.session_state.messages.append(HumanMessage(content=prompt, role="user"))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.session_state.rag_bot.ask(prompt)
        st.markdown(response)

    st.session_state.messages.append(AIMessage(content=response, role="assistant"))