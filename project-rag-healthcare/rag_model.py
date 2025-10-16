# rag_model.py

import os
import psycopg
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGBot:
    def __init__(self, file_path: str, collection_name: str = "rag-healthcare"):
        load_dotenv()
        self.file_path = file_path
        self.collection_name = collection_name

        print("\n[1] 문서 로드")
        loader = PyPDFLoader(file_path)
        self.documents = loader.load()

        print(f"총 {len(self.documents)} 페이지 로드 완료")

        print("\n[2] 문서 분할")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.splits = splitter.split_documents(self.documents)

        print(f"총 {len(self.splits)}개의 청크로 분할 완료")

        print("\n[3] 임베딩 모델 로드")
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

        print("\n[4] Postgres 연결")
        self.connection_string = self._setup_db()
        print("✅ Postgres 연결 및 vector extension 확인 완료")

        print("\n[5] 벡터스토어 생성")
        self.vectorstore = PGVector.from_documents(
            documents=self.splits,
            embedding=self.embeddings_model,
            collection_name=self.collection_name,
            connection=self.connection_string,
            pre_delete_collection=True
        )

        print(f"✅ 벡터스토어 저장 완료 ({len(self.splits)}개 청크)")

        print("\n[6] 검색기 생성")
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        print("\n[7] 프롬프트 구성")
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """당신은 생성형 인공지능 의료기기와 정부정책 전문가입니다.
                아래 제공된 문맥(Context)을 반드시 참고하여 질문에 답변하세요.
                1. 문맥에 관련 내용이 있으면 그것을 바탕으로 답변하세요.
                2. 문맥에 없는 내용만 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
                3. 답변 시 문맥의 근거를 간단히 표시하세요.

            문맥:
            {context}
            """
                        ),
                        ("human", "{question}")
                    ])

        print("\n[8] LLM 설정")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

        print("\n[9] RAG 체인 구성")
        self.rag_chain = self._create_chain()
        print("✅ RAG 체인 구성 완료")

    # 내부 메서드
    def _setup_db(self):
        DB_CONFIG = {
            'host': os.getenv("DB_HOST"),
            'port': 5432,
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASS'),
        }

        try:
            conn = psycopg.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            raise RuntimeError(f"Postgres 연결 실패: {e}")

        return (
            f"postgresql+psycopg://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        )

    def _create_chain(self):
        def format_docs(docs):
            formatted = []
            for idx, doc in enumerate(docs, 1):
                formatted.append(f"[문서 {idx}]\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)

        return (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # 외부 인터페이스
    def ask(self, question: str) -> str:
        """RAG 질의 실행"""
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            return f"⚠️ 오류 발생: {e}"