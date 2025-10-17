# rag_model.py
import os
import psycopg
import fitz  # PyMuPDF
from typing import List, Union
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredImageLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def extract_images_from_pdf(pdf_path: str, output_dir: str = "data/pic-data/") -> List[str]:
    """
    PDF에서 이미지를 추출하여 output_dir에 저장
    
    Args:
        pdf_path: PDF 파일 경로
        output_dir: 이미지 저장 디렉토리
    
    Returns:
        저장된 이미지 파일 경로 리스트
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_document = fitz.open(pdf_path)
    saved_images = []
    file_name = Path(pdf_path).stem  # 확장자 제외한 파일명
    
    for page_index in range(len(pdf_document)):
        page = pdf_document[page_index]
        images = page.get_images(full=True)
        print(f"페이지 {page_index}: {len(images)}개 이미지 발견")
        
        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # 파일명 생성: 원본파일명_페이지번호_이미지번호.확장자
                image_ext = base_image["ext"]
                filename = f"{file_name}_page{page_index}_img{img_index}.{image_ext}"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                print(f"✅ {filename} 저장 완료")
                saved_images.append(filepath)
            except Exception as e:
                print(f"❌ 에러: {e}")
    
    pdf_document.close()
    return saved_images


class RAGBot:
    def __init__(self, file_paths: Union[str, List[str]], collection_name: str = "rag-resume"):
        """
        RAG 챗봇 초기화
        
        Args:
            file_paths: 단일 파일 경로(str) 또는 파일 경로 리스트(List[str])
            collection_name: 벡터스토어 컬렉션 이름
        """
        load_dotenv()
        
        # 단일 파일인 경우 리스트로 변환
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths
            
        self.collection_name = collection_name
        self.documents = []

        print(f"\n[1] 문서 로드 ({len(self.file_paths)}개 파일)")
        self._load_all_documents()
        print(f"✅ 총 {len(self.documents)}개의 문서/페이지 로드 완료")

        print("\n[2] 문서 분할")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.splits = splitter.split_documents(self.documents)
        print(f"✅ 총 {len(self.splits)}개의 청크로 분할 완료")

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
            search_kwargs={"k": 5}  # 다중 파일이므로 k 증가
        )

        print("\n[7] 프롬프트 구성")
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """당신은 이력서 작성 전문가입니다.
                아래 제공된 개인 정보(Context)를 바탕으로 질문에 답변하세요.
                
                **중요 규칙:**
                1. 문맥에 있는 내용만 사용하여 답변하세요.
                2. 이력서 항목을 정리할 때는 다음 형식을 사용하세요:
                   ### 항목명
                   내용
                3. 여러 항목을 한 번에 제공할 때는 각 항목을 ### 으로 구분하세요.
                4. 문맥에 없는 내용은 "제공된 자료에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
                5. 가능한 한 구체적이고 상세하게 답변하세요.

                개인 정보 문맥:
                {context}
                """
            ),
            ("human", "{question}")
        ])

        print("\n[8] LLM 설정")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        print("\n[9] RAG 체인 구성")
        self.rag_chain = self._create_chain()
        print("✅ RAG 체인 구성 완료\n")

    def _load_all_documents(self):
        """모든 파일을 로드하여 documents 리스트에 추가"""
        for file_path in self.file_paths:
            try:
                file_ext = Path(file_path).suffix.lower()
                file_name = Path(file_path).name
                
                print(f"  📄 로딩 중: {file_name}")
                
                # 파일 타입별 로더 선택
                extracted_images = []
                if file_ext == '.pdf':
                    # PDF에서 이미지 추출
                    extracted_images = extract_images_from_pdf(file_path)
                    print(f"    🖼️ PDF에서 {len(extracted_images)}개 이미지 추출됨")
                    
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                elif file_ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    
                elif file_ext in ['.docx', '.doc']:
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                    
                elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    # 이미지는 OCR을 통해 텍스트 추출
                    try:
                        loader = UnstructuredImageLoader(file_path)
                        docs = loader.load()
                    except Exception as e:
                        print(f"    ⚠️ 이미지 로드 실패 ({file_name}): {e}")
                        # 이미지 로드 실패 시 파일명과 경로만 저장
                        docs = [Document(
                            page_content=f"이미지 파일: {file_name}",
                            metadata={"source": file_path, "type": "image"}
                        )]
                else:
                    print(f"    ⚠️ 지원하지 않는 파일 형식: {file_ext}")
                    continue
                
                # 메타데이터에 파일명 추가
                for doc in docs:
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = file_path
                    doc.metadata["filename"] = file_name
                    
                    # PDF 파일의 경우 추출된 이미지 정보 추가
                    if file_ext == '.pdf':
                        doc.metadata["extracted_images"] = extracted_images
                        doc.metadata["has_images"] = len(extracted_images) > 0
                
                self.documents.extend(docs)
                print(f"    ✅ {len(docs)}개 문서/페이지 로드됨")
                
            except Exception as e:
                print(f"    ❌ 파일 로드 실패 ({file_name}): {e}")
                continue

    def _setup_db(self):
        """PostgreSQL 데이터베이스 설정"""
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
        """RAG 체인 생성"""
        def format_docs(docs):
            formatted = []
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get('filename', doc.metadata.get('source', '알 수 없음'))
                
                # 문서 내용
                content = f"[출처: {source}]\n{doc.page_content}"
                
                # PDF에서 추출된 이미지 정보 추가
                if doc.metadata.get('has_images', False):
                    extracted_images = doc.metadata.get('extracted_images', [])
                    if extracted_images:
                        image_info = f"\n[포함된 이미지: {len(extracted_images)}개]"
                        image_list = "\n" + "\n".join([f"- 이미지 파일: {os.path.basename(img_path)}" for img_path in extracted_images])
                        content += image_info + image_list
                
                formatted.append(content)
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

    def ask(self, question: str) -> str:
        """RAG 질의 실행"""
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            return f"⚠️ 오류 발생: {e}"
    
    def get_loaded_files_info(self) -> dict:
        """로드된 파일 정보 반환"""
        return {
            "total_files": len(self.file_paths),
            "total_documents": len(self.documents),
            "total_chunks": len(self.splits),
            "files": [Path(fp).name for fp in self.file_paths]
        }