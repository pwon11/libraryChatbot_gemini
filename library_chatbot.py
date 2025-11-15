import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma

# ---------------- Gemini API 키 ----------------
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()

# ---------------- PDF 경로 지정 ----------------
# 여기서 PDF 경로를 코드에서 직접 지정
pdf_paths = [
    r"/mount/src/librarychatbot_gemini/11. KRISO_심해 탐사용 다관절 해저 로봇 시스템.pdf",
    r"/mount/src/librarychatbot_gemini/4.로봇기술리뷰_조한길.pdf",
    r"/mount/src/librarychatbot_gemini/조류.pdf",
    r"/mount/src/librarychatbot_gemini/연안.pdf",
    r"/mount/src/librarychatbot_gemini/무인 쓰레기 수거 로봇.pdf"
    # 필요 시 PDF 경로 추가 가능
]

# ---------------- PDF 로드 및 분할 ----------------
@st.cache_resource
def load_and_split_pdfs(paths):
    all_docs = []
    for path in paths:
        if not os.path.exists(path):
            st.error(f"❌ PDF 파일이 존재하지 않습니다: {path}")
            continue
        loader = PyPDFLoader(path)
        docs = loader.load_and_split()
        all_docs.extend(docs)
    st.success(f"📄 총 {len(all_docs)}개의 문서 청크 로드 완료!")
    return all_docs

# ---------------- 벡터 저장 ----------------
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"📄 {len(split_docs)}개의 텍스트 청크로 분할했습니다.")

    persist_directory = "./chroma_db"
    st.info("🤖 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.info("🔢 벡터 임베딩 생성 및 저장 중...")
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )
    st.success("💾 벡터 데이터베이스 생성 완료!")
    return vectorstore

@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        return create_vector_store(_docs)

# ---------------- RAG 체인 초기화 ----------------
@st.cache_resource
def initialize_components(selected_model, pdf_paths):
    pages = load_and_split_pdfs(pdf_paths)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer perfect. please use emoji with the answer. 
대답은 한국어로 하고 존댓말을 사용해주세요.\

{context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 'gemini-pro' 모델을 사용해보세요.")
        raise

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# ---------------- Streamlit UI ----------------
st.header("해양 자료 Q&A 챗봇 💬 🌊")
st.info("PDF 경로는 코드 내 pdf_paths 리스트에서 지정되어 있습니다.")

# Gemini 모델 선택
option = st.selectbox("Select Gemini Model",
                      ("gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite"),
                      index=0)

with st.spinner("🔧 챗봇 초기화 중... 잠시만 기다려주세요"):
    rag_chain = initialize_components(option, pdf_paths)
st.success("✅ 챗봇이 준비되었습니다!")

chat_history = StreamlitChatMessageHistory(key="chat_messages")
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                     "content": "해양 관련 자료에 대해 무엇이든 물어보세요!"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt_message := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": prompt_message}, config)
            answer = response['answer']
            st.write(answer)
            with st.expander("참고 문서 확인"):
                for doc in response.get('context', []):
                    st.markdown(doc.metadata.get('source', 'No source'))
