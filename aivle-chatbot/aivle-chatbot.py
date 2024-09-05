import openai
import streamlit as st
import os
import warnings

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key를 찾을 수 없습니다. Secrets에 API key를 설정해주세요.")
    st.stop()

k = 3

# 💬 앱 제목과 🚀 설명
st.title("💬 챗봇")
st.caption("🚀 OpenAI를 이용한 스트림릿 챗봇")

# 초기 메시지 설정
if "conversations" not in st.session_state:
    st.session_state["conversations"] = []
    st.session_state["current_conversation"] = []
    st.session_state["selected_conversation"] = None
    
    db_path = '../aivle_db'
    embeddings = None  # 초기화

    # OpenAI Embeddings 초기화
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", 
            openai_api_key=openai_api_key
        )
    except Exception as e:
        st.error(f"Embeddings 초기화 중 오류 발생: {e}")
        st.stop()

    # Chroma 데이터베이스 초기화
    try:
        st.session_state["database"] = Chroma(persist_directory=db_path, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Chroma 데이터베이스 초기화 중 오류 발생: {e}")
        st.stop()

    st.session_state["retriever"] = st.session_state["database"].as_retriever(search_kwargs={"k": k})
    st.session_state["chat_model"] = ChatOpenAI(model="gpt-3.5-turbo")

# 상단 오른쪽에 새 대화 버튼을 배치
col1, col2 = st.columns([3, 1])

with col2:
    if st.button("새 대화 시작", key="new_conv", help="새로운 대화를 시작합니다"):
        st.session_state["current_conversation"] = []
        st.session_state["selected_conversation"] = len(st.session_state["conversations"])
        new_memory = ConversationBufferMemory(memory_key="chat_history", input_key="question",
                                              output_key="answer", return_messages=True)
        st.session_state["conversations"].append({"messages": [], "memory": new_memory})

# 사이드바에 대화 히스토리 추가
with st.sidebar:
    st.header("대화 기록")
    
    for i, conversation in enumerate(st.session_state["conversations"]):
        if conversation["messages"]:
            if st.button(conversation["messages"][0]["content"], key=f"conv_{i}"):
                st.session_state["selected_conversation"] = i
                st.session_state["current_conversation"] = conversation["messages"]

# 선택된 대화 표시
if st.session_state["current_conversation"]:
    for msg in st.session_state["current_conversation"]:
        st.chat_message(msg["role"]).write(msg["content"])

# 사용자 채팅 입력 확인
if prompt := st.chat_input("메시지를 입력하세요..."):

    if not openai_api_key:
        st.info("OpenAI API key를 입력해주세요.")
        st.stop()

    if st.session_state["selected_conversation"] is None:
        st.session_state["selected_conversation"] = len(st.session_state["conversations"])
        new_memory = ConversationBufferMemory(memory_key="chat_history", input_key="question",
                                              output_key="answer", return_messages=True)
        st.session_state["conversations"].append({"messages": [], "memory": new_memory})

    st.session_state["current_conversation"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    st.session_state["conversations"][st.session_state["selected_conversation"]]["messages"] = st.session_state["current_conversation"]

    current_memory = st.session_state["conversations"][st.session_state["selected_conversation"]]["memory"]

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=st.session_state["chat_model"], 
        retriever=st.session_state["retriever"], 
        memory=current_memory,
        return_source_documents=True, 
        output_key="answer"
    )

    result = qa_chain({"question": prompt})
    msg = result["answer"]

    st.session_state["current_conversation"].append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    st.session_state["conversations"][st.session_state["selected_conversation"]]["messages"] = st.session_state["current_conversation"]

    history = current_memory.load_memory_variables({})
    print(history)