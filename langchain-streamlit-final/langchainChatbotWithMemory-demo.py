import openai
import streamlit as st
import os
import warnings

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

from langchain.memory import ConversationBufferMemory

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
k = 3

# 💬 앱 제목과 🚀 설명
st.title("💬 챗봇")
st.caption("🚀 OpenAI를 이용한 스트림릿 챗봇")

# 초기 메시지 설정
if "conversations" not in st.session_state:
    st.session_state["conversations"] = []
    st.session_state["current_conversation"] = []
    st.session_state["selected_conversation"] = None
    
    db_path = '../db3'
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    st.session_state["database"] = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    retriever = st.session_state["database"].as_retriever(search_kwargs={"k": k})
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", input_key="question",
                                                          output_key="answer", return_messages=True)
    st.session_state["qa"] = ConversationalRetrievalChain.from_llm(llm=chat, retriever=retriever, memory=st.session_state["memory"],
                                                                    return_source_documents=True, output_key="answer")

# 사이드바에 대화 히스토리 및 새 대화 버튼 추가
with st.sidebar:
    st.header("대화 기록")
    
    # 새 대화 버튼 추가
    if st.button("새 대화 시작"):
        st.session_state["current_conversation"] = []
        st.session_state["selected_conversation"] = None

    # 대화 기록 표시
    for i, conversation in enumerate(st.session_state["conversations"]):
        if conversation:  # 대화에 메시지가 있는지 확인
            if st.button(conversation[0]["content"], key=f"conv_{i}"):
                st.session_state["selected_conversation"] = i
                st.session_state["current_conversation"] = conversation

# 선택된 대화 표시
if st.session_state["current_conversation"]:
    for msg in st.session_state["current_conversation"]:
        st.chat_message(msg["role"]).write(msg["content"])

# 사용자 채팅 입력 확인
if prompt := st.chat_input("메시지를 입력하세요..."):

    # API 키가 입력되지 않았을 경우, 메시지를 표시하고 프로그램을 중단
    if not openai_api_key:
        st.info("OpenAI API key를 입력해주세요.")
        st.stop()

    # 사용자의 메시지를 현재 대화에 추가하고 화면에 표시
    st.session_state["current_conversation"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 처음 메시지를 보낼 때 대화 목록에 추가
    if len(st.session_state["current_conversation"]) == 1:
        st.session_state["conversations"].append(st.session_state["current_conversation"])
        st.session_state["selected_conversation"] = len(st.session_state["conversations"]) - 1

    # ConversationalRetrievalQA 체인 사용하여 응답 생성
    result = st.session_state["qa"]({"question": prompt})
    msg = result["answer"]

    # 어시스턴트의 응답을 현재 대화에 추가하고 화면에 표시
    st.session_state["current_conversation"].append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    # 대화 목록을 업데이트하여 이어진 대화를 반영
    st.session_state["conversations"][st.session_state["selected_conversation"]] = st.session_state["current_conversation"]

    # 현재 담겨 있는 메모리 내용 전체 확인
    history = st.session_state["memory"].load_memory_variables({})
    print(history)