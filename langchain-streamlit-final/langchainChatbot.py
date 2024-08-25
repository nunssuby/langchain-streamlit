# streamlit과 pyngrok openai 라이브러리를 설치
# !pip install streamlit -q
# !pip install pyngrok -q
# !pip install openai==0.28.1 -q


# openai, streamlit 라이브러리 불러오기
import openai
import streamlit as st
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

# sidebar에 OpenAI API 키 암호로 입력 받기
# API 키 발급 사이트 공지하기- https://platform.openai.com/account/api-keys
# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
openai_api_key = os.getenv("OPENAI_API_KEY")

db_path = '../db3'
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
database = Chroma(persist_directory= db_path, embedding_function = embeddings )  

# 💬 앱 제목과 🚀 설명
st.title("💬 챗봇")
st.caption("🚀 OpenAI를 이용한 스트림릿 챗봇")

# 초기 메시지 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]

# 모든 대화 메시지를 화면에 표시
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

  

# 사용자 입력 및 응답 처리

# 사용자 채팅 입력 확인
if prompt := st.chat_input():

    # API 키가 입력되지 않았을 경우, 메시지를 표시하고 프로그램을 중단
    if not openai_api_key:
        st.info("OpenAI API key를 입력해주세요.")
        st.stop()

    # OpenAI API 키 설정
    openai.api_key = openai_api_key

    # 사용자의 메시지를 히스토리에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # # OpenAI API를 사용하여 응답 생성
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=st.session_state.messages
    # )

    # 어시스턴트의 응답을 히스토리에 추가하고 화면에 표시
    # msg = response.choices[0].message["content"]

    k = 3
    retriever = database.as_retriever(search_kwargs={"k": k})
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    qa = RetrievalQA.from_llm(llm=chat,  retriever=retriever,  return_source_documents=True)
    result = qa(prompt)
    msg = result["result"]
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


