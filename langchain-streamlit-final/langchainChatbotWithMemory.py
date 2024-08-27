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

from langchain.memory import ConversationBufferMemory


# sidebar에 OpenAI API 키 암호로 입력 받기
# API 키 발급 사이트 공지하기- https://platform.openai.com/account/api-keys
# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


openai_api_key = os.getenv("OPENAI_API_KEY")
k = 3

# 💬 앱 제목과 🚀 설명
st.title("💬 챗봇")
st.caption("🚀 OpenAI를 이용한 스트림릿 챗봇")

# 초기 메시지 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]
    

    db_path = '../db3'
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    st.session_state["database"] = Chroma(persist_directory= db_path, embedding_function = embeddings )
    # database = Chroma(persist_directory= db_path, embedding_function = embeddings )  
    
    retriever =  st.session_state["database"].as_retriever(search_kwargs={"k": k})
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", input_key="question",
                                output_key="answer", return_messages=True)
    st.session_state["qa"] = ConversationalRetrievalChain.from_llm(llm=chat, retriever=retriever, memory=st.session_state["memory"],    
                                           return_source_documents=True,  output_key="answer")

# 모든 대화 메시지를 화면에 표시
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

  


# 대화 메모리 생성
# memory = ConversationBufferMemory(memory_key="chat_history", input_key="question",
#                                 output_key="answer", return_messages=True)
# qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=retriever, memory=memory,    
#                                            return_source_documents=True,  output_key="answer")

# count = 0
# count  += 1
# print("count", count)


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

    
    # ConversationalRetrievalQA 체인 생성
    
    
    # qa = get_conversation_chain_memory(memory,k)

    result = st.session_state["database"].similarity_search_with_score(prompt, k = k) #← 데이터베이스에서 유사도가 높은 문서를 가져옴
    # sim1 = round(result[0][1], 5)
    # sim2 = round(result[1][1], 5)
    # sim3 = round(result[2][1], 5)

    
    result = st.session_state["qa"]({"question": prompt})
    msg = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    # 현재 담겨 있는 메모리 내용 전체 확인
    history = st.session_state["memory"].load_memory_variables({})
    print(history)
