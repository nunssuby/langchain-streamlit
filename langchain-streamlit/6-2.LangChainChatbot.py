# !pip install streamlit -q
# !pip install pyngrok -q
# !pip install openai -q
# !pip install langchain -q
# !pip install langchain_openai -q

# 필요한 라이브러리 불러오기
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import streamlit as st

# sidebar에 OpenAI API 키 암호로 입력 받기
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")

# 앱 제목과 설명을 설정합니다
st.title("💬 챗봇")
st.caption(" 🦜️🔗 LangChain과 OpenAI를 이용한 스트림릿 챗봇")

# 초기 메시지 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="무엇을 도와드릴까요?")]

# 모든 대화 메시지를 화면에 표시
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# 사용자 채팅 입력 확인
if prompt := st.chat_input():
    # 사용자 입력 메시지 저장하고 화면에 표시
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    # OpenAI API 키가 입력되었는지 확인
    if not openai_api_key:
        st.info("OpenAI API key를 입력해주세요.")
        st.stop()

    # AI 모델의 응답을 처리하고 화면에 실시간 표시
    with st.chat_message("assistant"):
        # 응답을 받기 위한 빈 공간 설정
        response_container = st.empty()
        # OpenAI와의 연결 설정
        llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True)
        # 챗봇 응답을 저장할 빈 문자열 생성
        response_text = ""

        # AI 모델 응답을 조각으로 처리
        for chunk in llm.stream(st.session_state.messages):
            # 응답 조각을 response_text에 추가하고 실시간 표시
            response_text += chunk.content
            response_container.markdown(response_text)

        # AI의 응답 메시지 저장
        st.session_state.messages.append(ChatMessage(role="assistant", content=response_text))
