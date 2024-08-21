# !pip install streamlit -q
# !pip install pyngrok -q
# !pip install openai -q
# !pip install langchain -q
# !pip install -U langchain-community -q
# !pip install sentence-transformers -q
# !pip install faiss-cpu -q
# !pip install pyPDF2 -q

# 라이브러리 불러오기
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF 문서에서 텍스트 추출 함수 정의하기
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 텍스트 분할 함수 정의하기
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 임베딩 생성 & 벡터 저장소 생성 함수 정의하기
def get_vectorstore(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# openai 키 입력
# import os
# os.environ["OPENAI_API_KEY"] = 


# 주어진 벡터 저장소로 대화 체인 생성 함수 정의하기
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k'),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h: h,
        memory=memory
    )
    return conversation_chain


user_uploads = st.file_uploader("파일을 업로드해주세요~", accept_multiple_files=True)
if user_uploads is not None:
    if st.button("Upload"):
        with st.spinner("처리중.."):
            # PDF 텍스트 가져오기
            raw_text = get_pdf_text(user_uploads)
            # 텍스트에서 청크 검색
            text_chunks = get_text_chunks(raw_text)
            # PDF 텍스트 저장을 위해 FAISS 벡터 저장소 만들기
            vectorstore = get_vectorstore(text_chunks)
            # 대화 체인 만들기
            st.session_state.conversation = get_conversation_chain(vectorstore)

if user_query := st.chat_input("질문을 입력해주세요~"):
    # 대화 체인을 사용하여 사용자의 메시지를 처리
    if 'conversation' in st.session_state:
        result = st.session_state.conversation({
            "question": user_query,
            "chat_history": st.session_state.get('chat_history', [])
        })
        response = result["answer"]
    else:
        response = "먼저 문서를 업로드해주세요~."
    with st.chat_message("assistant"):
        st.write(response)
