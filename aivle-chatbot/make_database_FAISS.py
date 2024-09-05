import os
import streamlit as st
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key를 찾을 수 없습니다. Secrets에 API key를 설정해주세요.")
    st.stop()

# aivle.csv 데이터 로드
csv_path = 'aivle.csv'
try:
    aivle_data = pd.read_csv(csv_path)
    st.success("aivle.csv 파일을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    st.error(f"파일을 찾을 수 없습니다: {csv_path}")
    st.stop()

# QA 열 데이터를 리스트로 변환
qa_texts = aivle_data['QA'].tolist()

# OpenAI Embeddings 초기화
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
except Exception as e:
    st.error(f"OpenAI Embeddings 초기화 중 오류 발생: {e}")
    st.stop()

# FAISS 데이터베이스 생성
try:
    faiss_db = FAISS.from_texts(qa_texts, embeddings)
    st.success("FAISS 데이터베이스가 성공적으로 생성되었습니다.")
except Exception as e:
    st.error(f"FAISS 데이터베이스 생성 중 오류 발생: {e}")
    st.stop()

# FAISS 인덱스를 파일로 저장 (db_path 설정)
db_path = '../aivle_db_faiss/faiss_index'
try:
    faiss_db.save_local(db_path)
    st.success(f"FAISS 인덱스가 성공적으로 {db_path}에 저장되었습니다.")
except Exception as e:
    st.error(f"FAISS 인덱스 저장 중 오류 발생: {e}")