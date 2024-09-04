import streamlit as st
import boto3
import json
import numpy as np

# AWS S3에서 임베딩 데이터를 로드하는 함수
def load_data_from_s3(bucket_name, file_key):
    # S3 클라이언트 설정 (secrets에서 AWS 자격 증명 불러오기)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=st.secrets["aws"]["access_key"],
        aws_secret_access_key=st.secrets["aws"]["secret_key"]
    )
    # S3에서 파일 다운로드
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    data = response['Body'].read().decode('utf-8')
    return json.loads(data)

# JSON에서 임베딩 벡터와 메타데이터 추출
def extract_vectors_and_metadata(embedded_data):
    vectors = []
    metadatas = []
    for item in embedded_data:
        vectors.append(np.array(item['임베딩']))
        metadatas.append({"요약": item["요약"], "세부인정사항": item["세부인정사항"]})
    return vectors, metadatas

def main():
    # 제목 설정
    st.title("RAG 기반 LLM 모델 테스트")

    # 텍스트 입력창
    user_input = st.text_input("여기에 텍스트를 입력하세요:")

    # 입력된 텍스트 출력
    if user_input:
        st.write(f"입력된 내용: {user_input}")

    st.header("S3에서 임베딩 데이터 로드 및 추출")

    # AWS S3 설정 입력
    bucket_name = st.text_input("S3 버킷 이름", "hemochat-rag-database")
    file_key = st.text_input("파일 키", "18_aga_tagged_embedded_data.json")

    # 데이터를 로드하고 벡터와 메타데이터 추출
    if st.button("데이터 로드"):
        try:
            embedded_data = load_data_from_s3(bucket_name, file_key)
            vectors, metadatas = extract_vectors_and_metadata(embedded_data)
            st.write("임베딩 벡터 개수:", len(vectors))
            st.write("메타데이터 예시:", metadatas[:1])  # 첫 번째 메타데이터 예시 출력
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
