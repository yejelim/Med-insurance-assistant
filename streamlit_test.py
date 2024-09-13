import streamlit as st
import openai
import boto3
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API 키 설정 (Streamlit secrets 사용)
openai.api_key = st.secrets["openai"]["openai_api_key"]

# 사용자 입력을 임베딩하는 함수
def get_embedding_from_openai(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    # 임베딩 결과 반환
    return response['data'][0]['embedding']

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

# 코사인 유사도를 계산하여 상위 5개 결과 반환
def find_top_n_similar(embedding, vectors, metadatas, top_n=5):
    # 사용자 임베딩 벡터를 2차원 배열로 변환
    user_embedding = np.array(embedding).reshape(1, -1)
    # 모든 벡터와의 코사인 유사도 계산
    similarities = cosine_similarity(user_embedding, vectors).flatten()
    # 유사도가 높은 순서대로 인덱스 정렬
    top_indices = similarities.argsort()[-top_n:][::-1]
    # 상위 결과 출력
    top_results = [{"유사도": similarities[i], "메타데이터": metadatas[i]} for i in top_indices]
    return top_results

def main():
    # 제목 설정
    st.title("의료비 삭감 판정 모델 - beta version")

    # 텍스트 입력창을 가장 위로 이동
    st.subheader("임상노트를 붙여넣으세요.")
    user_input = st.text_area("여기에 텍스트를 입력하세요:", height=500)

    # 사용자 정보 입력
    st.subheader("어떤 분야에 종사하시나요?")
    occupation = st.radio(
        "직업을 선택하세요:",
        options=["의사", "간호사", "병원내 청구팀", "기타"],
        index=0
    )

    # '기타'를 선택하면 입력란 표시
    if occupation == "기타":
        other_occupation = st.text_input("직업을 입력해주세요:")

    # 의료인인 경우 분과 선택
    if occupation in ["의사", "간호사"]:
        st.subheader("의료인이라면 어떤 분과에 재직 중인지 알려주세요.")
        department = st.selectbox(
            "분과를 선택하세요:",
            options=[
                "가정의학부 (Family Medicine, FM)",
                "관상동맥질환 집중치료실 (Coronary Care Unit, CCU)",
                "내과 (Internal Medicine, IM)",
                # 나머지 분과는 동일
            ]
        )

    # S3에서 고정된 데이터 로드 설정
    bucket_name = "hemochat-rag-database"
    file_key = "18_aga_tagged_embedded_data.json"

    # S3에서 데이터 로드
    if st.button("데이터 로드"):
        try:
            embedded_data = load_data_from_s3(bucket_name, file_key)
            vectors, metadatas = extract_vectors_and_metadata(embedded_data)
            st.write("데이터 로드 및 처리 완료!")
            st.write(f"임베딩 벡터 개수: {len(vectors)}")
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")

    # '삭감 여부 확인' 버튼 추가
    if st.button("삭감 여부 확인"):
        if user_input:
            st.subheader("임베딩 생성 결과 및 유사도 분석")

            # OpenAI의 Ada-002 모델을 사용하여 사용자 입력 임베딩 생성
            try:
                embedding = get_embedding_from_openai(user_input)
                st.write("임베딩 생성 완료!")
                st.write(f"임베딩 벡터 크기: {len(embedding)}")
                st.write(embedding[:10])  # 첫 10개의 임베딩 값 예시 출력

                # 코사인 유사도를 계산하여 상위 5개의 결과 출력
                if vectors:
                    top_results = find_top_n_similar(embedding, vectors, metadatas)
                    st.subheader("상위 5개 유사 항목")
                    for result in top_results:
                        st.write(f"유사도: {result['유사도']:.4f}")
                        st.write(f"요약: {result['메타데이터']['요약']}")
                        st.write(f"세부인정사항: {result['메타데이터']['세부인정사항']}")
                        st.write("---")
                else:
                    st.warning("데이터셋이 로드되지 않았습니다. 먼저 데이터 로드를 진행하세요.")
            except Exception as e:
                st.error(f"임베딩 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
