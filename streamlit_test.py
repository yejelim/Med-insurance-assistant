import streamlit as st
import openai
import boto3
import json
import numpy as np
import re
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
    
    # embedded_data가 리스트인지 확인
    if not isinstance(embedded_data, list):
        st.error("임베딩 데이터가 리스트 형식이 아닙니다.")
        st.write("임베딩 데이터 구조 확인:", embedded_data)
        return [], []
    
    # 각 item이 예상한 딕셔너리인지 확인하고 필요한 정보 추출
    for idx, item in enumerate(embedded_data):
        if isinstance(item, dict):
            # 필요한 키가 모두 있는지 확인
            if all(key in item for key in ['임베딩', '제목', '요약', '세부인정사항']):
                try:
                    vectors.append(np.array(item['임베딩']))
                    metadatas.append({
                        "제목": item["제목"],
                        "요약": item["요약"],
                        "세부인정사항": item["세부인정사항"]
                    })
                except (TypeError, ValueError) as e:
                    st.warning(f"임베딩 데이터를 배열로 변환하는 중 오류 발생 (인덱스 {idx}): {e}")
            else:
                st.warning(f"필수 키가 누락된 아이템 발견 (인덱스 {idx}): {item}")
        else:
            st.warning(f"비정상적인 데이터 형식의 아이템 발견 (인덱스 {idx}): {item}")
    
    # 최종적으로 추출된 데이터 구조 확인
    st.write("추출된 벡터의 수:", len(vectors))
    st.write("추출된 메타데이터의 수:", len(metadatas))
    
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

# GPT-4 모델을 사용하여 연관성 점수를 평가하는 함수
def evaluate_relevance_with_gpt(user_input, items):
    # 프롬프트 템플릿 불러오기 (secrets 사용)
    prompt_template = st.secrets["openai"]["prompt_scoring"]
    # 항목들을 포맷에 맞게 나열
    formatted_items = "\n\n".join([f"항목 {i+1}: {item['요약']}" for i, item in enumerate(items)])
    # 프롬프트 작성
    prompt = prompt_template.format(user_input=user_input, items=formatted_items)

    # GPT-4 모델 호출
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # 응답에서 평가 점수 추출
    result = response.choices[0].message.content.strip()
    return result

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

    # '삭감 여부 확인' 버튼 추가
    if st.button("삭감 여부 확인"):
        if user_input:
            st.subheader("임베딩 생성 및 유사도 분석 시작")

            try:
                # 사용자 입력의 임베딩 생성
                embedding = get_embedding_from_openai(user_input)
                st.write("임베딩 생성 완료!")

                # S3에서 임베딩 데이터 로드
                bucket_name = "hemochat-rag-database"
                file_key = "18_aga_tagged_embedded_data.json"
                embedded_data = load_data_from_s3(bucket_name, file_key)
                vectors, metadatas = extract_vectors_and_metadata(embedded_data)
                st.write("S3 데이터 로드 및 처리 완료!")

                # 코사인 유사도를 계산하여 상위 5개의 결과 출력
                top_results = find_top_n_similar(embedding, vectors, metadatas)
                st.subheader("상위 5개 유사 항목")
                for result in top_results:
                    st.write(f"유사도: {result['유사도']:.4f}")
                    st.write(f"제목: {result['메타데이터']['제목']}")
                    st.write(f"요약: {result['메타데이터']['요약']}")
                    st.write("---")

                # GPT-4 모델을 사용하여 각 항목의 연관성 평가
                full_response = evaluate_relevance_with_gpt(user_input, [result['메타데이터'] for result in top_results])

                # 7점 이상 항목 필터링
                relevant_results = []
                for idx, doc in enumerate(top_results, 1):
                    score_match = re.search(rf"항목 {idx}:\s*(\d+)", full_response)
                    if score_match:
                        score = int(score_match.group(1))
                        if score >= 7:  # 7점 이상인 항목만 추가
                            with st.expander(f"항목 {idx} (GPT Score: {score})"):
                                st.write(f"세부인정사항: {doc['메타데이터']['세부인정사항']}")
                            relevant_results.append(doc['메타데이터'])

                # 7점 이상 항목에 대해 개별 분석 수행
                if relevant_results:
                    explanations = []
                    for idx, criteria in enumerate(relevant_results, 1):
                        prompt_template = st.secrets["openai"]["prompt_interpretation"]
                        response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are an expert in analyzing medical documents."},
                                {"role": "user", "content": prompt_template.format(user_input=user_input, criteria=criteria['세부인정사항'])}
                            ],
                            max_tokens=1000,
                            temperature=0.3,
                        )
                        analysis = response.choices[0].message.content.strip()
                        explanations.append(f"Analysis of criteria {idx}:\n{analysis}")
                    st.write("\n\n".join(explanations))

            except Exception as e:
                st.error(f"임베딩 생성 및 유사도 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
