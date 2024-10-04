# To modulization each process as function -> to minimalize main()

import streamlit as st
import openai
import boto3
import json
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity


# OpenAI API 키 설정 (Streamlit secrets 사용)
openai.api_key = st.secrets["openai"]["openai_api_key"]


# 사용자 정보 및 입력을 수집하는 함수
def collect_user_input():
    user_input = st.text_area("임상노트를 입력하세요:", height=500)
    st.subheader("어떤 분야에 종사하시나요?")
    occupation = st.radio(
        "직업을 선택하세요:",
        options=["의사", "간호사", "병원내 청구팀", "기타"],
        index=0
    )

    if occupation == "기타":
        other_occupation = st.text_input("직업을 입력해주세요:")
    else:
        other_occupation = None

    department = None
    if occupation:
        st.subheader("어떤 분과에 재직 중인지 알려주세요.")
        department = st.selectbox(
            "분과를 선택하세요:",
            options=[
                "신경외과 (Neuro-Surgery)",
                "혈관외과 (Vascular Surgery)",
                "대장항문외과 (Colorectal Surgery)"
            ]
        ) # 여기까지 해서 분과를 department으로 입력받음

    return occupation, other_occupation, department, user_input


# 분과 데이터셋: 추가될 때마다 업데이트 할 부분
department_datasets = {
    "신경외과 (Neuro-Surgery)": {
        "bucket_name": "hemochat-rag-database",
        "file_key": "18_aga_tagged_embedded_data.json"
    }, # 이후에는 외과학회 염두에 둔 데이터셋 먼저 넣기
    "혈관외과 (Vascular Surgery)": {
        "bucket_name": "hemochat-rag-database",
        "file_key": "vascular_embedded_data.json" # 나중에 이 이름으로 파일 생성필요
    },
    "대장항문외과 (Colorectal Surgery)": {
        "bucket_name": "hemochat_rag_database",
        "file_key": "colorectal_embedded_data.json"
    }
}


# 선택된 분과에 따라 해당 데이터셋을 로드하는 함수
def load_data_if_department_selected(department):
    # 선택한 분과가 매핑에 존재하는지 먼저 확인
    if department in department_datasets:
        dataset_info = department_datasets[department]
        bucket_name = dataset_info["bucket_name"]
        file_key = dataset_info["file_key"]

        st.write(f"{department} 데이터 로드 중...")
        return load_data_from_s3(bucket_name, file_key)
    else:
        st.warning(f"현재 {department}에 대한 데이터셋은 준비 중입니다.")
        return [], []


# 사용자 입력을 구조화하는 함수
def structure_user_input(user_input):
    try:
        # 프롬프트 템플릿 불러오기 (secrets 사용)
        prompt_template = st.secrets["openai"]["prompt_structuring"]
        # 프롬프트 작성
        prompt = prompt_template.format(user_input=user_input)

        # GPT-4o-mini 모델 호출
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 의료 기록을 구조화하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5,
        )

        structured_input = response.choices[0].message.content.strip()
        return structured_input

    except Exception as e:
        st.error(f"입력 구조화 중 오류 발생: {e}")
        return None


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
        # item이 딕셔너리인지 확인
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
                    st.write(f"문제가 있는 임베딩 데이터 내용: {item['임베딩']}")
                    continue  # 문제가 있는 항목은 무시하고 다음 항목으로 이동
            else:
                st.warning(f"필수 키가 누락된 아이템 발견 (인덱스 {idx}): {item}")
        else:
            st.warning(f"비정상적인 데이터 형식의 아이템 발견 (인덱스 {idx}): {item}")
    
    # 최종적으로 추출된 데이터 구조 확인
    # st.write("추출된 벡터의 수:", len(vectors))
    # st.write("추출된 메타데이터의 수:", len(metadatas))
    
    return vectors, metadatas


# 사용자 입력을 임베딩하는 함수
def get_embedding_from_openai(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']


# 코사인 유사도를 계산하여 상위 5개 결과 반환
def find_top_n_similar(embedding, vectors, metadatas, top_n=5):
    # 벡터와 메타데이터의 길이 확인
    if len(vectors) != len(metadatas):
        st.error(f"벡터 수와 메타데이터 수가 일치하지 않습니다: 벡터 수 = {len(vectors)}, 메타데이터 수 = {len(metadatas)}")
        return []

    # 사용자 임베딩 벡터를 2차원 배열로 변환
    user_embedding = np.array(embedding).reshape(1, -1)
    
    # 모든 벡터와의 코사인 유사도 계산
    similarities = cosine_similarity(user_embedding, vectors).flatten()
    
    # 유사도가 높은 순서대로 인덱스 정렬
    top_indices = similarities.argsort()[-top_n:][::-1]

    # 상위 결과 출력
    top_results = [{"유사도": similarities[i], "메타데이터": metadatas[i]} for i in top_indices]
    
    return top_results


# 사용자 입력 처리 및 임베딩 생성
def process_user_input(user_input):
    with st.spinner("사용자 입력 분석중..."):
        structured_input = structure_user_input(user_input)
        if not structured_input:
            st.error("입력 텍스트 분석에 실패했습니다.")

    st.write("입력 처리 완료!")
    st.write(structured_input) # 이 부분은 필요에 따라 숨겨도 됨.

    with st.spinner("임베딩 생성 중..."):
        embedding = get_embedding_from_openai(structured_input)
        if not embedding:
            st.error("임베딩 생성에 문제가 있습니다.")
            return None
    
    st.write("임베딩 생성 완료!")
    return structured_input, embedding


# GPT-4 모델을 사용하여 연관성 점수를 평가하는 함수
def evaluate_relevance_with_gpt(structured_input, items):
    try:
        # 프롬프트 템플릿 불러오기 (secrets 사용)
        prompt_template = st.secrets["openai"]["prompt_scoring"]
        # 항목들을 포맷에 맞게 나열
        formatted_items = "\n\n".join([f"항목 {i+1}: {item['요약']}" for i, item in enumerate(items)])
        # 프롬프트 작성
        prompt = prompt_template.format(user_input=structured_input, items=formatted_items)

        # GPT-4 모델 호출
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 도움이 되는 어시스턴트입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )

        # 응답에서 평가 점수 추출
        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        st.error(f"GPT 모델 호출 중 오류 발생: {e}")
        return None


# 검색 결과 및 분석 결과를 출력하는 함수
def display_results(embedding, vectors, metadatas, structured_input):
    top_results = find_top_n_similar(embedding, vectors, metadatas)
    st.subheader("상위 유사 항목")
    for idx, result in enumerate(top_results, 1):
        with st.expander(f"항목 {idx} - {result['메타데이터']['제목']}"):
            st.write(f"제목: {result['메타데이터']['제목']}")
            st.write(f"요약: {result['메타데이터']['요약']}")

    items = [result['메타데이터'] for result in top_results]

    with st.spinner("AI assist의 연관성 평가 중..."):
        full_response = evaluate_relevance_with_gpt(structured_input, items)

    if full_response:
        st.subheader("연관성 평가 결과")
        with st.expander("연관성 평가 결과 상세 보기"):
            st.write(full_response)

        relevant_results = []
        for idx, doc in enumerate(top_results, 1):
            score_match = re.search(rf"항목 {idx}:\s*(\d+)", full_response)
            if score_match:
                score = int(score_match.group(1))
                if score >= 7:
                    with st.expander(f"항목 {idx} (score: {score})"):
                        st.text(f"세부인정사항: {doc['메타데이터']['세부인정사항']}")
                    relevant_results.append(doc['메타데이터'])
            else:
                st.warning(f"항목 {idx}의 점수를 추출하지 못했습니다. '삭감 여부 확인' 버튼을 다시한번 눌러주세요.")

        return relevant_results, full_response
    else:
        st.error("서버에서 응답을 받지 못했습니다.")
        return None, None


# 유효 기준에 대한 세부적인 분석과 심사
def analyze_criteria(relevant_results, user_input):
    explanations = []
    overall_decision = "삭감 안될 가능성 높음"

    prompt_template = st.secrets["openai"]["prompt_interpretation"]

    with st.spinner("개별 기준에 대한 심사 진행중..."):
        for idx, criteria in enumerate(relevant_results, 1):
            try:
                prompt = prompt_template.format(user_input=user_input, criteria=criteria['세부인정사항'])
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 의료 문서를 분석하는 보험 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10000,
                    temperature=0.3,
                )
                analysis = response['choices'][0]['message']['content'].strip()
                explanations.append(f"기준 {idx}에 대한 분석: \n{analysis}")
                if "의료비는 삭감됩니다." in analysis:
                    overall_decision = "삭감될 가능성 높음"
            except Exception as e:
                st.error(f"기준 {idx}에 대한 분석 중 오류 발생: {e}")
       
    return overall_decision, explanations


def main():
    st.title("의료비 삭감 판정 어시스트 - beta version.")

    # 1. 사용자 정보 및 입력 수집
    occupation, other_occupation, department, user_input = collect_user_input()

    # 2. 특정 분과 선택시 해당 분과의 데이터 로드
    if st.button("삭감 여부 확인"):
        # 사용자 정보가 입력 되어야지만 삭감판정 가능
        if len(user_input.strip()) < 10:
            st.error("유효한 임상 노트 입력이 필요합니다. 최소 10자 이상의 텍스트를 입력해 주세요.")
            return # 함수 종료됨, 이후 프로세스 진행 안됨

        embedded_data = load_data_if_department_selected(department)
        if embedded_data is None or not embedded_data:
            st.error("데이터 로드 실패, 또는 해당 분과의 데이터가 아직 없습니다.")
            return
        else: 
            vectors, metadatas = extract_vectors_and_metadata(embedded_data)
            st.write("해당 분과의 급여기준 로드 완료")
        
        # 3. 사용자의 입력 처리
        structured_input, embedding = process_user_input(user_input) 
        if not structured_input or not embedding:
            return
        
        # 4. 검색된 급여기준 및 분석 결과 출력
        relevant_results, full_response = display_results(embedding, vectors, metadatas, structured_input)
        if not relevant_results:
            st.warning("검색 결과 중 유효한 항목이 없습니다. 적절한 분과를 선택하셨는지 확인해주세요.")
            return
        
        # 5. 개별 기준에 대한 분석
        overall_decision, explanations = analyze_criteria(relevant_results, user_input)
        st.subheader("심사 결과")
        st.write(overall_decision)

        # 6. 개별 기준에 대한 분석 결과 표시
        st.subheader("개별 기준에 대한 심사 결과")
        for explanation in explanations:
            with st.expander("개별 분석 보기"):
                st.write(explanation)

if __name__ == "__main__":
    main()
