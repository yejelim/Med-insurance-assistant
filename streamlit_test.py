import streamlit as st
import openai
import boto3
import json
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API 키 설정 (Streamlit secrets 사용)
openai.api_key = st.secrets["openai"]["openai_api_key"]

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
            max_tokens=5000,
            temperature=0.5,
        )

        structured_input = response.choices[0].message.content.strip()
        return structured_input

    except Exception as e:
        st.error(f"입력 구조화 중 오류 발생: {e}")
        return None

# 사용자 입력을 임베딩하는 함수
def get_embedding_from_openai(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
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

def main():
    # 제목 설정
    st.title("의료비 삭감 판정 모델 - beta version")

    # 텍스트 입력창을 가장 위로 이동
    st.subheader("임상노트를 붙여넣으세요.")
    user_input = st.text_area("여기에 텍스트를 입력하세요:", height=500)

    # 사용자 정보 입력
    st.subheader("환영합니다. 어떤 분야에 종사하시나요?")
    occupation = st.radio(
        "직업을 선택하세요:",
        options=["의사", "간호사", "병원내 청구팀", "기타"],
        index=0
    )
    # '기타'를 선택하면 입력란 표시
    if occupation == "기타":
        other_occupation = st.text_input("직업을 입력해주세요:")
    # 의료인인 경우 분과 선택
    if occupation in ["의사", "간호사", "병원내 청구팀"]:
        st.subheader("어떤 분과에 재직 중인지 알려주세요.")
        department = st.selectbox(
            "분과를 선택하세요:",
            options=[
                "가정의학부 (Family Medicine, FM)",
                "관상동맥질환 집중치료실 (Coronary Care Unit, CCU)",
                "내과 (Internal Medicine, IM)",
                "내과계 중환자실 (Medical Intensive Care Unit, MICU)",
                "내분비내과 (Endocrinology, ED)",
                "마취과 (Anesthesiology, AN)",
                "분만실 (Delivery Room, DR)",
                "비뇨기과 (Urology, URO)",
                "산부인과 (Obstetrics/Gynecology, OB/GY)",
                "성형외과 (Plastic Surgery, PS)",
                "소아과 (Pediatrics, PD)",
                "소화기내과 (Gastrointestinal Medicine, GI)",
                "수술실 (Operating Room, OR)",
                "신경과 (Neurology, NR)",
                "신경외과 (Neuro-Surgery, NS)",
                "신경정신과 (Neuro-Psychiatry, NP)",
                "신생아중환자실 (Neonatal Intensive Care Unit, NICU)",
                "신장내과 (Nephrology, NH)",
                "심장내과 (Cardiovascular Medicine, CV)",
                "안과 (Ophthalmology, OPH(PT))",
                "외과계중환자실 (Surgical Intensive Care Unit, SICU)",
                "응급처치부 (Emergency Service, ER)",
                "이비인후과 (Ear, Nose & Throat, ENT)",
                "일반외과 (General Surgery, GS)",
                "정신과 (Psychiatry, PY)",
                "정형외과 (Orthopedic Surgery, OS)",
                "중환자실 (Intensive Care Unit, ICU)",
                "치과 (Dentistry, DN)",
                "피부과 (Dermatology, DM)",
                "혈액종양학과 (Hematology-Oncology, HO)",
                "회복실 (Postanasthesia Care Unit, PACU)",
                "흉부내과 (Chest Medicine, CM)",
                "흉부외과 (Chest Surgery, CS)"
            ]
        )

    # '삭감 여부 확인' 버튼 추가
    if st.button("삭감 여부 확인"):
        if user_input:
            st.subheader("입력 구조화 및 임베딩 생성 시작")

            try:
                # 1. 사용자 입력을 구조화
                with st.spinner("입력 구조화 중..."):
                    structured_input = structure_user_input(user_input)
                    if not structured_input:
                        st.error("입력 구조화에 실패했습니다.")
                        return

                st.write("입력 구조화 완료!")
                st.write(structured_input)

                # 2. 구조화된 입력을 임베딩
                with st.spinner("임베딩 생성 중..."):
                    embedding = get_embedding_from_openai(structured_input)
                    if not embedding:
                        st.error("임베딩 생성에 실패했습니다.")
                        return

                st.write("임베딩 생성 완료!")

                # 3. S3에서 임베딩 데이터 로드
                bucket_name = "hemochat-rag-database"
                file_key = "18_aga_tagged_embedded_data.json"
                embedded_data = load_data_from_s3(bucket_name, file_key)
                vectors, metadatas = extract_vectors_and_metadata(embedded_data)
                st.write("해당 분과의 급여기준 데이터 로드 완료!")

                # 4. 코사인 유사도를 계산하여 상위 결과 출력
                top_results = find_top_n_similar(embedding, vectors, metadatas)
                st.subheader("상위 유사 항목")
                for idx, result in enumerate(top_results, 1):
                    with st.expander(f"항목 {idx} - {result['메타데이터']['제목']}"):
                        # st.write(f"유사도: {result['유사도']:.4f}")
                        st.write(f"제목: {result['메타데이터']['제목']}")
                        st.write(f"요약: {result['메타데이터']['요약']}")

                # 5. 'top_results'의 메타데이터를 'items'로 정의하여 'evaluate_relevance_with_gpt'로 전달
                items = [result['메타데이터'] for result in top_results]
                # st.write("evaluate_relevance_with_gpt로 전달된 items:", items)

                # 6. 연관성 평가
                with st.spinner("GPT-4를 사용하여 연관성 평가 중..."):
                    full_response = evaluate_relevance_with_gpt(structured_input, items)

                if full_response:
                    st.subheader("GPT-4 연관성 평가 결과")
                    with st.expander("연관성 평가 상세 보기"):
                        st.write(full_response)

                    # 7점 이상 항목 필터링 및 개별 기준 분석
                    relevant_results = []
                    for idx, doc in enumerate(top_results, 1):
                        score_match = re.search(rf"항목 {idx}:\s*(\d+)", full_response)
                        if score_match:
                            score = int(score_match.group(1))
                            if score >= 7:
                                with st.expander(f"항목 {idx} (GPT Score: {score})"):
                                    st.text(f"세부인정사항: {doc['메타데이터']['세부인정사항']}")
                                relevant_results.append(doc['메타데이터'])
                        else:
                            st.warning(f"항목 {idx}의 점수를 추출하지 못했습니다.")

                    if relevant_results:
                        explanations = []
                        overall_decision = "인정"  # 기본값을 "인정"으로 설정

                        # 프롬프트 템플릿 불러오기
                        prompt_template = st.secrets["openai"]["prompt_interpretation"]

                        with st.spinner("개별 기준에 대한 분석 중..."):
                            for idx, criteria in enumerate(relevant_results, 1):
                                try:
                                    # 프롬프트 작성
                                    prompt = prompt_template.format(
                                        user_input=user_input,  # 원문 텍스트 사용
                                        criteria=criteria['세부인정사항']
                                    )

                                    response = openai.ChatCompletion.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": "당신은 의료 문서를 분석하는 전문가입니다."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        max_tokens=10000,
                                        temperature=0.3,
                                    )

                                    analysis = response['choices'][0]['message']['content'].strip()
                                        
                                    explanations.append(f"기준 {idx}에 대한 분석:\n{analysis}")

                                    # 심사 결과 확인
                                    if "의료비는 삭감됩니다." in analysis:
                                        overall_decision = "삭감"  # 하나라도 "삭감"이 있으면 전체 결과를 "삭감"으로 설정
                                except Exception as e:
                                    st.error(f"기준 {idx}에 대한 분석 중 오류 발생: {e}")

                        # 심사 결과 표시
                        st.subheader("심사 결과")
                        st.write(overall_decision)

                        # 개별 기준에 대한 분석 결과 표시
                        st.subheader("각 기준에 대한 GPT-4 분석 결과")
                        with st.expander(f"분석 보기"):
                            st.write("\n\n".join(explanations))
                    else:
                        st.warning("7점 이상인 항목이 없습니다.")
                else:
                    st.error("GPT 모델의 응답을 받지 못했습니다.")

            except Exception as e:
                st.error(f"프로세스 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
