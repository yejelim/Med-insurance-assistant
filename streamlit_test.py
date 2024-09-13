# This one is deployed with Streamlit Share 

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
    st.subheader("임상 노트를 복사해서 붙여넣으세요.")
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
    if occupation in ["의사", "간호사"]:
        st.subheader("의료인이라면 어떤 분과에 재직 중인지 알려주세요.")
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
            st.subheader("결과")
            st.write(f"입력된 내용: {user_input}")
            st.success("삭감 여부를 확인했습니다.")  # 결과 메시지 예시


'''
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
'''

if __name__ == "__main__":
    main()
