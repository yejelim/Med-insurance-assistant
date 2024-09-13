import streamlit as st
import openai

# OpenAI API 키 설정 (Streamlit secrets 사용)
openai.api_key = st.secrets[openai]["openai_api_key"]

# 사용자 입력을 임베딩하는 함수
def get_embedding_from_openai(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    # 임베딩 결과 반환
    return response['data'][0]['embedding']

def main():
    # 제목 설정
    st.title("의료비 삭감 판정 모델 - beta version")

    # 텍스트 입력창을 가장 위로 이동
    st.subheader("임상노트를 붙여넣으세요.")
    user_input = st.text_area("여기에 텍스트를 입력하세요:", height=200)

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
            st.subheader("임베딩 생성 결과")

            # OpenAI의 Ada-002 모델을 사용하여 사용자 입력 임베딩 생성
            try:
                embedding = get_embedding_from_openai(user_input)
                st.write("임베딩 생성 완료!")
                st.write(f"임베딩 벡터 크기: {len(embedding)}")
                st.write(embedding[:10])  # 첫 10개의 임베딩 값 예시 출력
            except Exception as e:
                st.error(f"임베딩 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
