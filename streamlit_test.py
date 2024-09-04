import streamlit as st

def main():
    # 제목 설정
    st.title("RAG 기반 LLM 모델 테스트")

    # 텍스트 입력창
    user_input = st.text_input("여기에 텍스트를 입력하세요:")

    # 입력된 텍스트 출력
    if user_input:
        st.write(f"입력된 내용: {user_input}")

if __name__ == "__main__":
    main()
