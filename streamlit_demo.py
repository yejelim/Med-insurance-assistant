import json
import numpy as np
import streamlit as st
import faiss
import openai
import re
import os

# Streamlit secrets에서 API 키 가져오기
openai.api_key = st.secrets["openai"]["api_key"]

# 현재 실행 중인 디렉토리에서 JSON 파일 경로 설정
current_dir = os.getcwd()  # 현재 앱의 실행 경로를 가져옵니다.
file_path = os.path.join(current_dir, '18_aga_tagged_embedded_data.json')  # JSON 파일 경로 설정

# JSON 파일 로드
with open(file_path, 'r', encoding='utf-8') as f:
    embedded_data = json.load(f)

# JSON에서 임베딩 벡터와 메타데이터 추출
vectors = []
metadatas = []
for item in embedded_data:
    vectors.append(np.array(item['임베딩']))
    metadatas.append({"요약": item["요약"], "세부인정사항": item["세부인정사항"]})

# FAISS 인덱스 생성 (벡터의 크기에 맞게 초기화)
dimension = len(vectors[0])  # 벡터의 차원 수를 가져옵니다.
index = faiss.IndexFlatL2(dimension)  # L2 거리 기반의 FAISS 인덱스를 생성합니다.
index.add(np.array(vectors))  # 벡터들을 FAISS 인덱스에 추가합니다.

# Streamlit UI 구성
st.title("Medical Insurance Determination with OpenAI")
user_input = st.text_area("Enter patient case description:")

if st.button("Analyze"):
    if user_input.strip():
        # 쿼리 텍스트를 임베딩 벡터로 변환
        response = openai.Embedding.create(input=user_input, model="text-embedding-ada-002")
        query_embedding = np.array(response['data'][0]['embedding']).reshape(1, -1)

        # FAISS 인덱스를 사용해 유사한 문서 검색
        D, I = index.search(query_embedding, k=5)
        
        # 검색 결과에서 메타데이터 가져오기
        similar_docs = [metadatas[i] for i in I[0]]

        # GPT를 통한 연관성 평가
        items = ""
        for idx, doc in enumerate(similar_docs, 1):
            items += f"항목 {idx}: {doc['요약']}\n\n"

        # 프롬프트 템플릿 불러오기
        prompt_template = st.secrets["app"]["prompt_template"]
        filled_prompt = prompt_template.format(user_input=user_input, items=items)

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # 사용하려는 모델 지정
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": filled_prompt}
            ],
            max_tokens=150,
            temperature=0.3,
        )
        
        full_response = response['choices'][0]['message']['content'].strip()

        relevant_results = []
        for idx, doc in enumerate(similar_docs, 1):
            score_match = re.search(rf"항목 {idx}:\s*(\d+)", full_response)  # raw string 사용
            if score_match:
                score = int(score_match.group(1))
                if score >= 7:  # 7점 이상인 항목만 추가
                    with st.expander(f"항목 {idx} (GPT Score: {score})"):
                        st.write(f"요약: {doc['요약']}")
                    relevant_results.append(doc)

        # 7점 이상을 부여한 항목들에 대해 개별 기준 분석 수행
        if relevant_results:
            decisions = []
            explanations = []

            for idx, criteria in enumerate(relevant_results, 1):
                decision_prompt_template = st.secrets["app"]["decision_prompt_template"]
                decision_prompt = decision_prompt_template.format(user_input=user_input, criteria=criteria['세부인정사항'])

                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",  # 사용하려는 모델 지정
                    messages=[
                        {"role": "system", "content": "You are an expert in analyzing medical documents."},
                        {"role": "user", "content": decision_prompt}
                    ],
                    max_tokens=150,
                    temperature=0.3,
                )

                analysis = response['choices'][0]['message']['content'].strip()
                explanations.append(f"Analysis of criteria {idx}:\n{analysis}")

                # 조건에 따라 결정을 리스트에 추가
                if "의료비는 삭감됩니다" in analysis:
                    decisions.append("의료비는 삭감됩니다")
                elif "추가질문이 필요합니다" in analysis:
                    decisions.append("추가질문이 필요합니다")
                else:
                    decisions.append("결과 불명확")

                with st.expander(f"Detailed Explanation for Criteria {idx}"):
                    st.markdown(analysis)

            final_decision = "의료비는 삭감됩니다" if "의료비는 삭감됩니다" in decisions else "추가질문이 필요합니다"

            st.subheader("Final Decision")
            st.write(final_decision)

        else:
            st.write("연관성 높은 항목이 없습니다.")
    else:
        st.write("Please enter a patient case description.")
