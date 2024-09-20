import openai
import pandas as pd

# OpenAI API 키 설정 (여기에 실제 API 키를 넣으세요)
openai.api_key = 'your-api-key'

# GPT-4o-mini로 클러스터 기준 분석 요청 함수
def analyze_cluster_with_gpt(texts, cluster_num):
    # GPT에게 각 클러스터의 텍스트를 분석해 기준을 제시하게 요청
    prompt = f"다음은 {cluster_num}번 클러스터에 포함된 텍스트입니다:\n{texts}\n" \
             f"이 클러스터의 공통된 특징과 어떤 기준으로 구분되었는지 추론하여 설명해 주세요. 공통된 특징이나 기준을 발견하기 어렵다면 다시 분석할 것을 제안해주세요."
    
    # ChatCompletion을 사용하여 GPT 호출
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # GPT-4o-mini 모델을 사용
        messages=[
            {"role": "system", "content": "You are an assistant that analyzes cluster data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.5
    )
    
    # GPT로부터 받은 응답 반환
    return response['choices'][0]['message']['content'].strip()

# 클러스터 파일명 리스트
cluster_files = ['cluster_1.csv', 'cluster_2.csv', 'cluster_3.csv', 'cluster_4.csv']

# 각 클러스터 분석 결과를 저장할 리스트
cluster_analysis_results = []

# 각 클러스터에 대해 GPT 분석 수행
for cluster_file in cluster_files:
    # 클러스터 데이터 불러오기
    cluster_data = pd.read_csv(cluster_file)
    
    # 클러스터 번호 추출 (파일명에서 추출)
    cluster_num = cluster_file.split('_')[1].split('.')[0]
    
    # '세부인정사항' 열에서 텍스트 추출 (NaN 처리)
    all_texts = " ".join(cluster_data['세부인정사항'].fillna('').astype(str))
    
    # GPT-4o-mini로 클러스터 기준 분석
    cluster_analysis = analyze_cluster_with_gpt(all_texts, cluster_num)
    
    # 결과를 리스트에 저장
    cluster_analysis_results.append((cluster_num, cluster_analysis))

# 각 클러스터의 분석 결과 출력
for cluster_num, analysis in cluster_analysis_results:
    print(f"클러스터 {cluster_num} 분석 결과:\n{analysis}\n")
