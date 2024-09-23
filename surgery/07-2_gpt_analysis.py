import os
import pandas as pd
import openai
import time

# 1. 데이터 로드 및 확인
data = pd.read_csv('clustered_data_with_labels.csv', encoding='utf-8')

print("데이터프레임 정보:")
print(data.info())

print("\n데이터프레임 샘플:")
print(data.head())

# 2. 클러스터별 텍스트 수집
clusters_texts = {}
for cluster in data['cluster_label'].unique():
    clusters_texts[cluster] = data[data['cluster_label'] == cluster]['세부인정사항'].dropna().tolist()

# 각 클러스터의 텍스트 수 확인
for cluster, texts in clusters_texts.items():
    print(f"Cluster {cluster}: {len(texts)} texts")

# 3. 클러스터별 텍스트를 별도의 파일로 저장
# a. 텍스트 파일로 저장
output_dir = 'clusters_texts'
os.makedirs(output_dir, exist_ok=True)

for cluster, texts in clusters_texts.items():
    # 파일 이름에 사용할 수 없는 문자를 대체 (필요시)
    safe_cluster_name = str(cluster).replace('/', '_').replace('\\', '_').replace(':', '_')
    file_path = os.path.join(output_dir, f"cluster_{safe_cluster_name}.txt")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    print(f"Cluster {cluster}의 텍스트가 {file_path}에 저장되었습니다.")

# b. Excel 파일의 시트로 저장
excel_path = 'clustered_texts.xlsx'

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for cluster, texts in clusters_texts.items():
        # 데이터프레임 생성
        cluster_df = pd.DataFrame({
            '세부인정사항': texts
        })
        
        # 시트 이름에 사용할 수 없는 문자를 대체 (필요시)
        safe_cluster_name = str(cluster).replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # 시트 이름이 31자 이하인지 확인 및 조정
        if len(f"Cluster_{safe_cluster_name}") > 31:
            safe_cluster_name = safe_cluster_name[:28] + '...'
        
        # 시트에 쓰기
        cluster_df.to_excel(writer, sheet_name=f"Cluster_{safe_cluster_name}", index=False)
    
    print(f"모든 클러스터의 텍스트가 {excel_path}에 저장되었습니다.")

# 4. GPT를 사용한 클러스터별 텍스트 분석
# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# 클러스터별 텍스트 분석
clusters_analysis = {}

for cluster, texts in clusters_texts.items():
    print(f"\nAnalyzing Cluster {cluster} with {len(texts)} texts...")
    
    # 텍스트 샘플 추출 (전체 텍스트가 많을 경우 샘플링)
    sample_texts = texts[:5]  # 상위 5개 텍스트 사용, 필요에 따라 조정 가능
    
    # 텍스트를 하나의 문자열로 결합
    combined_text = "\n".join(sample_texts)
    
    # GPT에게 보낼 프롬프트 작성
    prompt = (
        f"다음은 클러스터 {cluster}에 속하는 텍스트들의 샘플입니다:\n\n"
        f"{combined_text}\n\n"
        "이 클러스터의 텍스트들이 의미적으로 유사한지, 그리고 클러스터 내에서 일관된 주제가 있는지 평가해 주세요. "
        "또한, 이 클러스터가 어떤 주제를 다루고 있는지 간단히 요약해 주세요."
    )
    
    try:
        # OpenAI API를 사용하여 GPT에게 요청
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 또는 "gpt-4"
            messages=[
                {"role": "system", "content": "당신은 텍스트 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        # GPT의 응답 추출
        analysis = response.choices[0].message['content'].strip()
        clusters_analysis[cluster] = analysis
        
        print(f"Cluster {cluster} analysis completed.")
        
        # API 호출 제한 방지를 위해 잠시 대기
        time.sleep(1)  # 1초 대기, 필요에 따라 조정 가능
    except Exception as e:
        print(f"Error analyzing cluster {cluster}: {e}")  # 오류 메시지 출력
        clusters_analysis[cluster] = f"Analysis failed: {e}"

# 5. 분석 결과 저장
# a. 텍스트 파일로 저장
with open('clusters_analysis.txt', 'w', encoding='utf-8') as f:
    for cluster, analysis in clusters_analysis.items():
        f.write(f"=== Analysis for Cluster {cluster} ===\n")
        f.write(f"{analysis}\n\n")
print("\nClusters analysis have been saved to 'clusters_analysis.txt'")

# b. Excel 파일로 저장
analysis_df = pd.DataFrame(list(clusters_analysis.items()), columns=['Cluster Label', 'Analysis'])
analysis_df.to_excel('clusters_analysis.xlsx', index=False)
print("Clusters analysis have been saved to 'clusters_analysis.xlsx'")
