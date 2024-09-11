import json
import pandas as pd

# JSON 파일 경로
json_file_path = 'merged_data.json'

# JSON 파일 로드
with open(json_file_path, 'r', encoding='utf-8') as file:
    merged_data = json.load(file)  # JSON 파일을 로드, 리스트로 변환됨

# '항목', '제목', '세부인정사항'을 저장할 리스트
data = []

# 리스트를 순회하면서 "제목들"과 상위 "항목" 정보 추출
for item in merged_data:
    if "제목들" in item:  # 각 항목에서 "제목들"이 있는지 확인
        항목 = item["항목"]  # 상위 항목 정보
        for entry in item["제목들"]:
            # 각 "제목들"에 대해 "항목", "제목", "세부인정사항"을 함께 저장
            data.append({
                "항목": 항목,
                "제목": entry["제목"],
                "세부인정사항": entry["세부인정사항"]
            })

# 여러 키워드 값들 (키워드 리스트)
keywords = ["트리글리세라이드", "콜레스테롤", "혈류량", "초음파 검사", "다245", "전산화단층", "CT", "자기공명영상진단", "Angiograpy", "혈관조영술", "2가지 이상","동일 피부", "Arterial", "정맥포트법", "혈전제거술", "자205", "단단문합술", "혈관성형술"]  # 키워드 리스트

# pandas로 변환하여 데이터프레임으로 처리
df = pd.DataFrame(data)

# 키워드를 기준으로 필터링
# 키워드 리스트 중 하나라도 포함된 제목을 찾음 (대소문자 구분 없음)
filtered_df = df[df['제목'].str.contains('|'.join(keywords), case=False, na=False)]

# 중복된 행 제거
filtered_df.drop_duplicates(subset=["항목", "제목", "세부인정사항"], inplace=True)

# 데이터프레임 확인
print(filtered_df)

# pandas DataFrame을 CSV로 저장 (필요 시)
filtered_df.to_csv('vascular_filtered_criterion.csv', index=False)
print("CSV 파일로 저장 완료")
