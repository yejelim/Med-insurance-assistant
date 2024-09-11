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

# 여러 specific_value 값들
specific_values = ["주 2회의 개념", "화장품으로 인한 피부 과민반응으로 접촉성피부염이 생긴 경우 급여여부"]  # 여러 관심 있는 제목 값

# "제목" 값이 specific_values 목록에 있는 항목들만 필터링
filtered_data = [item for item in data if item["제목"] in specific_values]

# pandas로 변환하여 데이터프레임으로 처리
df = pd.DataFrame(filtered_data)

# 중복된 행 제거
df.drop_duplicates(subset=["항목", "제목", "세부인정사항"], inplace=True)

# 데이터프레임 확인
print(df)

# pandas DataFrame을 CSV로 저장 (필요 시)
# df.to_csv('filtered_titles_with_items.csv', index=False)
# print("CSV 파일로 저장 완료")
