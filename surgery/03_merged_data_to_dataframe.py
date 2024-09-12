# To remove duplicated items in merged_data.json and return to dataframe csv

import json
import pandas as pd

# 로그: JSON 파일 로드 시작
print("1. JSON 파일 로드를 시작합니다...")

# JSON 파일 경로
json_file_path = 'merged_data.json'

# JSON 파일 로드
with open(json_file_path, 'r', encoding='utf-8') as file:
    merged_data = json.load(file)  # JSON 파일을 로드, 리스트로 변환됨

# 로그: JSON 파일 로드 완료
print("2. JSON 파일이 성공적으로 로드되었습니다.")
print(f"총 {len(merged_data)} 개의 항목이 로드되었습니다.\n")

# '항목', '제목', '세부인정사항'을 저장할 리스트
data = []

# 리스트를 순회하면서 "제목들"과 상위 "항목" 정보 추출
print("3. '제목들'을 추출하여 데이터 준비를 시작합니다...")
for idx, item in enumerate(merged_data, start=1):
    if "제목들" in item:  # 각 항목에서 "제목들"이 있는지 확인
        항목 = item["항목"]  # 상위 항목 정보
        for entry in item["제목들"]:
            # 각 "제목들"에 대해 "항목", "제목", "세부인정사항"을 함께 저장
            data.append({
                "항목": 항목,
                "제목": entry["제목"],
                "세부인정사항": entry["세부인정사항"]
            })
    # 로그: 각 항목이 처리될 때마다 로그 출력
    print(f"   항목 {idx} 처리 완료: '{항목}'에서 {len(item.get('제목들', []))}개의 제목을 추출했습니다.")

# pandas로 변환하여 데이터프레임으로 처리
print("\n4. DataFrame으로 변환 중...")
df = pd.DataFrame(data)

# 로그: 변환된 데이터프레임 정보 출력
print(f"   DataFrame으로 변환 완료. 총 {len(df)}개의 행이 생성되었습니다.\n")

# 중복된 행 제거 ("항목", "제목", "세부인정사항"을 기준으로)
print("5. 중복된 행 제거 중...")
df.drop_duplicates(subset=["항목", "제목", "세부인정사항"], inplace=True)

# 로그: 중복 제거 결과
print(f"   중복 제거 완료. 총 {len(df)}개의 행이 남았습니다.\n")

# 데이터프레임 확인 (상위 5개 항목만 출력)
print("6. 최종 데이터프레임 미리보기 (상위 5개 항목):")
print(df.head())

# pandas DataFrame을 CSV로 저장
print("\n7. CSV 파일 저장 중...")
df.to_csv('cleaned_data.csv', index=False, encoding='utf-8-sig')  # 한글 지원을 위해 encoding='utf-8-sig' 사용

# 로그: 저장 완료 메시지
print("CSV 파일로 저장 완료: 'cleaned_data.csv'")
