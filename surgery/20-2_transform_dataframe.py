import json
import pandas as pd

# 원본 JSON 파일 경로
input_file = 'tagged_vascular_filtered_criterion.json'

# 변환된 JSON 파일을 저장할 경로
output_file = 'tagged_vascular_filtered_criterion_fixed.json'

# JSON 파일 로드
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 변환된 데이터를 저장할 리스트
flattened_data = []

# 데이터 변환
for item in data:
    항목 = item.get('항목', 'N/A')
    제목들 = item.get('제목들', [])
    
    # '제목들' 리스트가 비어있지 않은지 확인
    if 제목들:
        for 제목 in 제목들:
            new_entry = {
                '항목': 항목,
                '제목': 제목.get('제목', 'N/A'),
                '세부인정사항': 제목.get('세부인정사항', 'N/A'),
                '요약': 제목.get('요약', 'N/A'),
                '임베딩': 제목.get('임베딩', [])
            }
            flattened_data.append(new_entry)
    else:
        # '제목들'이 비어있을 경우 기본 값으로 추가
        new_entry = {
            '항목': 항목,
            '제목': 'N/A',
            '세부인정사항': 'N/A',
            '요약': 'N/A',
            '임베딩': []
        }
        flattened_data.append(new_entry)

# 변환된 데이터를 새로운 JSON 파일로 저장
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(flattened_data, f, ensure_ascii=False, indent=4)

print(f"변환된 데이터가 '{output_file}'에 저장되었습니다.")

# 변환된 데이터를 데이터프레임으로 생성
df = pd.DataFrame(flattened_data)

# 데이터프레임을 CSV 파일로 저장 (선택 사항)
csv_output_file = 'tagged_vascular_filtered_criterion_fixed.csv'
df.to_csv(csv_output_file, index=False, encoding='utf-8-sig')

print(f"데이터프레임이 '{csv_output_file}'에 저장되었습니다.")
