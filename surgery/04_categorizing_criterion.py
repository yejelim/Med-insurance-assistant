# To categorize criterion into few main categories: General, Materials etc.

import pandas as pd

# 기존 CSV 파일 경로
input_csv_file = 'cleaned_data.csv'

# CSV 파일 로드
df = pd.read_csv(input_csv_file)

# 로그: 데이터 로드 완료
print("1. 기존 CSV 파일 로드 완료.")
print(f"총 {len(df)}개의 행이 로드되었습니다.\n")

# 특정 '항목' 값을 기준으로 필터링
target_value = "치료재료에 관한 급여기준"  # 필터링하고자 하는 '항목' 값 (예: '일반사항')

# '항목' 값이 target_value와 같은 행들만 필터링
filtered_df = df[df['항목'] == target_value]

# 로그: 필터링 완료
print(f"2. '항목' 값이 '{target_value}'인 행들만 필터링 완료.")
print(f"총 {len(filtered_df)}개의 행이 필터링되었습니다.\n")

# 필터링된 데이터프레임을 새로운 CSV 파일로 저장
output_csv_file = 'material_criterion.csv'
filtered_df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')

# 로그: 새로운 CSV 파일 저장 완료
print(f"3. 새로운 CSV 파일로 저장 완료: '{output_csv_file}'")
