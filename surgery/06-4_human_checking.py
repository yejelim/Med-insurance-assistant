import pandas as pd

# 'cluster_1.csv' 파일 읽기
df = pd.read_csv('cluster_4.csv')

# 필요한 열('항목', '제목', '세부인정사항')만 추출
df_selected = df[['항목', '제목', '세부인정사항']]

# 새로운 CSV 파일로 저장
df_selected.to_csv('selected_columns_cluster_4.csv', index=False)

print("선택한 열만 포함된 CSV 파일이 저장되었습니다.")
