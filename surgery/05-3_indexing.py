import pandas as pd

# 결합된 임베딩 데이터 불러오기 (pkl 파일)
df_combined = pd.read_pickle('Clinicalbert_embeddings_combined.pkl')

# 고유 인덱스 추가 (DataFrame 인덱스를 사용하거나 새로 생성)
df_combined['index'] = df_combined.index

# 결합된 데이터 최종 저장 (인덱스 추가된 상태로)
df_combined.to_pickle('05-3_indexing.pkl')
df_combined.to_csv('05-3_indexing.csv', index=False)

print("결합된 파일에 인덱스를 추가하여 저장하였습니다.")
