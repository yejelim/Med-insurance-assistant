import pandas as pd
import os

# 저장된 pickle 파일 경로 지정 (clinicalbert_embeddings_batch_ 파일들 불러오기)
pickle_files = [f for f in os.listdir() if f.startswith('clinicalbert_embeddings_batch_') and f.endswith('.pkl')]

# 여러 pickle 파일을 하나의 리스트에 저장
df_list = []

for file in pickle_files:
    df_batch = pd.read_pickle(file)
    df_list.append(df_batch)

# 여러 DataFrame을 하나로 결합
df_combined = pd.concat(df_list, ignore_index=True)

# 결합된 데이터 확인
print(df_combined.head())

# 결합된 데이터 최종 저장
# 1. Pickle 파일로 저장
df_combined.to_pickle('Clinicalbert_embeddings_combined.pkl')

# 2. CSV 파일로 저장
df_combined.to_csv('Clinicalbert_embeddings_combined.csv', index=False)

print("결합된 파일이 저장되었습니다.")
