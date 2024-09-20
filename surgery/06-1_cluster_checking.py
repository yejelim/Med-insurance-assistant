import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np

# 1. 결합된 임베딩 데이터 불러오기 (pkl 파일)
df_combined = pd.read_pickle('05-3_indexing.pkl')

# 2. 임베딩 벡터 추출
embeddings = np.vstack(df_combined['embedding'].values)

# 3. 계층적 클러스터링을 위한 linkage 계산 (ward 방식 사용)
Z = linkage(embeddings, method='ward')

# 5. 임계값 설정 및 클러스터링 수행 (예: distance = 60)
threshold_distance = 60
cluster_labels = fcluster(Z, t=threshold_distance, criterion='distance')

# 6. 각 텍스트에 해당 클러스터 레이블 할당
df_combined['cluster'] = cluster_labels

# 7. 각 클러스터별 데이터 pkl 및 csv 파일로 저장
for cluster_num in sorted(set(cluster_labels)):
    cluster_data = df_combined[df_combined['cluster'] == cluster_num]
    
    # Pickle 파일로 저장
    pkl_filename = f'cluster_{cluster_num}.pkl'
    cluster_data.to_pickle(pkl_filename)
    
    # CSV 파일로 저장
    csv_filename = f'cluster_{cluster_num}.csv'
    cluster_data.to_csv(csv_filename, index=False)  # index=False로 인덱스를 저장하지 않음
    
    print(f"클러스터 {cluster_num} 데이터가 '{pkl_filename}' 및 '{csv_filename}'로 저장되었습니다.")
