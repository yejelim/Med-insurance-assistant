import pickle
import pandas as pd

# 결과가 저장된 파일 로드
with open('3_clustered_data_hierarchical_labels_limited.pkl', 'rb') as f:
    data = pickle.load(f)

# 데이터가 pandas DataFrame 형식인지 확인
if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

# 클러스터별 데이터 개수 확인
cluster_counts = data['cluster_label'].value_counts().sort_index()
print("Cluster Counts:")
print(cluster_counts)

# 각 클러스터별 데이터 샘플 보기
for cluster in cluster_counts.index:
    print(f"\nCluster: {cluster}")
    print(data[data['cluster_label'] == cluster].head())  # 각 클러스터의 상위 5개 데이터 출력
