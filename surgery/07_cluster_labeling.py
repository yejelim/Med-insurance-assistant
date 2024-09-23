import pickle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# 1. 데이터 로드 및 확인
with open('valid_cluster_3.pkl', 'rb') as f:
    data = pickle.load(f)

if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

print("데이터프레임 정보:")
print(data.info())

print("\n데이터프레임 샘플:")
print(data.head())

# 2. 임베딩 벡터 변환
try:
    embeddings = np.array(data['embedding'].tolist())
except AttributeError as e:
    raise ValueError("임베딩 벡터가 리스트 형태로 저장되어 있지 않습니다. 데이터를 확인하세요.") from e

print(f"\n임베딩 벡터의 형태: {embeddings.shape}")

# 3. 데이터프레임 인덱스 재설정
data = data.reset_index(drop=True)
print("\n인덱스 재설정 후 데이터프레임 정보:")
print(data.info())

# 4. 계층적 군집 분석 수행
Z = linkage(embeddings, method='ward')
print("\n링크리지 행렬의 형태:", Z.shape)

# 5. 클러스터 레이블 할당
root, nodes = to_tree(Z, rd=True)

def traverse_and_label_combined(node, current_label, labels_dict, max_depth, current_depth=0):
    if node.is_leaf():
        labels_dict[node.id] = current_label
    elif current_depth < max_depth:
        traverse_and_label_combined(node.left, f"{current_label}_1", labels_dict, max_depth, current_depth + 1)
        traverse_and_label_combined(node.right, f"{current_label}_2", labels_dict, max_depth, current_depth + 1)
    else:
        for leaf in get_leaves(node):
            labels_dict[leaf.id] = current_label

def get_leaves(node):
    """주어진 노드의 모든 리프 노드를 반환"""
    if node.is_leaf():
        return [node]
    else:
        return get_leaves(node.left) + get_leaves(node.right)

# 최대 계층 깊이 설정 (예: 3)
MAX_DEPTH = 3

labels_dict = {}
traverse_and_label_combined(root, 'A', labels_dict, MAX_DEPTH)

# 레이블 할당 확인
print(f"\n총 할당된 레이블 수: {len(labels_dict)}")
print(f"총 데이터 포인트 수: {len(data)}")

if len(labels_dict) != len(data):
    missing = set(range(len(data))) - set(labels_dict.keys())
    print(f"경고: 클러스터 레이블이 할당되지 않은 데이터 인덱스: {missing}")
else:
    print("모든 데이터 포인트에 클러스터 레이블이 할당되었습니다.")

# 데이터프레임에 클러스터 레이블 추가
data['cluster_label'] = data.index.map(labels_dict)

# NaN 확인
num_nan = data['cluster_label'].isnull().sum()
print(f"\n클러스터 레이블이 누락된 데이터 포인트 수: {num_nan}")

# 6. 클러스터링 결과 확인
cluster_counts = data['cluster_label'].value_counts().sort_index()
print("\nCluster Counts:")
print(cluster_counts)

for cluster in cluster_counts.index:
    print(f"\nCluster: {cluster}")
    cluster_data = data[data['cluster_label'] == cluster]
    print(cluster_data[['index', '세부인정사항', 'cluster']].head())

# 7. 클러스터 품질 평가 (실루엣 스코어)
# 실루엣 스코어는 두 개 이상의 클러스터가 존재할 때만 계산 가능
if cluster_counts.shape[0] > 1:
    silhouette_avg = silhouette_score(embeddings, data['cluster_label'])
    print(f"\nSilhouette Score: {silhouette_avg}")
else:
    print("\n실루엣 스코어를 계산할 클러스터가 충분하지 않습니다.")

# 8. 클러스터 시각화
# 8.1. 덴드로그램 시각화
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='level', p=MAX_DEPTH, color_threshold=None)
plt.title('Hierarchical Clustering Dendrogram (Max Depth)')
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.show()

# 8.2. t-SNE 시각화
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

data['tsne_1'] = embeddings_2d[:, 0]
data['tsne_2'] = embeddings_2d[:, 1]

plt.figure(figsize=(12, 8))
sns.scatterplot(x='tsne_1', y='tsne_2', hue='cluster_label', palette='tab10', data=data, legend='full', alpha=0.7)
plt.title('t-SNE Visualization of Hierarchical Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Cluster Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 9. 클러스터별 평균 텍스트 길이 계산 및 시각화 (선택 사항)
# '세부인정사항' 열이 텍스트 데이터라고 가정
if '세부인정사항' in data.columns:
    data['text_length'] = data['세부인정사항'].apply(len)
    cluster_text_length = data.groupby('cluster_label')['text_length'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='cluster_label', y='text_length', data=cluster_text_length)
    plt.title('Average Text Length per Cluster')
    plt.xlabel('Cluster Label')
    plt.ylabel('Average Text Length')
    plt.xticks(rotation=90)
    plt.show()
else:
    print("\nWarning: '세부인정사항' 열이 데이터프레임에 존재하지 않습니다. 텍스트 열 이름을 확인하세요.")

# 10. 특정 클러스터의 샘플 텍스트 확인 (예시)
specific_cluster = 'A_1'
if '세부인정사항' in data.columns:
    cluster_data = data[data['cluster_label'] == specific_cluster]
    print(f"\nSample texts from cluster {specific_cluster}:")
    print(cluster_data['세부인정사항'].head(10))
else:
    print("\nWarning: '세부인정사항' 열이 데이터프레임에 존재하지 않습니다. 텍스트 열 이름을 확인하세요.")

# 11. 클러스터링 결과 저장
data.to_csv('clustered_data_with_labels.csv', index=False)
print("\nClustered data with labels have been saved to 'clustered_data_with_labels.csv'")
