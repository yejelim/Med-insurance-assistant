# Dendrogram test with visualization before real clustering
import pandas as pd
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np

# 결합된 임베딩 데이터 불러오기 (pkl 파일)
df_combined = pd.read_pickle('05-3_indexing.pkl')

# 임베딩 벡터 추출
embeddings = np.vstack(df_combined['embedding'].values)

# 계층적 클러스터링을 위한 linkage 계산 (ward 방식 사용)
Z = linkage(embeddings, method='ward')

# 덴드로그램 시각화
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='level', p=5)  # 상위 5개 레벨만 표시
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
