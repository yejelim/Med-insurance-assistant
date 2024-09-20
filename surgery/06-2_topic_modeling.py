import pandas as pd
import gensim
from gensim import corpora
import re
from nltk.corpus import stopwords
import csv

# 한국어 불용어 리스트 (직접 추가 가능)
stop_words = set(['경우', '및', '시행', '고시', '인정함', '-', '(고시)'])

# 텍스트 전처리 함수 (특수문자 및 불용어 제거)
def preprocess_text(text):
    # 특수 문자 및 숫자 제거
    text = re.sub(r'[^가-힣\s]', '', text)  # 한글과 공백을 제외한 문자 제거
    # 공백 제거 및 불용어 처리
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# 클러스터별 텍스트에서 주제 모델링(LDA) 적용하는 함수
def apply_topic_modeling_to_cluster(cluster_data, num_topics=3):
    # NaN을 빈 문자열로 대체하고, 숫자형 데이터를 문자열로 변환
    cluster_data['제목'] = cluster_data['제목'].fillna('').astype(str)
    
    # 텍스트 전처리 (특수문자 제거 및 불용어 처리)
    tokenized_texts = [preprocess_text(text) for text in cluster_data['제목']]
    
    # 단어 딕셔너리 및 코퍼스 생성
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # LDA 모델 학습
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    # 각 주제별 키워드 출력
    topics = lda_model.print_topics(num_words=5)
    return topics

# 클러스터 파일명 리스트 (CSV 파일 리스트로 변경)
cluster_files = ['cluster_1.csv', 'cluster_2.csv', 'cluster_3.csv', 'cluster_4.csv']

# CSV에 저장할 결과 리스트 (각 클러스터별 주제 및 키워드)
csv_data = []

# 각 클러스터별로 주제 모델링 수행 후 결과 저장
for cluster_file in cluster_files:
    # 클러스터 데이터 불러오기 (CSV 파일로 변경)
    cluster_data = pd.read_csv(cluster_file)
    
    # 클러스터 번호 추출 (파일명에서 번호 추출)
    cluster_num = cluster_file.split('_')[1].split('.')[0]
    
    # 클러스터 내 주제 모델링 수행
    topics = apply_topic_modeling_to_cluster(cluster_data)
    
    # 주제 결과를 리스트로 저장
    for topic_num, topic_keywords in topics:
        csv_data.append([cluster_num, topic_num, topic_keywords])

# 결과를 CSV 파일로 저장
csv_filename = 'cluster_topics_for_4.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # 헤더 작성
    writer.writerow(['Cluster', 'Topic', 'Keywords'])
    
    # 각 클러스터별 주제와 키워드 작성
    writer.writerows(csv_data)

print(f"주제 모델링 결과가 '{csv_filename}'로 저장되었습니다.")
