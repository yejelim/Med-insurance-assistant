import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# CSV 파일 불러오기
df = pd.read_csv('cleaned_data.csv')

# 데이터 확인
print(df.head())

# ClinicalBERT 토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# 청킹 및 오버랩 적용 함수
MAX_LENGTH = 512  # BERT의 최대 토큰 길이
CHUNK_SIZE = 256  # 청크 크기
OVERLAP_SIZE = 50  # 오버랩 크기

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap_size):
        chunk = tokens[i:i + chunk_size]
        if chunk:  # 빈 청크가 발생하지 않도록 조건 추가
            chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

# 각 행의 '항목', '제목', '세부인정사항'을 결합하여 텍스트로 만듦
df['combined_text'] = df['항목'] + " " + df['제목'] + " " + df['세부인정사항']

# 결합된 텍스트 확인
print(df['combined_text'].head())

# ClinicalBERT 모델 불러오기
model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# GPU 사용 가능 시 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 각 문서에 대해 ClinicalBERT 임베딩을 계산하는 함수
def embed_document(text):
    text_chunks = chunk_text(text)  # 텍스트 청킹
    
    embeddings = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
        outputs = model(**inputs)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().detach().numpy()
        embeddings.append(chunk_embedding)
    
    # 청크별 임베딩의 평균 계산
    return np.mean(embeddings, axis=0)

# 각 문서의 임베딩 생성
df['embedding'] = df['combined_text'].apply(embed_document)

# 임베딩 데이터 확인
print(df[['combined_text', 'embedding']].head())

# 임베딩 데이터를 저장 (파이클 파일로 저장)
df[['항목', '제목', '세부인정사항', 'embedding']].to_pickle('clinicalbert_embeddings.pkl')

# CSV로 저장할 경우 임베딩을 문자열로 변환
# df['embedding_str'] = df['embedding'].apply(lambda x: ','.join(map(str, x)))
# df.to_csv('clinicalbert_embeddings.csv', index=False)
