import pandas as pd
import os
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# CSV 파일 불러오기
df = pd.read_csv('cleaned_data.csv')

# ClinicalBERT 토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# 청킹 및 오버랩 적용 함수
MAX_LENGTH = 512  # BERT의 최대 토큰 길이
CHUNK_SIZE = 256  # 청크 크기
OVERLAP_SIZE = 50  # 오버랩 크기
BATCH_SIZE = 100  # 배치 크기 (한 번에 처리할 문서 수)
total_data = len(df)
total_batches = (total_data + BATCH_SIZE -1) // BATCH_SIZE
print(f"총 배치 개수: {total_batches}" )

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap_size):
        chunk = tokens[i:i + chunk_size]
        if chunk:  # 빈 청크가 발생하지 않도록 조건 추가
            chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

# 각 행의 '항목', '제목', '세부인정사항'을 결합하여 텍스트로 만듦
# NaN 값을 빈 문자열로 대체
df['combined_text'] = df['항목'].fillna('') + " " + df['제목'].fillna('') + " " + df['세부인정사항'].fillna('')

# 결합된 텍스트가 문자열인지 확인
df['combined_text'] = df['combined_text'].astype(str)


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

# 배치로 임베딩 처리
def process_batch(batch_df, batch_index):
    # 각 문서의 임베딩 생성
    batch_df['embedding'] = batch_df['combined_text'].apply(embed_document)

    # 배치별로 중간 저장 (pickle 파일)
    batch_df[['항목', '제목', '세부인정사항', 'embedding']].to_pickle(f'clinicalbert_embeddings_batch_{batch_index}.pkl')

# 저장된 파일을 확인해 이미 처리된 배치를 건너뛰는 함수
def get_completed_batches():
    # 이미 저장된 배치 파일 목록 가져오기
    completed_batches = set()
    for file in os.listdir():
        if file.startswith("clinicalbert_embeddings_batch_") and file.endswith(".pkl"):
            batch_num = int(file.split("_")[-1].split(".")[0])  # 배치 번호 추출
            completed_batches.add(batch_num)
    return completed_batches

# 배치 처리
completed_batches = get_completed_batches()  # 이미 처리된 배치 확인
for batch_index in range(0, len(df), BATCH_SIZE):
    batch_num = batch_index // BATCH_SIZE  # 배치 번호
    if batch_num in completed_batches:
        print(f"Skipping batch {batch_num} (already completed)...")
        continue  # 이미 처리된 배치는 건너뜀
    batch_df = df.iloc[batch_index:batch_index + BATCH_SIZE]  # 배치 생성
    print(f"Processing batch {batch_num}...")
    process_batch(batch_df, batch_num)
