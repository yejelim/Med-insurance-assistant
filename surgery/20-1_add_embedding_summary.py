import json
import openai
import numpy as np
import time

# OpenAI API 키 설정
openai.api_key = ""  # 실제 API 키로 변경하세요.

# GPT를 사용한 텍스트 요약 함수
def summarize_text(text, model="gpt-4o-mini"):
    prompt = f"다음 텍스트를 한 문장으로 요약해 주세요:\n\n{text}\n\n요약:"
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,  # 요약이 한 문장으로 되도록 최대 토큰 수를 제한합니다.
            temperature=0.5
        )
        summary = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        summary = ""
    
    return summary

# OpenAI 임베딩 함수 (text-embedding-ada-002)
def embed_text_openai(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        embeddings = response['data'][0]['embedding']
        return np.array(embeddings).tolist()  # 리스트 형태로 변환하여 JSON에 저장 가능하도록 합니다.
    except Exception as e:
        print(f"Error embedding text: {e}")
        return []

# JSON 데이터 로드
with open('vascular_filtered_criterion.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 각 항목의 "세부인정사항"을 요약하여 "요약" 태그로 추가하고, 요약된 결과를 포함하여 임베딩 생성
for index, 항목 in enumerate(data):
    제목들 = 항목.get('제목들', [])
    for j, 제목_dict in enumerate(제목들):
        detail_text = 제목_dict.get('세부인정사항', '')
        제목_text = 제목_dict.get('제목', '')
        if detail_text and 제목_text:
            print(f"Processing 항목 {index + 1}, 제목 {j + 1} of {len(제목들)}...")
            
            # 요약 생성
            summary = summarize_text(detail_text)
            제목_dict['요약'] = summary  # '요약'을 데이터에 추가
            
            # 임베딩을 위한 텍스트 구성 (제목과 세부인정사항을 결합)
            full_text = f"{제목_text}\n{detail_text}"
            
            # 임베딩 생성
            embedding = embed_text_openai(full_text)
            제목_dict['임베딩'] = embedding  # '임베딩'을 데이터에 추가
            
            # API 호출 간 짧은 대기 시간 추가 (Rate Limit 방지)
            time.sleep(1)  # 필요에 따라 조정하세요

# 수정된 데이터를 새로운 JSON 파일로 저장
with open('tagged_vascular_filtered_criterion.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("요약 및 임베딩된 데이터가 파일로 저장되었습니다.")
