import pdfplumber
import pandas as pd

# PDF 파일 경로
pdf_path = 'vascular_criterion.pdf'

# '제목' 열에 해당하는 데이터를 저장할 리스트
title_column_data = []

# PDF 파일 열기
with pdfplumber.open(pdf_path) as pdf:
    # 각 페이지를 순회하면서 테이블 추출
    for page in pdf.pages:
        tables = page.extract_tables()  # 페이지에서 테이블 추출
        
        # 추출한 테이블들 순회
        for table in tables:
            # 2열 테이블만 처리
            if len(table[0]) == 2:
                # 첫 번째 행에 '제목'과 '세부인정사항'이 있는지 확인
                header = table[0]
                if '제목' in header[0] and '세부인정사항' in header[1]:
                    # 해당 테이블의 '제목' 열만 추출
                    for row in table[1:]:  # 첫 행(헤더)은 제외
                        title_column_data.append(row[0])  # 첫 번째 열(제목)만 추출

# 추출한 데이터를 데이터프레임으로 변환
df = pd.DataFrame(title_column_data, columns=['제목'])

# 데이터프레임을 CSV 파일로 저장
df.to_csv('vascular_extracted_titles.csv', index=False)

print("CSV 파일로 저장이 완료되었습니다.")
