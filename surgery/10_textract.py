# To extract text from table pdf by Amazon textract

import boto3
import pandas as pd
import os

# AWS Textract 클라이언트 설정
textract = boto3.client('textract', region_name='ap-northeast-2') 

# PDF 파일을 Amazon Textract로 전송하여 텍스트 추출
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as document:
        response = textract.analyze_document(
            Document={'Bytes': document.read()},
            FeatureTypes=['TABLES']
        )
    return response

# Amazon Textract의 응답에서 테이블 데이터 추출 및 데이터프레임으로 변환
def extract_tables_from_response(response):
    tables = []
    
    for block in response['Blocks']:
        if block['BlockType'] == 'TABLE':
            table = []
            # 해당 테이블의 모든 셀 정보를 수집
            for relationship in block.get('Relationships', []):
                if relationship['Type'] == 'CHILD':
                    cells = []
                    for cell_id in relationship['Ids']:
                        cell_block = next((b for b in response['Blocks'] if b['Id'] == cell_id), None)
                        if cell_block and cell_block['BlockType'] == 'CELL':
                            row_index = cell_block['RowIndex']
                            column_index = cell_block['ColumnIndex']
                            text = ''
                            if 'Relationships' in cell_block:
                                for cell_relationship in cell_block['Relationships']:
                                    if cell_relationship['Type'] == 'CHILD':
                                        for word_id in cell_relationship['Ids']:
                                            word_block = next((b for b in response['Blocks'] if b['Id'] == word_id), None)
                                            if word_block and word_block['BlockType'] == 'WORD':
                                                text += word_block.get('Text', '') + ' '
                            cells.append((row_index, column_index, text.strip()))
                    if cells:
                        table.append(cells)
            if table:
                tables.append(table)
    return tables

# 데이터프레임으로 변환
def tables_to_dataframes(tables):
    dataframes = []
    
    for table in tables:
        # 테이블의 최대 행/열 찾기
        max_row = max(cell[0] for row in table for cell in row)
        max_column = max(cell[1] for row in table for cell in row)
        
        # 빈 데이터프레임 생성
        df = pd.DataFrame(index=range(1, max_row + 1), columns=range(1, max_column + 1))
        
        # 테이블 데이터 채우기
        for row in table:
            for cell in row:
                row_index, column_index, text = cell
                df.iat[row_index - 1, column_index - 1] = text
        
        # 인덱스와 컬럼 번호를 제거하고 기본 0부터 시작하는 인덱스로 설정
        df.reset_index(drop=True, inplace=True)
        df.columns = [f"Column {i}" for i in df.columns]
        
        dataframes.append(df)
    
    return dataframes

# 데이터프레임을 파일로 저장
def save_dataframes(dataframes, output_dir='output_tables', file_format='csv'):
    """
    dataframes: list of pandas DataFrame objects
    output_dir: 디렉토리 이름 (기본값: 'output_tables')
    file_format: 'csv' 또는 'excel' 중 선택
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, df in enumerate(dataframes):
        if file_format == 'csv':
            file_path = os.path.join(output_dir, f"table_{i + 1}.csv")
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"Table {i + 1} saved to {file_path}")
        elif file_format == 'excel':
            file_path = os.path.join(output_dir, f"table_{i + 1}.xlsx")
            df.to_excel(file_path, index=False)
            print(f"Table {i + 1} saved to {file_path}")
        else:
            print(f"Unsupported file format: {file_format}. Skipping table {i + 1}.")

# PDF에서 테이블 추출 및 데이터프레임 저장
def process_pdf(pdf_path, output_dir='output_tables', file_format='csv'):
    response = extract_text_from_pdf(pdf_path)
    tables = extract_tables_from_response(response)
    dataframes = tables_to_dataframes(tables)
    
    # 추출된 데이터프레임 확인
    for i, df in enumerate(dataframes):
        print(f"Table {i + 1}")
        print(df)
    
    # 데이터프레임을 파일로 저장
    save_dataframes(dataframes, output_dir, file_format)

# 실행 부분
if __name__ == "__main__":
    pdf_path = 'general_criterion.pdf'  # 여기에 PDF 파일 경로 입력
    process_pdf(pdf_path, output_dir='extracted_tables', file_format='csv')  # 'csv' 또는 'excel' 선택
