import boto3
import pandas as pd
import os
from pdf2image import convert_from_path

# AWS Textract 클라이언트 설정
textract = boto3.client('textract', region_name='ap-northeast-2') 

# PDF를 이미지로 변환
def convert_pdf_to_images(pdf_path, output_dir='images', dpi=300):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f'page_{i + 1}.png')
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
        print(f'Saved {image_path}')
    
    return image_paths

# 이미지 파일을 Amazon Textract로 전송하여 텍스트 추출
def extract_text_from_image(image_path):
    with open(image_path, 'rb') as document:
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
        max_row = max(cell[0] for row in table for cell in row)
        max_column = max(cell[1] for row in table for cell in row)
        
        df = pd.DataFrame(index=range(1, max_row + 1), columns=range(1, max_column + 1))
        
        for row in table:
            for cell in row:
                row_index, column_index, text = cell
                df.iat[row_index - 1, column_index - 1] = text
        
        df.reset_index(drop=True, inplace=True)
        df.columns = [f"Column {i}" for i in df.columns]
        
        dataframes.append(df)
    
    return dataframes

# 데이터프레임을 파일로 저장
def save_dataframes(dataframes, output_dir='output_tables', file_format='csv', page_number=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, df in enumerate(dataframes):
        if file_format == 'csv':
            file_path = os.path.join(output_dir, f"page_{page_number}_table_{i + 1}.csv")
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"Table {i + 1} from page {page_number} saved to {file_path}")
        elif file_format == 'excel':
            file_path = os.path.join(output_dir, f"page_{page_number}_table_{i + 1}.xlsx")
            df.to_excel(file_path, index=False)
            print(f"Table {i + 1} from page {page_number} saved to {file_path}")
        else:
            print(f"Unsupported file format: {file_format}. Skipping table {i + 1}.")

# PDF에서 테이블 추출 및 데이터프레임 저장
def process_pdf(pdf_path, output_dir='output_tables', file_format='csv', image_output_dir='images'):
    # Step 1: PDF를 이미지로 변환
    image_paths = convert_pdf_to_images(pdf_path, output_dir=image_output_dir)
    
    # Step 2: 각 이미지에 대해 Textract 실행 및 테이블 추출
    for page_number, image_path in enumerate(image_paths, start=1):
        response = extract_text_from_image(image_path)
        tables = extract_tables_from_response(response)
        dataframes = tables_to_dataframes(tables)
        
        # 추출된 데이터프레임 저장
        save_dataframes(dataframes, output_dir, file_format, page_number=page_number)

# 실행 부분
if __name__ == "__main__":
    pdf_path = 'general_criterion.pdf'  # 여기에 PDF 파일 경로 입력
    process_pdf(pdf_path, output_dir='extracted_tables', file_format='csv', image_output_dir='converted_images')
