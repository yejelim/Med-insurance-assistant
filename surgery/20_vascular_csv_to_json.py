import pandas as pd
import json

def csv_to_structured_json(input_csv_path, output_json_path):
    """
    주어진 CSV 파일을 읽어 지정된 구조의 JSON 파일로 변환합니다.
    
    Parameters:
    - input_csv_path: 입력 CSV 파일의 경로
    - output_json_path: 출력 JSON 파일의 경로
    """
    try:
        # CSV 파일 읽기 (인코딩 확인 필요, 보통 UTF-8)
        df = pd.read_csv(input_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 다른 인코딩 시도 (예: 'euc-kr')
        df = pd.read_csv(input_csv_path, encoding='euc-kr')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # '항목', '제목', '세부인정사항' 열이 존재하는지 확인
    required_columns = {'항목', '제목', '세부인정사항'}
    if not required_columns.issubset(df.columns):
        print(f"CSV 파일에 필요한 열이 없습니다. 필요한 열: {required_columns}")
        return

    # '항목'을 기준으로 그룹화
    grouped = df.groupby('항목')

    # 최종 JSON 구조를 담을 리스트
    structured_data = []

    for 항목, group in grouped:
        # 각 '항목'에 해당하는 '제목들' 리스트 생성
        제목들 = []
        for _, row in group.iterrows():
            제목_dict = {
                "제목": row['제목'],
                "세부인정사항": row['세부인정사항']
            }
            제목들.append(제목_dict)
        
        # '항목'과 '제목들'을 포함하는 딕셔너리 생성
        항목_dict = {
            "항목": 항목,
            "제목들": 제목들
        }
        structured_data.append(항목_dict)

    # JSON 파일로 저장 (ensure_ascii=False로 한글이 깨지지 않도록 함)
    try:
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(structured_data, json_file, ensure_ascii=False, indent=4)
        print(f"JSON 파일이 성공적으로 저장되었습니다: {output_json_path}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    # 입력 CSV 파일 경로
    input_csv = 'vascular_filtered_criterion.csv'  # 실제 파일 경로로 변경하세요

    # 출력 JSON 파일 경로
    output_json = 'vascular_filtered_criterion.json'  # 원하는 출력 파일 경로로 변경하세요

    # 함수 호출
    csv_to_structured_json(input_csv, output_json)
