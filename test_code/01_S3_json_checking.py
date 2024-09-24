import subprocess
import json

def test_load_data_from_s3_with_cli(bucket_name, file_key):
    try:
        # AWS CLI 명령을 사용하여 S3에서 파일 가져오기
        cmd = f"aws s3 cp s3://{bucket_name}/{file_key} -"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"파일 다운로드 실패: {result.stderr}")
            return None
        
        data = result.stdout
        embedded_data = json.loads(data)
        
        # JSON 데이터 확인
        print(f"S3에서 로드한 데이터의 타입: {type(embedded_data)}")
        if isinstance(embedded_data, list):
            print(f"리스트의 길이: {len(embedded_data)}")
            for idx, item in enumerate(embedded_data):
                if not isinstance(item, dict):
                    print(f"인덱스 {idx}에서 비정상적인 데이터 발견: {item}")
                else:
                    missing_keys = [key for key in ['임베딩', '제목', '요약', '세부인정사항'] if key not in item]
                    if missing_keys:
                        print(f"인덱스 {idx}에서 누락된 키 발견: {missing_keys}")
                    else:
                        print(f"인덱스 {idx}의 임베딩 길이: {len(item['임베딩'])}")
                        if len(item['임베딩']) != 1536:
                            print(f"인덱스 {idx}의 임베딩 길이가 예상과 다릅니다.")
        else:
            print("embedded_data가 리스트가 아닙니다.")
        
        return embedded_data
    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
        return None

# 테스트 실행
bucket_name = "hemochat-rag-database"
file_key = "18_aga_tagged_embedded_data.json"
embedded_data = test_load_data_from_s3_with_cli(bucket_name, file_key)
