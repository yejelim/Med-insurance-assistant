import os
import json
import boto3
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수에서 자격 증명 가져오기
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

# Boto3 클라이언트 설정
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# S3 객체 다운로드 및 로드
s3_uri = 's3://hemochat-rag-database/18_aga_tagged_embedded_data.json'
bucket_name = s3_uri.split('/')[2]
object_key = '/'.join(s3_uri.split('/')[3:])

try:
    obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    data = json.loads(obj['Body'].read())

    print("JSON 파일의 첫 5개 항목:")
    print(json.dumps(data[:5], indent=4, ensure_ascii=False))
    
except Exception as e:
    print(f"Error fetching the file: {e}")
