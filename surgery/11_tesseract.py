import pytesseract
import pandas as pd
import os
import subprocess

pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# 이미지 폴더 및 결과 저장 폴더 설정
image_folder = 'converted_images'
output_folder = 'tesseract_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 파일 목록 가져오기
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

# 디버그 로그 파일 초기화
debug_log_path = 'tesseract_debug_log.txt'
with open(debug_log_path, 'w') as debug_log:
    debug_log.write('Tesseract Debug Log\n')
    debug_log.write('=' * 50 + '\n')

# 각 이미지에서 한글 텍스트 추출 및 CSV로 저장
for i, image_file in enumerate(image_files, start=1):
    image_path = os.path.join(image_folder, image_file)
    
    # Tesseract를 이용하여 한글 텍스트 추출
    try:
        # 디버그 모드로 Tesseract 실행
        command = f'{pytesseract.pytesseract.tesseract_cmd} {image_path} stdout -l kor --tessdata-dir /usr/local/share/tessdata'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # 디버그 로그 저장
        with open(debug_log_path, 'a') as debug_log:
            debug_log.write(f'Processing file: {image_file}\n')
            debug_log.write(result.stderr)  # stderr에 디버그 정보가 저장됨
            debug_log.write('\n' + '=' * 50 + '\n')
        
        # 결과 확인 및 에러가 있으면 중단
        if result.returncode != 0:
            print(f"Error occurred while processing {image_file}. Check debug log for details.")
            break

        # 추출된 텍스트를 데이터프레임에 저장 (각 줄을 행으로 분리)
        extracted_text = result.stdout
        lines = extracted_text.strip().split('\n')
        df = pd.DataFrame(lines, columns=['Extracted Text'])
        
        # CSV 파일로 저장
        output_path = os.path.join(output_folder, f'page_{i}.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"Extracted text from {image_file} saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        break
