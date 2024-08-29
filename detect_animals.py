import cv2
import time
from datetime import datetime
import torch
# AWS SDK (주석 처리)
# import boto3
# from botocore.exceptions import NoCredentialsError

# AWS S3 설정 (주석 처리)
# ACCESS_KEY = 'YOUR_ACCESS_KEY'
# SECRET_KEY = 'YOUR_SECRET_KEY'
# BUCKET_NAME = 'your-bucket-name'

# def upload_to_s3(file_name, bucket, object_name=None):
#     """Upload a file to an S3 bucket"""
#     if object_name is None:
#         object_name = file_name

#     # Upload the file
#     s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
#     try:
#         s3_client.upload_file(file_name, bucket, object_name)
#         print(f"Upload Successful: {file_name}")
#     except FileNotFoundError:
#         print(f"The file was not found: {file_name}")
#     except NoCredentialsError:
#         print("Credentials not available")

# 전이학습한 모델 (모든 클래스 탐지용)
custom_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\DS\Desktop\raccoon_dataset\yolov5\runs\train\exp6\weights\best.pt', trust_repo=True)

def detect_and_capture_animals():
    # 웹캠 시작
    cap = cv2.VideoCapture(0)
    
    last_capture_time = time.time()
    capture_interval = 5  # 5초마다 캡처

    while True:
        # 웹캠에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # 전이학습 모델로 모든 클래스 탐지
        custom_results = custom_model(frame)

        # 탐지 결과 처리
        for i, det in enumerate(custom_results.xyxy[0]):
            cls_id = int(det[-1])
            if cls_id in [15, 16, 17, 91]:  # 새, 고양이, 강아지, 너구리 클래스 ID
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = ('Bird' if cls_id == 15 else 
                         'Cat' if cls_id == 16 else 
                         'Dog' if cls_id == 17 else 
                         'Raccoon')
                
                # 프레임에 탐지된 객체 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 캡처 및 저장
                current_time = time.time()
                if current_time - last_capture_time >= capture_interval:
                    animal_img = frame[y1:y2, x1:x2]
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'captured_{label}_{timestamp}.png'
                    cv2.imwrite(filename, animal_img)
                    print(f"{label} detected and image saved as {filename}!")
                    last_capture_time = current_time

                    # S3에 업로드 (주석 처리)
                    # upload_to_s3(filename, BUCKET_NAME)

        # 프레임 표시
        cv2.imshow('Animal Detector', frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠 해제 및 모든 OpenCV 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_capture_animals()
