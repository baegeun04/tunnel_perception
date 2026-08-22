from ultralytics import YOLO

# 가벼운 nano 모델로 시작 (CPU 학습에 적합)
model = YOLO('yolov8n.pt')

# 학습 실행
model.train(
    data='Fire-Smoke--1/data.yaml',  # 데이터셋 설정 파일
    epochs=30,        # 반복 횟수 (CPU라 30으로 시작)
    imgsz=416,        # 이미지 크기 (작게 = 빠름)
    batch=8,          # 한 번에 처리할 이미지 수
    device='cpu',     # CPU로 학습
    project='fire_train',   # 결과 저장 폴더
    name='exp1',      # 실험 이름
    patience=10,      # 10번 개선 없으면 조기 종료
)

print("학습 완료! 결과: fire_train/exp1/weights/best.pt")
