import sys
import cv2
import os
from ultralytics import YOLO

# 1. 원본 모델명 정의 (자동차 인식을 위해 nano 규격 사용)
PT_MODEL_PATH = "yolov8n.pt"
NCNN_MODEL_PATH = "yolov8n_ncnn_model"

print("\n" + "="*50)
print(" [*] 라즈베리 파이 5 하드웨어 가속 체크 중...")

# NCNN 가속 모델이 없다면 최초 1회 자동 변환 및 저장
if not os.path.exists(NCNN_MODEL_PATH):
    print(" [*] 경량화 NCNN 가속 모델이 없습니다. 변환을 시작합니다 (최초 1회 소요)...")
    pt_model = YOLO(PT_MODEL_PATH)
    pt_model.export(format="ncnn", imgsz=640)
    print(" [*] NCNN 변환 완료!")

# 2. 가속 완료된 NCNN 경량 모델
model = YOLO(NCNN_MODEL_PATH, task="detect")
print(" [*] 하드웨어 가속 모델 탑재 완료!")
print("="*50 + "\n")

# 3. 비디오 스트림 개방
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[오류] 카메라를 열 수 없습니다.")
    sys.exit(1)

print("차량 감지 시작 (종료: 'q')\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # YOLOv8 차량 및 번호판 유관 객체 감지 (2: car, 5: bus, 7: truck)
    results = model.predict(source=frame, classes=[2, 5, 7], conf=0.35, verbose=False)

    # 감지 박스 시각화
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # 레이블
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {box.conf[0]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Raspberry Pi 5 - Fast YOLO Real-Time", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[*] 시스템이 정상 종료되었습니다.")