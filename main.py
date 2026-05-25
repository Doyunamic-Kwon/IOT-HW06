import sys
import cv2
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("yolov8n.pt")

# 2. 진짜 실시간 스트림 카메라 오픈 (0번 노드)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# [핵심] 꼬이지 않게 OpenCV 표준 포맷(YUYV)과 해상도를 강제로 꽂아넣음
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[오류] 카메라를 열 수 없습니다.")
    sys.exit(1)

print(">>> 진짜 실시간 스트림 인식 시작 (종료하려면 'q')")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[*] 프레임을 읽을 수 없습니다. 스트림 튕김")
        break
        
    # YOLOv8 차량 감지 (2프레임당 1번씩 가볍게 돌리기)
    results = model.predict(frame, classes=[2, 5, 7], conf=0.45, verbose=False)
    
    # 감지 박스 그리기
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    # 화면 표시
    cv2.imshow("YOLOv8 Real-Time Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()