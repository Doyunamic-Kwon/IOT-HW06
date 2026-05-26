import sys
import cv2
from ultralytics import YOLO

print("\n[*] YOLOv8 Nano 모델 로드 완료!")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[오류] 카메라 스트링을 열 수 없습니다.")
    sys.exit(1)

print(">>> 차량 감지 시작! (종료하려면 'q')\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # CPU 부하를 없애기 위해 연산은 320x240으로 낮춰서 수행 (렉 제거)
    small_frame = cv2.resize(frame, (320, 240))
    results = model.predict(source=small_frame, classes=[2, 5, 7], conf=0.4, verbose=False)

    # 감지된 박스를 원래 640x480 화면 크기에 맞게 2배 튀겨서 매핑
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * 2).astype(int)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {box.conf[0]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # 최종 디스플레이 팝업
    cv2.imshow("Raspberry Pi 5 - YOLO Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("quit")