import sys
import cv2
from ultralytics import YOLO

# 1. 최신 YOLO 모델 로드 (공식 문서 기준)
print("[*] YOLO 모델 로드 중...")
model = YOLO("yolo26n.pt")  # 혹은 기존 쓰시던 yolov8n.pt도 무방합니다.
print("[*] 모델 로드 완료!")

# 2. 순정 비디오 오픈 (도커가 드라이버를 가속 매핑해 주므로 0번으로 다이렉트 개방)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[오류] 카메라 스트림을 열 수 없습니다.")
    sys.exit(1)

print("\n>>> 도커 환경 하드웨어 가속 스트림 시작! (종료하려면 'q')\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[*] 프레임을 읽을 수 없습니다.")
        break
        
    # YOLO 차량 감지 (2: car, 5: bus, 7: truck)
    results = model.predict(frame, classes=[2, 5, 7], conf=0.45, verbose=False)
    
    # 박스 시각화
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
            
    # 화면 출력 (도커 밖 호스트 모니터로 화면이 팝업됩니다)
    cv2.imshow("YOLO Real-Time Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()