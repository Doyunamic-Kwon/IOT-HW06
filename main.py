import sys
import cv2
import numpy as np
from ultralytics import YOLO

# 1. 라즈베리 파이 5 순정 카메라 라이브러리 로드
try:
    from picamera import PiCamera  # 과거 레거시 방지 체크
    from picamera2 import Picamera2
except ImportError:
    print("[오류] Picamera2 라이브러리가 없습니다. 시스템 패키지를 확인하세요.")
    sys.exit(1)

# 2. YOLOv8 모델 로드
print("[*] YOLOv8 모델 로드 중...")
model = YOLO("yolov8n.pt")
print("[*] 모델 로드 완료!")

# 3. Picamera2 카메라 객체 생성 및 해상도 지정
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

print("\n>>> 파이 5 순정 Picamera2 스트림 시작! (종료하려면 'q')\n")

try:
    while True:
        # 4. 하드웨어 가속 통로에서 실시간 프레임을 스냅샷(numpy 배열)으로 가져오기
        frame = picam2.capture_array()
        
        # Picamera2는 기본 RGB 포맷이므로 OpenCV 시각화를 위해 BGR로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 5. YOLOv8 차량 감지 (2: car, 5: bus, 7: truck)
        results = model.predict(frame, classes=[2, 5, 7], conf=0.45, verbose=False)

        # 6. 감지된 자동차 박스 그리기
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)

        # 7. 화면에 디스플레이
        cv2.imshow("Raspberry Pi 5 - YOLOv8 Real-Time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 8. 카메라 자원 반환
    picam2.stop()
    cv2.destroyAllWindows()
    print("[*] 시스템이 안전하게 종료되었습니다.")