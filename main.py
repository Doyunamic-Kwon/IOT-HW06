import os
import cv2
import time
from ultralytics import YOLO

print("\n[*] YOLOv8 Nano 모델 로드 완료!")
model = YOLO("yolov8n.pt")

print(">>> [최종 마스터] 실시간 차량 감지 시작 (종료: Ctrl+C)")


IMG_PATH = "/dev/shm/temp.jpg"

try:
    while True:
        # 2. 파이 5 하드웨어 명령어로 초고속 스냅샷 (640x480 해상도 고정, 프리뷰 끔)
        os.system(f"rpicam-still -t 1 --width 640 --height 480 --output {IMG_PATH} --immediate --nopreview > /dev/null 2>&1")
        
        # 3. 캡처된 이미지 읽기
        frame = cv2.imread(IMG_PATH)
        if frame is None:
            continue
            
        # 4. YOLOv8 차량 감지 (2: car, 5: bus, 7: truck)
        results = model.predict(frame, classes=[2, 5, 7], conf=0.4, verbose=False)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {box.conf[0]:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
        # 6. 화면 출력
        cv2.imshow("Raspberry Pi 5 - YOLO Real-Time", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[*] 사용자가 프로그램을 종료했습니다.")

finally:
    cv2.destroyAllWindows()
    if os.path.exists(IMG_PATH):
        os.remove(IMG_PATH)