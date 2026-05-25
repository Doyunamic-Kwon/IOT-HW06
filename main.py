#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IoT HW06: Raspberry Pi 5 & YOLOv8 Car 실시간 감지 시스템 (순정 완전판)
리눅스 표준 V4L2 인덱스 방식으로 복잡한 가속 파이프라인 없이 가장 직관적으로 카메라를 오픈합니다.
"""

import os
import sys
import time
import argparse
import cv2
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 Car Real-Time Detection for Raspberry Pi 5")
    # rpicam-hello가 열렸으므로 기본 소스 매핑을 0번 또는 4번으로 접근합니다.
    parser.add_argument("--source", type=str, default="0", help="카메라 입력 소스 (기본값: 0)")
    parser.add_argument("--width", type=int, default=640, help="입력 영상 가로 해상도")
    parser.add_argument("--height", type=int, default=480, help="입력 영상 세로 해상도")
    parser.add_argument("--frame-skip", type=int, default=2, help="성능 가속을 위한 프레임 건너뛰기 간격")
    parser.add_argument("--conf", type=float, default=0.45, help="자동차 감지 임계값 (Confidence Threshold)")
    parser.add_argument("--headless", action="store_true", help="화면 창을 띄우지 않고 터미널에 상태만 표시")
    return parser.parse_args()


class CarDetectionSystem:
    def __init__(self, args):
        self.args = args
        
        print("\n" + "="*50)
        print(" [*] YOLOv8 Nano 모델 로드 중...")
        self.model = YOLO("yolov8n.pt")
        print(" [*] 모델 로드 완료! (yolov8n.pt)")
        print("="*50 + "\n")

    def process_stream(self):
        source = self.args.source
        
        # =========================================================================
        # [수정] 이상한 가속 파이프라인 전부 제거, 오직 순정 리눅스 V4L2 인덱스 오픈!
        # =========================================================================
        if str(source).isdigit():
            camera_index = int(source)
            print(f"[*] 순정 리눅스 V4L2 백엔드로 카메라 [{camera_index}]번을 직접 오픈합니다.")
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        else:
            # 동영상 파일 등 문자열 경로가 들어왔을 때 처리
            cap = cv2.VideoCapture(source)
        # =========================================================================

        if not cap.isOpened():
            print(f"[오류] 카메라 소스 '{source}'를 최종적으로 열 수 없습니다.")
            print("[팁] --source 0 또는 --source 4 로 인덱스를 변경해서 실행해보세요.")
            sys.exit(1)

        # 해상도 및 영상 크기 강제 세팅
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[*] 비디오 오픈 성공! 입력 크기: {actual_w}x{actual_h}")

        frame_count = 0
        fps_accum = 0
        fps_ticks = 0
        cached_boxes = []

        print("\n>>> 실시간 차량 감지를 시작합니다! 키보드 'q'를 누르면 종료됩니다.\n")

        while True:
            start_tick = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[*] 비디오 스트림이 종료되었거나 프레임을 읽을 수 없습니다.")
                break

            frame_count += 1
            is_detection_frame = (frame_count % self.args.frame_skip == 0)

            if is_detection_frame:
                # 2: car (승용차), 5: bus (버스), 7: truck (트럭)
                results = self.model.predict(
                    frame, 
                    classes=[2, 5, 7], 
                    conf=self.args.conf, 
                    verbose=False
                )
                
                boxes_in_frame = []
                for result in results:
                    for box in result.boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().item()
                        cls = int(box.cls[0].cpu().item())
                        label = self.model.names[cls]
                        boxes_in_frame.append({"box": coords, "conf": conf, "label": label})
                
                cached_boxes = boxes_in_frame

            # 감지된 차량 박스 시각화 (형광 초록색)
            for item in cached_boxes:
                x1, y1, x2, y2 = item["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                text = f"{item['label']} {item['conf']*100:.1f}%"
                cv2.putText(
                    frame, text, (x1, y1 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 127), 2
                )

            # FPS 계산 및 오버레이
            end_tick = time.time()
            sec = end_tick - start_tick
            fps = 1.0 / sec if sec > 0 else 0
            
            fps_accum += fps
            fps_ticks += 1
            avg_fps = fps_accum / fps_ticks
            
            cv2.putText(
                frame, f"FPS: {fps:.1f} (AVG: {avg_fps:.1f})", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

            # GUI 화면 표시
            if not self.args.headless:
                cv2.imshow("YOLOv8 Real-Time Car Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("\n[*] 프로그램이 정상 종료되었습니다. 리소스를 완전히 반환했습니다.")


def main():
    args = parse_arguments()
    system = CarDetectionSystem(args)
    system.process_stream()


if __name__ == "__main__":
    main()