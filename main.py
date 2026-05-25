#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IoT HW06: Raspberry Pi 5 & YOLOv8 Car 실시간 감지 시스템 (최종 완결본)
YOLOv8 Nano 모델을 활용하여 카메라 영상 속의 자동차를 실시간으로 감지하고 시각화합니다.
라즈베리 파이 5의 rpicam(libcamera) 하드웨어 가속 파이프라인을 완벽하게 지원합니다.
"""

import os
import sys
import time
import argparse
import cv2
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 Car Real-Time Detection for Raspberry Pi 5")
    # rpicam 가속이 확인되었으므로 기본 소스 매핑을 4번으로 고정합니다.
    parser.add_argument("--source", type=str, default="4", help="카메라 입력 소스 (라즈베리파이 5 추천: 4)")
    parser.add_argument("--width", type=int, default=640, help="입력 영상 가로 해상도 (640 또는 320)")
    parser.add_argument("--height", type=int, default=480, help="입력 영상 세로 해상도 (480 또는 240)")
    parser.add_argument("--frame-skip", type=int, default=2, help="성능 가속을 위한 프레임 건너뛰기 간격 (N개 프레임당 1회 YOLO 연산)")
    parser.add_argument("--conf", type=float, default=0.45, help="자동차 감지 임계값 (Confidence Threshold)")
    parser.add_argument("--headless", action="store_true", help="화면 창(cv2.imshow)을 띄우지 않고 터미널에 상태만 표시 (SSH용)")
    parser.add_argument("--save", action="store_true", help="감지 결과 화면을 동영상 파일로 저장합니다.")
    parser.add_argument("--save-path", type=str, default="output.avi", help="결과 동영상 저장 경로")
    return parser.parse_args()


class CarDetectionSystem:
    def __init__(self, args):
        self.args = args
        
        print("\n" + "="*50)
        print(" [*] YOLOv8 Nano 모델 로드 중...")
        # COCO 데이터셋 기반 차량(car=2, bus=5, truck=7) 인식용 YOLOv8 Nano 로드
        self.model = YOLO("yolov8n.pt")
        print(" [*] 모델 로드 완료! (yolov8n.pt)")
        print("="*50 + "\n")

    def process_stream(self):
        source = self.args.source
        
        # =========================================================================
        # [하드웨어 락 해제 핵심] rpicam 구동 전용 GStreamer 가속 파이프라인 구축
        # =========================================================================
        if str(source).isdigit():
            print(f"[*] 라즈베리 파이 5 하드웨어 가속(rpicam) 모드로 카메라 [{source}]번을 오픈합니다.")
            gst_str = (
                f"libcamerasrc ! "
                f"video/x-raw, width={self.args.width}, height={self.args.height} ! "
                f"videoconvert ! appsink"
            )
            cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        else:
            # 비디오 파일 경로 등이 들어왔을 때의 예외 처리
            cap = cv2.VideoCapture(source)
        # =========================================================================

        if not cap.isOpened():
            print(f"[오류] 카메라/비디오 소스 '{self.args.source}'를 최종적으로 열 수 없습니다.")
            sys.exit(1)

        # 해상도 확인 및 보정
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == 0 or actual_h == 0:
            actual_w, actual_h = self.args.width, self.args.height
        print(f"[*] 입력 비디오 크기: {actual_w}x{actual_h}")

        # 비디오 저장을 위한 Writer 설정
        writer = None
        if self.args.save:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(self.args.save_path, fourcc, 15, (actual_w, actual_h))
            print(f"[*] 결과 비디오 저장 파일 활성화: {self.args.save_path}")

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
            
            # 성능 향상을 위해 N번째 프레임마다만 감지 수행
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
                
                if is_detection_frame:
                    print(f"[감지] {item['label']} 발견! 정확도: {item['conf']*100:.1f}%")

            # FPS(초당 프레임 수) 계산
            end_tick = time.time()
            sec = end_tick - start_tick
            fps = 1.0 / sec if sec > 0 else 0
            
            fps_accum += fps
            fps_ticks += 1
            avg_fps = fps_accum / fps_ticks
            
            # 정보 오버레이 그리기
            cv2.putText(
                frame, f"FPS: {fps:.1f} (AVG: {avg_fps:.1f})", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            cv2.putText(
                frame, "Q: Quit", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

            # 비디오 저장 파일 쓰기
            if writer is not None:
                writer.write(frame)

            # GUI 화면 표시
            if not self.args.headless:
                cv2.imshow("Raspberry Pi 5 - YOLOv8 Real-Time Car Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if frame_count % 30 == 0:
                    print(f"[Headless Running] Frame: {frame_count} | Avg FPS: {avg_fps:.1f}")

        # 종료 및 리소스 메모리 해제
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("\n[*] 프로그램이 정상 종료되었습니다. 리소스를 완전히 반환했습니다.")


def main():
    args = parse_arguments()
    
    if args.save:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
    system = CarDetectionSystem(args)
    system.process_stream()


if __name__ == "__main__":
    main()