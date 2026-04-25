import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import cv2
import torch

# Keep Ultralytics settings/cache local to this project folder.
YOLO_CONFIG_DIR = Path(__file__).resolve().parent / ".ultralytics"
YOLO_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

from ultralytics import YOLO


PERSON_CLASS_ID = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect whether a human is visible in a live webcam feed."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path/name. yolov8n.pt is small and fast.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum person confidence required to count as human detected.",
    )
    parser.add_argument("--width", type=int, default=960, help="Camera frame width.")
    parser.add_argument("--height", type=int, default=540, help="Camera frame height.")
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="FPS requested from the webcam. Actual FPS depends on the camera.",
    )
    parser.add_argument(
        "--inference-size",
        type=int,
        default=320,
        help="YOLO inference image size. Lower is faster; 320 is good for CPU.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. Use cuda only if PyTorch detects an NVIDIA GPU.",
    )
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device_arg == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detect_people(model, frame, confidence_threshold, inference_size, device, use_half):
    results = model.predict(
        frame,
        imgsz=inference_size,
        conf=0.10,
        classes=[PERSON_CLASS_ID],
        device=device,
        half=use_half,
        verbose=False,
    )

    boxes = []
    best_confidence = 0.0

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            best_confidence = max(best_confidence, confidence)

            if confidence < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, confidence))

    return boxes, best_confidence, time.perf_counter()


def draw_status_overlay(frame, human_detected, best_confidence, fps, inference_fps):
    overlay = frame.copy()
    height, width = frame.shape[:2]

    panel_height = 86
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    status = "HUMAN DETECTED" if human_detected else "NO HUMAN"
    status_color = (40, 220, 80) if human_detected else (50, 80, 255)
    confidence_text = f"Prediction: {best_confidence * 100:.1f}%"

    cv2.putText(
        frame,
        status,
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        confidence_text,
        (24, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    fps_text = f"FPS: {fps:.1f} | AI: {inference_fps:.1f}"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.putText(
        frame,
        fps_text,
        (width - fps_size[0] - 24, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )


def draw_person_box(frame, box):
    x1, y1, x2, y2, confidence = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)

    label = f"Human {confidence * 100:.1f}%"
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    label_y = max(y1 - 10, label_size[1] + 10)

    cv2.rectangle(
        frame,
        (x1, label_y - label_size[1] - baseline - 8),
        (x1 + label_size[0] + 12, label_y + baseline - 4),
        (40, 220, 80),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 6, label_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    print("Loading YOLO model...")
    model = YOLO(args.model)
    device = choose_device(args.device)
    use_half = device == "cuda"
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    if not cap.isOpened():
        print(
            "Error: Webcam not found. Try --camera 1, close other camera apps, "
            "or check webcam permissions."
        )
        return

    print("Webcam started.")
    print("Press q to quit.")

    previous_frame_time = time.perf_counter()
    previous_inference_time = time.perf_counter()
    inference_fps = 0.0
    latest_boxes = []
    latest_confidence = 0.0
    pending_detection = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Could not read frame from webcam.")
                break

            if pending_detection is None:
                pending_detection = executor.submit(
                    detect_people,
                    model,
                    frame.copy(),
                    args.confidence,
                    args.inference_size,
                    device,
                    use_half,
                )
            elif pending_detection.done():
                latest_boxes, latest_confidence, completed_at = pending_detection.result()
                inference_fps = 1.0 / max(completed_at - previous_inference_time, 0.001)
                previous_inference_time = completed_at
                pending_detection = executor.submit(
                    detect_people,
                    model,
                    frame.copy(),
                    args.confidence,
                    args.inference_size,
                    device,
                    use_half,
                )

            for box in latest_boxes:
                draw_person_box(frame, box)

            human_detected = latest_confidence >= args.confidence

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_frame_time, 0.001)
            previous_frame_time = current_time

            draw_status_overlay(frame, human_detected, latest_confidence, fps, inference_fps)

            cv2.imshow("AI Human Webcam Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
