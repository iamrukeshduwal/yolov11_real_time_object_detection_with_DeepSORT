import cv2
import numpy as np
import time
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = "./models/best.pt"
VIDEO_PATH = "./rolling_video/Video_20241001164011269.avi"

# Set the desired input size for YOLO (416x416)
YOLO_INPUT_SIZE = 416

# Set the desired FPS for frame control
DESIRED_FPS = 3  # Slow down to 5 frames per second

def resize_with_padding(image, target_size):
    h, w,_ = image.shape[:3]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded_image = cv2.copyMakeBorder(resized_image, pad_h, target_size - new_h - pad_h,
                                      pad_w, target_size - new_w - pad_w, cv2.BORDER_CONSTANT, value=[128, 128, 128])

    return padded_image, scale, pad_w, pad_h

def correct_bbox(bbox, scale, pad_w, pad_h, original_w, original_h):
    # Correct bounding box by reversing the scaling and padding
    x1 = (bbox[0] - pad_w) / scale
    y1 = (bbox[1] - pad_h) / scale
    x2 = (bbox[2] - pad_w) / scale
    y2 = (bbox[3] - pad_h) / scale

    # Clip the bounding box to ensure it's within image bounds
    x1 = max(0, min(original_w, x1))
    y1 = max(0, min(original_h, y1))
    x2 = max(0, min(original_w, x2))
    y2 = max(0, min(original_h, y2))

    return [int(x1), int(y1), int(x2), int(y2)]

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.4)
    tracker = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    # Frame control using a delay (calculated based on desired FPS)
    frame_delay = int(1000 / DESIRED_FPS)  # Delay in milliseconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_h, original_w = frame.shape[:2]

        # Resize the frame with padding to 416x416
        resized_frame, scale, pad_w, pad_h = resize_with_padding(frame, YOLO_INPUT_SIZE)

        start_time = time.perf_counter()

        # Detect objects on the resized 416x416 frame
        detections = detector.detect(resized_frame)
        tracking_ids, boxes = tracker.track(detections, resized_frame)

        # Draw the bounding boxes on the original frame (resize back)
        for tracking_id, bbox in zip(tracking_ids, boxes):
            corrected_bbox = correct_bbox(bbox, scale, pad_w, pad_h, original_w, original_h)

            # Draw the bounding box and tracking ID on the original frame
            cv2.rectangle(frame, (corrected_bbox[0], corrected_bbox[1]), (corrected_bbox[2], corrected_bbox[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"{str(tracking_id)}", (corrected_bbox[0], corrected_bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print(f"Current fps: {fps}")

        # Show the frame
        cv2.imshow("Frame", frame)

        # Frame control: wait for a keypress or based on desired FPS
        key = cv2.waitKey(frame_delay) & 0xFF  # Slows down frame processing

        # Break the loop if 'q' or 'ESC' is pressed
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
