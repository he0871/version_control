import cv2
from ultralytics import YOLO
import time

# 0 = default webcam. Replace with RTSP/HTTP/video file if needed.
SOURCE = 0
SAVE_DIR = "./screenshot"

model = YOLO("yolov8n.pt")  # small & fast
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Failed to open video source: {SOURCE}")

print("Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Run detection
    results = model(frame, verbose=False)[0]

    # Filter to "person" class only (COCO class id 0)
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id not in [0, 2]:       # 0 == person, 2 == car
                continue
            if conf < 0.37:       # confidence threshold
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if cls_id == 0:
                cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif cls_id == 2:
                cv2.putText(frame, f"car {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/screen_{ts}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[Saved] {filename}")
            time.sleep(0.3)

    cv2.imshow("People Only", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()