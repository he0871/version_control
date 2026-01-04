import cv2
from ultralytics import YOLO
import numpy as np
import webbrowser

# ==============================
# Config
# ==============================
SOURCE = 0
CONF_THRESH = 0.4
MODEL_PATH = "yolo11n-pose.pt"

# COCO keypoints
L_SHOULDER = 5
R_SHOULDER = 6
L_WRIST = 9
R_WRIST = 10

# ==============================
# Helper Functions
# ==============================
def is_hand_raised(kpts, conf, side="right", threshold=0.4):
    """
    Returns True if hand is above shoulder
    """
    if side == "right":
        shoulder, wrist = R_SHOULDER, R_WRIST
    else:
        shoulder, wrist = L_SHOULDER, L_WRIST

    if conf[shoulder] < threshold or conf[wrist] < threshold:
        return False

    # y-axis goes DOWN → smaller = higher
    return kpts[wrist][1] < kpts[shoulder][1]


def detect_gesture(kpts, conf):
    right = is_hand_raised(kpts, conf, "right")
    left = is_hand_raised(kpts, conf, "left")

    if right and left:
        return "🙌 BOTH HANDS UP"
    elif right:
        return "👉 RIGHT HAND UP"
    elif left:
        return "👈 LEFT HAND UP"
    return None

def openbrowser():
    

    url = "https://weather.com/weather/today/l/abb260ed659872b115ea85d2524618ccc611b2bb5226a079f44277beb847c9b8"

    # This line attempts to open the URL in a new tab of the default browser
    # 'new=2' specifies opening a new tab
    webbrowser.open(url, new=2) 

# ==============================
# Main
# ==============================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera")

print("Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, verbose=False)[0]

    if results.boxes is not None and results.keypoints is not None:
        for i, box in enumerate(results.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0 or conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            kpts = results.keypoints.xy[i].cpu().numpy()
            kconf = results.keypoints.conf[i].cpu().numpy()

            gesture = detect_gesture(kpts, kconf)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Person {conf:.2f}"
            if gesture:
                label += f" | {gesture}"
                openbrowser()

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()