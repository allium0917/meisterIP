import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

VIDEO_PATH = "133962-758543819_small.mp4"
OUTPUT_PATH = "analyzed_output(1).mp4"   # 💾 분석된 영상 저장 경로
MOVE_THRESHOLD = 1.0
MODEL_WEIGHTS = "yolov8n.pt"
NUM_ROWS, NUM_COLS = 3, 3
FRAME_TIME = 1/30
DURATION_WINDOW = 300

model = YOLO(MODEL_WEIGHTS)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"영상 파일 '{VIDEO_PATH}'를 열 수 없습니다.")

# 영상 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --------------------
# 🟡 VideoWriter 설정
# --------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# 첫 프레임
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("첫 프레임을 읽지 못했습니다.")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
frame_height, frame_width = prev_gray.shape
zone_width = frame_width / NUM_COLS
zone_height = frame_height / NUM_ROWS
dwell_time = [0.0] * (NUM_ROWS * NUM_COLS)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
mag_history = deque(maxlen=DURATION_WINDOW)
motion_history = deque(maxlen=DURATION_WINDOW)
zone_history = deque(maxlen=DURATION_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = bg_subtractor.apply(cur_gray)
    motion_pixels = cv2.countNonZero(fg_mask)
    motion_history.append(motion_pixels)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mean_mag = mag.mean()
    mag_history.append(mean_mag)

    results = model(frame, verbose=False)
    boxes = results[0].boxes
    current_zones = set()

    for b in boxes:
        cls = int(b.cls[0])
        name = model.names[cls]
        if name != "cat":
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0])
        x1 = max(0, min(frame_width-1, x1))
        x2 = max(0, min(frame_width, x2))
        y1 = max(0, min(frame_height-1, y1))
        y2 = max(0, min(frame_height, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        roi_mag = mag[y1:y2, x1:x2]
        if roi_mag.size == 0:
            continue
        cat_mean_mag = float(np.mean(roi_mag))

        if cat_mean_mag > MOVE_THRESHOLD:
            state = "MOVING"
            color = (0,255,0)
        else:
            state = "STATIONARY"
            color = (0,0,255)

        cx, cy = (x1+x2)//2, (y1+y2)//2
        zone_x = int(cx // zone_width)
        zone_y = int(cy // zone_height)
        zone_id = zone_y * NUM_COLS + zone_x
        dwell_time[zone_id] += FRAME_TIME
        current_zones.add(zone_id)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"cat {state} {cat_mean_mag:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (cx, cy), 5, (255,255,0), -1)

    zone_history.append(current_zones)

    avg_mag = np.mean(mag_history) if mag_history else 0
    avg_motion = np.mean(motion_history) if motion_history else 0

    zone_counts = {}
    for zones in list(zone_history)[-int(10/FRAME_TIME):]:
        for z in zones:
            zone_counts[z] = zone_counts.get(z, 0) + 1

    pattern = "Unknown"
    if avg_mag < 0.6 and any(v*FRAME_TIME > 30 for v in zone_counts.values()):
        pattern = "Bored"
    elif avg_mag >= 2.0 and sum(1 for v in zone_counts.values() if v>3) >= 1:
        pattern = "Playing"
    elif avg_motion < 1000:
        pattern = "Resting/Sleeping"

    cv2.putText(frame, f"Behavior Pattern: {pattern}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            cv2.rectangle(frame,
                          (int(c*zone_width), int(r*zone_height)),
                          (int((c+1)*zone_width), int((r+1)*zone_height)),
                          (255,255,255), 1)

    cv2.imshow("Cat Activity + Zone", frame)

    # 🟡 영상 저장
    writer.write(frame)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

    prev_gray = cur_gray.copy()

cap.release()
writer.release()   # 🟡 저장 마무리
cv2.destroyAllWindows()

print("Zone dwell time (sec):", dwell_time)
print(f"저장 완료 ✅ : {OUTPUT_PATH}")