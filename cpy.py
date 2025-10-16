import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# --------------------
# 설정
# --------------------
VIDEO_PATH = "46320-447422988_small.mp4"
MOVE_THRESHOLD = 2.0  # Optical flow magnitude 기준
MODEL_WEIGHTS = "yolov8n.pt"

NUM_ROWS, NUM_COLS = 3, 3  # Zone 분할
FRAME_TIME = 1/30           # 영상 fps 기준 1프레임 시간 (초)
DURATION_WINDOW = 300       # 행동 패턴 평균 계산용 윈도우(frames)

# --------------------
# 모델 로드
# --------------------
model = YOLO(MODEL_WEIGHTS)

# --------------------
# 비디오 열기
# --------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"영상 파일 '{VIDEO_PATH}'를 열 수 없습니다.")

# 첫 프레임 읽기
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("첫 프레임을 읽지 못했습니다.")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
frame_height, frame_width = prev_gray.shape
zone_width = frame_width / NUM_COLS
zone_height = frame_height / NUM_ROWS
dwell_time = [0.0] * (NUM_ROWS * NUM_COLS)

# 배경 차분 초기화
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# 행동 패턴 추적용 큐
mag_history = deque(maxlen=DURATION_WINDOW)
motion_history = deque(maxlen=DURATION_WINDOW)
zone_history = deque(maxlen=DURATION_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------
    # 1. 움직임 감지 (Motion Detection)
    # --------------------
    fg_mask = bg_subtractor.apply(cur_gray)
    motion_pixels = cv2.countNonZero(fg_mask)
    motion_history.append(motion_pixels)

    # --------------------
    # 2. 행동 강도 (Optical Flow)
    # --------------------
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mean_mag = mag.mean()
    mag_history.append(mean_mag)

    # --------------------
    # 3. YOLO 고양이 감지 + Zone 계산
    # --------------------
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

        # 바운딩 내부 optical flow
        roi_mag = mag[y1:y2, x1:x2]
        if roi_mag.size == 0:
            continue
        cat_mean_mag = float(np.mean(roi_mag))

        # 움직임 판단
        if cat_mean_mag > MOVE_THRESHOLD:
            state = "MOVING"
            color = (0,255,0)
        else:
            state = "STATIONARY"
            color = (0,0,255)

        # 객체 중심 좌표
        cx, cy = (x1+x2)//2, (y1+y2)//2
        zone_x = int(cx // zone_width)
        zone_y = int(cy // zone_height)
        zone_id = zone_y * NUM_COLS + zone_x
        dwell_time[zone_id] += FRAME_TIME
        current_zones.add(zone_id)

        # 시각화
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"cat {state} {cat_mean_mag:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (cx, cy), 5, (255,255,0), -1)

    zone_history.append(current_zones)

    # --------------------
    # 4. 행동 패턴 추론
    # --------------------
    avg_mag = np.mean(mag_history) if mag_history else 0
    avg_motion = np.mean(motion_history) if motion_history else 0

    # 동일 Zone 체류 시간 계산 (마지막 10초 기준)
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

    # Zone grid 표시
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            cv2.rectangle(frame,
                          (int(c*zone_width), int(r*zone_height)),
                          (int((c+1)*zone_width), int((r+1)*zone_height)),
                          (255,255,255), 1)

    # --------------------
    # 프레임 표시
    # --------------------
    cv2.imshow("Cat Activity + Zone", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

    prev_gray = cur_gray.copy()

cap.release()
cv2.destroyAllWindows()

# --------------------
# Zone별 체류 시간 출력
# --------------------
print("Zone dwell time (sec):", dwell_time)