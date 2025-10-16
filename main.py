import cv2
import numpy as np
from ultralytics import YOLO

# --------------------
# 설정
# --------------------
VIDEO_PATH = "46320-447422988_small.mp4"  # 로컬 영상 파일 경로
MOVE_THRESHOLD = 1.0         # 움직임 평균 magnitude 기준
MODEL_WEIGHTS = "yolov8n.pt" # YOLO 모델

# --------------------
# YOLO 모델 로드
# --------------------
model = YOLO(MODEL_WEIGHTS)

# --------------------
# 비디오 열기 (저장된 영상 전용)
# --------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"영상 파일 '{VIDEO_PATH}'를 열 수 없습니다.")

# 첫 프레임 읽기
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("첫 프레임을 읽지 못했습니다.")
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# --------------------
# 프레임 반복 처리
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 영상 끝

    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow 계산 (Farneback)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # YOLO로 객체 감지 (한 프레임마다)
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    # 각 바운딩박스 처리
    for b in boxes:
        cls = int(b.cls[0])
        name = model.names[cls]
        if name != "cat":
            continue  # 고양이만 처리

        # 바운딩박스 좌표
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        x1 = max(0, min(frame.shape[1]-1, x1))
        x2 = max(0, min(frame.shape[1], x2))
        y1 = max(0, min(frame.shape[0]-1, y1))
        y2 = max(0, min(frame.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # 바운딩 내부 optical flow magnitude 평균 계산
        roi_mag = mag[y1:y2, x1:x2]
        if roi_mag.size == 0:
            continue
        mean_mag = float(np.mean(roi_mag))

        # 움직임 판단
        if mean_mag > MOVE_THRESHOLD:
            state = "MOVING"
            color = (0, 255, 0)
        else:
            state = "STATIONARY"
            color = (0, 0, 255)

        # 시각화
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"cat {state} {mean_mag:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 화면 출력
    cv2.imshow("Moving Cats Only", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC로 종료
        break

    prev_gray = cur_gray.copy()

cap.release()
cv2.destroyAllWindows()