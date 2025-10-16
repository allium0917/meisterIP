import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드 (작고 빠른 기본 모델)
model = YOLO("yolov8n.pt")

# 🐈 샘플 고양이 영상 (무료 영상 URL)
video_url = "https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi"
cap = cv2.VideoCapture(video_url)

# 첫 프레임 읽기
ret, prev_frame = cap.read()
if not ret:
    print("❌ 영상 불러오기 실패. 인터넷 연결을 확인하세요.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 영상 끝

    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical flow 계산
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # YOLO 객체 감지
    results = model(frame, verbose=False)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = model.names[cls]

        # 🐱 고양이 또는 사람(샘플 영상은 사람)만 감지
        if name in ["cat", "person"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi_mag = mag[y1:y2, x1:x2]
            mean_mag = roi_mag.mean()

            if mean_mag > 1.0:
                state = "Active"
                color = (0, 255, 0)
            else:
                state = "Resting"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}: {state} ({mean_mag:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    cv2.imshow("Object + Optical Flow Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC로 종료
        break

    prev_gray = cur_gray.copy()

cap.release()
cv2.destroyAllWindows()