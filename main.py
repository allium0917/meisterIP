import cv2
import numpy as np

height, width = 400, 400

# frame1
frame1 = np.ones((height, width, 3), dtype=np.uint8) * 255
cv2.circle(frame1, (150, 200), 20, (0,0,255), -1)
cv2.imwrite("images/frame1.jpg", frame1)

# frame2
frame2 = np.ones((height, width, 3), dtype=np.uint8) * 255
cv2.circle(frame2, (180, 220), 20, (0,0,255), -1)
cv2.imwrite("images/frame2.jpg", frame2)

print("frame1.jpg, frame2.jpg 생성 완료")