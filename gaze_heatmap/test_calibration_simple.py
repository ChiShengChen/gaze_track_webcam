#!/usr/bin/env python3
"""簡單的校準測試腳本"""
import cv2
import numpy as np
import screeninfo
import time

# 獲取螢幕尺寸
monitor = screeninfo.get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

print(f"螢幕: {screen_width}x{screen_height}")

# 創建窗口
window_name = "calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, screen_width, screen_height)

# 顯示準備畫面
prep = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
cv2.putText(prep, "READY?", (screen_width//2 - 200, screen_height//2),
            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
cv2.imshow(window_name, prep)
cv2.waitKey(2000)

# 顯示9個校準點
points = []
for row in range(3):
    for col in range(3):
        x = int(screen_width * (0.1 + col * 0.4))
        y = int(screen_height * (0.1 + row * 0.4))
        points.append((x, y))

for i, (x, y) in enumerate(points):
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    cv2.circle(canvas, (x, y), 100, (0, 255, 0), -1)
    cv2.putText(canvas, f"{i+1}/9", (x-50, y+50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    print(f"顯示點 {i+1} 在 ({x}, {y})")
    cv2.imshow(window_name, canvas)
    key = cv2.waitKey(2000)
    if key == 27:
        break

cv2.destroyAllWindows()
print("完成")
