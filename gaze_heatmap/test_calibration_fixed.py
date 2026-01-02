#!/usr/bin/env python3
"""測試修復後的校準顯示"""
import cv2
import numpy as np
import screeninfo
import time

# 獲取螢幕尺寸
monitor = screeninfo.get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

print(f"螢幕: {screen_width}x{screen_height}")

# 創建窗口 - 不使用全屏
window_name = "test_calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, screen_width, screen_height)
cv2.moveWindow(window_name, 0, 0)

# 準備畫面
prep = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
cv2.putText(prep, "TEST - Press any key", (screen_width//2 - 300, screen_height//2 - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
cv2.putText(prep, "Press ESC or 'q' to exit", (screen_width//2 - 300, screen_height//2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
cv2.imshow(window_name, prep)

print("\n顯示準備畫面...")
print("按任意鍵開始，ESC 或 'q' 退出")
start = time.time()
key = -1
while time.time() - start < 10:
    k = cv2.waitKey(100) & 0xFF
    if k != 255:
        key = k
        break
    if k == 27 or k == ord('q'):
        print("退出")
        cv2.destroyAllWindows()
        exit(0)

if key == 27 or key == ord('q'):
    print("退出")
    cv2.destroyAllWindows()
    exit(0)

# 顯示一個測試點
canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
x, y = screen_width // 2, screen_height // 2
cv2.circle(canvas, (x, y), 100, (0, 255, 0), -1)
cv2.putText(canvas, "GREEN DOT - Press ESC or 'q' to exit", 
            (x - 300, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
print(f"\n顯示測試點在 ({x}, {y})")
print("應該看到：黑底 + 綠色大圓圈")
print("按 ESC 或 'q' 退出")

cv2.imshow(window_name, canvas)
start = time.time()
while time.time() - start < 10:
    k = cv2.waitKey(100) & 0xFF
    if k == 27 or k == ord('q'):
        print("\n✅ 成功退出！")
        break

cv2.destroyAllWindows()
print("完成")
