import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img = cv2.imread("paper.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# ===================== 专为你这张原图写的精准四个角点 =====================
pts_src = np.float32([
    [620,  950],   # 左上角
    [2350, 980],   # 右上角
    [550,  3250],  # 左下角
    [2450, 3200]   # 右下角
])

# 标准正视目标尺寸
tw, th = 1000, 1414
pts_dst = np.float32([
    [0, 0],
    [tw, 0],
    [0, th],
    [tw, th]
])
# ======================================================================

# 透视变换
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
img_corrected = cv2.warpPerspective(img, M, (tw, th))

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img), plt.title("Original"), plt.axis('off')
plt.subplot(122), plt.imshow(img_corrected), plt.title("Front View (Correct)"), plt.axis('off')
plt.savefig("perspective_correction.png")
plt.show()