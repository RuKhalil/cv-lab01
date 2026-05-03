import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 读取图像 ----------------------
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# ---------------------- ORB 特征检测与匹配 ----------------------
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# ---------------------- 提取对应点 ----------------------
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# ---------------------- RANSAC 计算单应矩阵 ----------------------
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)


# 1. 获取 box.png 的四个角点
h, w = img1.shape
corners = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]]).reshape(-1, 1, 2)

# 2. 投影到场景图
dst_corners = cv2.perspectiveTransform(corners, H)

# 3. 画出目标框
img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
cv2.polylines(img2_color, [np.int32(dst_corners)], True, (0,255,0), 3)

# 4. 显示结果
plt.imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Task4: Target Localization')
plt.savefig('task4_result.png')
plt.show()