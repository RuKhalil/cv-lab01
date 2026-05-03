import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取两张图片
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 2. 创建 ORB 检测器，设置 nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点和计算描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 可视化关键点
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

# 5. 输出关键点数量和描述子维度
print("box.png 关键点数量：", len(kp1))
print("box_in_scene.png 关键点数量：", len(kp2))
print("描述子维度：", des1.shape[1] if des1 is not None else "未生成")

# 6. 保存可视化结果
cv2.imwrite('box_keypoints.png', img1_kp)
cv2.imwrite('box_in_scene_keypoints.png', img2_kp)

# 7. 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1), plt.imshow(img1_kp, cmap='gray'), plt.title('box.png ORB Keypoints')
plt.subplot(1,2,2), plt.imshow(img2_kp, cmap='gray'), plt.title('box_in_scene.png ORB Keypoints')
plt.show()