import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图片并转为灰度图
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 2. 创建 ORB 检测器
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 4. 进行匹配
matches = bf.match(des1, des2)

# 5. 按匹配距离从小到大排序
matches = sorted(matches, key = lambda x:x.distance)

# 6. 输出总匹配数量
print(f"总匹配数量：{len(matches)}")

# 7. 绘制前50个匹配结果
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

# 8. 保存匹配图
cv2.imwrite('orb_matches.png', img_matches)

# 9. 显示结果
plt.figure(figsize=(15, 8))
plt.imshow(img_matches, cmap='gray')
plt.title('ORB Matches (Top 50)')
plt.axis('off')
plt.show()