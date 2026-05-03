import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 创建ORB检测器
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 创建暴力匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# -------------------------- 2. 提取对应点坐标 --------------------------
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# -------------------------- 3. 使用RANSAC估计单应矩阵并剔除错误匹配 --------------------------
# 计算Homography矩阵，使用RANSAC，重投影误差阈值设为5.0
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# 提取内点匹配
matchesMask = mask.ravel().tolist()
inlier_matches = [matches[i] for i in range(len(matches)) if matchesMask[i] == 1]

# -------------------------- 4. 输出结果 --------------------------
total_matches = len(matches)
inlier_count = len(inlier_matches)
inlier_ratio = inlier_count / total_matches

print("Homography矩阵：")
print(H)
print(f"总匹配数量：{total_matches}")
print(f"RANSAC内点数量：{inlier_count}")
print(f"内点比例：{inlier_ratio:.4f}")

# -------------------------- 5. 可视化RANSAC后的内点匹配 --------------------------
draw_params = dict(matchColor=(0, 255, 0),  # 匹配线颜色为绿色
                   singlePointColor=None,
                   matchesMask=matchesMask,  # 只绘制内点
                   flags=2)

img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

cv2.imwrite('orb_ransac_matches.png', img_ransac)

plt.figure(figsize=(15, 8))
plt.imshow(img_ransac, cmap='gray')
plt.title('ORB Matches after RANSAC')
plt.axis('off')
plt.show()