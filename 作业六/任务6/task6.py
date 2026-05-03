import cv2
import numpy as np

# 读取图片
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 测试参数列表
nfeatures_list = [500, 1000, 2000]

print("nfeatures | 模板图关键点 | 场景图关键点 | 匹配数量 | RANSAC内点数 | 内点比例 | 是否成功定位")
print("-" * 90)

for nfeatures in nfeatures_list:
    # 1. ORB特征检测
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 2. 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 3. 提取对应点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 4. RANSAC计算单应矩阵
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inlier_count = int(np.sum(mask))
    total_matches = len(matches)
    inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0

    # 5. 目标定位（判断是否成功）
    h, w = img1.shape
    corners = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]]).reshape(-1,1,2)
    dst_corners = cv2.perspectiveTransform(corners, H)

    # 判断定位是否成功
    success = True
    h2, w2 = img2.shape
    for pt in dst_corners:
        x, y = pt[0]
        if x < 0 or x > w2 or y < 0 or y > h2:
            success = False
            break

    # 输出结果
    print(f"{nfeatures:>8} | {len(kp1):>12} | {len(kp2):>12} | {total_matches:>8} | {inlier_count:>12} | {inlier_ratio:.4f} | {'是' if success else '否'}")